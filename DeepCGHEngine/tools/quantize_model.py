#!/usr/bin/env python3
"""
quantize_model.py — INT8 quantization for DeepCGH ONNX models.

Uses onnxruntime.quantization to convert FP32 ONNX models to INT8,
with static calibration using synthetic RGB-D data.

Usage:
    # Default: quantize the 256x256 model
    py -3.10 quantize_model.py

    # Custom paths
    py -3.10 quantize_model.py --input model.onnx --output model_int8.onnx

    # With calibration samples and mode selection
    py -3.10 quantize_model.py --calibration-samples 200 --mode static

    # QDQ mode (default) vs QOps mode
    py -3.10 quantize_model.py --quant-format QDQ
    py -3.10 quantize_model.py --quant-format QOps
"""

import argparse
import os
import sys
import tempfile
import numpy as np

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import (
        QuantFormat,
        QuantType,
        CalibrationDataReader,
        quantize_static,
        quantize_dynamic,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("Install with: pip install onnx onnxruntime")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Calibration Data Reader — generates synthetic RGB-D data
# ---------------------------------------------------------------------------

class DeepCGHCalibrationReader(CalibrationDataReader):
    """Generates synthetic calibration data matching the DeepCGH input shape.

    The model expects input 'target' with shape [1, H, W, num_planes],
    which is the preprocessed RGB-D volume (color + depth blended across
    depth planes). We generate random float32 data in [0, 1] to cover
    the typical input range after preprocessing.
    """

    def __init__(self, model_path: str, num_samples: int = 100,
                 input_range: tuple = (0.0, 1.0)):
        self.num_samples = num_samples
        self.input_range = input_range
        self.current_idx = 0

        # Read input shape from the model
        session = ort.InferenceSession(model_path)
        input_info = session.get_inputs()[0]
        self.input_name = input_info.name
        self.input_shape = input_info.shape

        # Replace dynamic dimensions with concrete values (default 256)
        self.input_shape = [
            1 if isinstance(d, str) else d for d in self.input_shape
        ]

    def get_next(self):
        if self.current_idx >= self.num_samples:
            return None

        lo, hi = self.input_range
        data = np.random.uniform(lo, hi, self.input_shape).astype(np.float32)
        self.current_idx += 1
        return {self.input_name: data}

    def rewind(self):
        self.current_idx = 0


# ---------------------------------------------------------------------------
# Model size helper
# ---------------------------------------------------------------------------

def get_model_size_mb(path: str) -> float:
    """Return file size in MB."""
    return os.path.getsize(path) / (1024 * 1024)


# ---------------------------------------------------------------------------
# Accuracy comparison
# ---------------------------------------------------------------------------

def compare_outputs(fp32_path: str, int8_path: str,
                    num_test_samples: int = 5) -> dict:
    """Run inference on both models and compare outputs."""
    print("\n" + "=" * 60)
    print("Accuracy Comparison: FP32 vs INT8")
    print("=" * 60)

    fp32_sess = ort.InferenceSession(fp32_path)
    int8_sess = ort.InferenceSession(int8_path)

    input_info = fp32_sess.get_inputs()[0]
    input_name = input_info.name
    input_shape = [1 if isinstance(d, str) else d for d in input_info.shape]

    all_max_diffs = []
    all_mean_diffs = []

    for i in range(num_test_samples):
        test_data = np.random.uniform(0.0, 1.0, input_shape).astype(np.float32)

        fp32_outputs = fp32_sess.run(None, {input_name: test_data})
        int8_outputs = int8_sess.run(None, {input_name: test_data})

        for j, (fp32_out, int8_out) in enumerate(zip(fp32_outputs, int8_outputs)):
            diff = np.abs(fp32_out.astype(np.float64) - int8_out.astype(np.float64))
            max_diff = float(np.max(diff))
            mean_diff = float(np.mean(diff))
            all_max_diffs.append(max_diff)
            all_mean_diffs.append(mean_diff)

            output_name = fp32_sess.get_outputs()[j].name
            print(f"  Sample {i+1}, output '{output_name}': "
                  f"max_abs_diff={max_diff:.6f}, mean_abs_diff={mean_diff:.6f}")

    summary = {
        "max_abs_diff": float(np.max(all_max_diffs)),
        "mean_abs_diff": float(np.mean(all_mean_diffs)),
        "num_test_samples": num_test_samples,
    }

    print(f"\n  Overall max_abs_diff:  {summary['max_abs_diff']:.6f}")
    print(f"  Overall mean_abs_diff: {summary['mean_abs_diff']:.6f}")

    return summary


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize_model(input_path: str, output_path: str,
                   calibration_samples: int = 100,
                   mode: str = "static",
                   quant_format: str = "QDQ",
                   per_channel: bool = True) -> bool:
    """Quantize an FP32 ONNX model to INT8.

    Args:
        input_path: Path to FP32 ONNX model.
        output_path: Path to write INT8 ONNX model.
        calibration_samples: Number of calibration samples (static mode).
        mode: 'static' or 'dynamic'.
        quant_format: 'QDQ' (QuantizeDequantize) or 'QOps' (QuantizeOps).
        per_channel: Use per-channel quantization for weights.

    Returns:
        True if quantization succeeded.
    """
    if not os.path.exists(input_path):
        print(f"[ERROR] Input model not found: {input_path}")
        return False

    print("=" * 60)
    print("DeepCGH Model INT8 Quantization")
    print("=" * 60)
    print(f"  Input:    {input_path}")
    print(f"  Output:   {output_path}")
    print(f"  Mode:     {mode}")
    print(f"  Format:   {quant_format}")
    print(f"  Per-channel: {per_channel}")
    print(f"  Calibration samples: {calibration_samples}")

    # Step 1: Pre-process the model (shape inference + optimization)
    print("\n[1/4] Pre-processing model (shape inference)...")
    preprocessed_path = input_path + ".preprocessed.onnx"
    try:
        quant_pre_process(input_path, preprocessed_path)
    except Exception as e:
        print(f"  [WARN] Pre-processing failed ({e}), using original model.")
        preprocessed_path = input_path

    # Step 2: Quantize
    qformat = QuantFormat.QDQ if quant_format.upper() == "QDQ" else QuantFormat.QOperator
    weight_type = QuantType.QInt8

    if mode == "static":
        print(f"\n[2/4] Running static quantization with {calibration_samples} "
              "calibration samples...")
        print("  Generating synthetic calibration data...")

        cal_reader = DeepCGHCalibrationReader(
            preprocessed_path,
            num_samples=calibration_samples,
        )

        try:
            quantize_static(
                model_input=preprocessed_path,
                model_output=output_path,
                calibration_data_reader=cal_reader,
                quant_format=qformat,
                weight_type=weight_type,
                activation_type=QuantType.QUInt8,
                per_channel=per_channel,
                extra_options={
                    "ActivationSymmetric": False,
                    "WeightSymmetric": True,
                },
            )
        except Exception as e:
            print(f"  [ERROR] Static quantization failed: {e}")
            # Clean up
            if preprocessed_path != input_path and os.path.exists(preprocessed_path):
                os.remove(preprocessed_path)
            return False
    else:
        print("\n[2/4] Running dynamic quantization...")
        try:
            quantize_dynamic(
                model_input=preprocessed_path,
                model_output=output_path,
                weight_type=weight_type,
                per_channel=per_channel,
                extra_options={
                    "WeightSymmetric": True,
                },
            )
        except Exception as e:
            print(f"  [ERROR] Dynamic quantization failed: {e}")
            if preprocessed_path != input_path and os.path.exists(preprocessed_path):
                os.remove(preprocessed_path)
            return False

    # Step 3: Clean up preprocessed file
    if preprocessed_path != input_path and os.path.exists(preprocessed_path):
        os.remove(preprocessed_path)

    # Step 4: Report results
    print("\n[3/4] Quantization complete. Reporting size comparison...")
    fp32_size = get_model_size_mb(input_path)
    int8_size = get_model_size_mb(output_path)
    reduction = (1.0 - int8_size / fp32_size) * 100

    print(f"  FP32 model size: {fp32_size:.2f} MB")
    print(f"  INT8 model size: {int8_size:.2f} MB")
    print(f"  Size reduction:  {reduction:.1f}%")

    # Step 5: Accuracy comparison
    print("\n[4/4] Running accuracy comparison...")
    try:
        accuracy = compare_outputs(input_path, output_path)
    except Exception as e:
        print(f"  [WARN] Accuracy comparison failed: {e}")
        accuracy = None

    # Final summary
    print("\n" + "=" * 60)
    print("Quantization Summary")
    print("=" * 60)
    print(f"  Input:           {input_path}")
    print(f"  Output:          {output_path}")
    print(f"  Mode:            {mode}")
    print(f"  Quant format:    {quant_format}")
    print(f"  FP32 size:       {fp32_size:.2f} MB")
    print(f"  INT8 size:       {int8_size:.2f} MB")
    print(f"  Size reduction:  {reduction:.1f}%")
    if accuracy:
        print(f"  Max abs diff:    {accuracy['max_abs_diff']:.6f}")
        print(f"  Mean abs diff:   {accuracy['mean_abs_diff']:.6f}")
    print("=" * 60)

    return True


# ---------------------------------------------------------------------------
# Auto-detect models
# ---------------------------------------------------------------------------

def find_models(base_dir: str) -> list:
    """Find DeepCGH ONNX models in the models directory."""
    models_dir = os.path.join(base_dir, "models")
    if not os.path.isdir(models_dir):
        return []

    found = []
    for fname in os.listdir(models_dir):
        if fname.endswith(".onnx") and "_int8" not in fname:
            found.append(os.path.join(models_dir, fname))
    return found


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="INT8 quantization for DeepCGH ONNX models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quantize default 256x256 model
  python quantize_model.py

  # Custom input/output
  python quantize_model.py --input model.onnx --output model_int8.onnx

  # More calibration samples for better accuracy
  python quantize_model.py --calibration-samples 200

  # Use QOps format instead of QDQ
  python quantize_model.py --quant-format QOps

  # Dynamic quantization (no calibration needed)
  python quantize_model.py --mode dynamic
""",
    )

    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to FP32 ONNX model (default: auto-detect in models/)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to write INT8 ONNX model (default: <input>_int8.onnx)",
    )
    parser.add_argument(
        "--calibration-samples", type=int, default=100,
        help="Number of calibration samples for static quantization (default: 100)",
    )
    parser.add_argument(
        "--mode", type=str, choices=["static", "dynamic"], default="static",
        help="Quantization mode: 'static' (with calibration) or 'dynamic' (default: static)",
    )
    parser.add_argument(
        "--quant-format", type=str, choices=["QDQ", "QOps"], default="QDQ",
        help="Quantization format: QDQ (QuantizeDequantize) or QOps (QuantizeOps) "
             "(default: QDQ)",
    )
    parser.add_argument(
        "--per-channel", action="store_true", default=True,
        help="Use per-channel weight quantization (default: True)",
    )
    parser.add_argument(
        "--no-per-channel", action="store_true", default=False,
        help="Disable per-channel weight quantization",
    )
    parser.add_argument(
        "--skip-accuracy", action="store_true", default=False,
        help="Skip accuracy comparison after quantization",
    )
    parser.add_argument(
        "--quantize-all", action="store_true", default=False,
        help="Quantize all FP32 ONNX models found in the models/ directory",
    )

    args = parser.parse_args()

    # Determine base directory (DeepCGHEngine root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # tools/ -> DeepCGHEngine/

    if args.no_per_channel:
        args.per_channel = False

    # Determine input/output paths
    if args.quantize_all:
        models = find_models(base_dir)
        if not models:
            print("[ERROR] No FP32 ONNX models found in models/ directory.")
            sys.exit(1)

        print(f"Found {len(models)} model(s) to quantize:")
        for m in models:
            print(f"  - {m}")
        print()

        all_ok = True
        for input_path in models:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_int8{ext}"
            ok = quantize_model(
                input_path=input_path,
                output_path=output_path,
                calibration_samples=args.calibration_samples,
                mode=args.mode,
                quant_format=args.quant_format,
                per_channel=args.per_channel,
            )
            if not ok:
                all_ok = False
                print(f"[FAIL] Quantization failed for {input_path}")
        sys.exit(0 if all_ok else 1)

    else:
        # Single model quantization
        if args.input:
            input_path = args.input
        else:
            # Default: look for deepcgh_unet.onnx
            default_model = os.path.join(base_dir, "models", "deepcgh_unet.onnx")
            if not os.path.exists(default_model):
                print(f"[ERROR] Default model not found: {default_model}")
                print("Use --input to specify the model path.")
                sys.exit(1)
            input_path = default_model

        if args.output:
            output_path = args.output
        else:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_int8{ext}"

        ok = quantize_model(
            input_path=input_path,
            output_path=output_path,
            calibration_samples=args.calibration_samples,
            mode=args.mode,
            quant_format=args.quant_format,
            per_channel=args.per_channel,
        )
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
