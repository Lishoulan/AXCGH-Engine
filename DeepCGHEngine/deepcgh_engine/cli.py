"""
AXCGH-Engine Command-Line Interface.

Usage:
    axcgh-demo generate --model <path> --output <path> [--width 256] [--height 256]
    axcgh-demo benchmark --model <path> [--frames 30] [--width 256] [--height 256]
    axcgh-demo quantize --input <path> --output <path> [--samples 100]
    axcgh-demo info
"""

import argparse
import sys
import os
import time
import math

import numpy as np


def cmd_generate(args):
    """Generate a hologram and save to file."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from deepcgh_engine import EngineAPI, EngineConfig

    engine = EngineAPI()
    config = EngineConfig(height=args.height, width=args.width, num_planes=args.planes)
    status = engine.init(args.model, config)

    if status != 0:  # Status.OK
        print(f"Error: Failed to initialize engine (status={status})")
        sys.exit(1)

    print(f"Engine initialized: {args.width}x{args.height}, {args.planes} planes")

    # Generate test data
    np.random.seed(42)
    rgb = np.random.randint(0, 255, (args.height, args.width, 3), dtype=np.uint8)
    depth = np.random.rand(args.height, args.width).astype(np.float32)

    # Generate hologram
    status, phase = engine.generate_hologram(rgb, depth)
    if status != 0:
        print(f"Error: Hologram generation failed (status={status})")
        sys.exit(1)

    # Save output
    ext = os.path.splitext(args.output)[1].lower()
    if ext in ('.png', '.jpg', '.bmp', '.pgm'):
        from PIL import Image
        img = ((phase + np.pi) / (2 * np.pi) * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).save(args.output)
    elif ext == '.npy':
        np.save(args.output, phase)
    else:
        # Default: save as numpy
        np.save(args.output, phase)
        args.output += '.npy'

    print(f"Phase range: [{phase.min():.4f}, {phase.max():.4f}]")
    print(f"Saved to: {args.output}")
    engine.shutdown()


def cmd_benchmark(args):
    """Run performance benchmark."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from deepcgh_engine import EngineAPI, EngineConfig

    engine = EngineAPI()
    config = EngineConfig(height=args.height, width=args.width, num_planes=args.planes)
    status = engine.init(args.model, config)

    if status != 0:
        print(f"Error: Failed to initialize engine (status={status})")
        sys.exit(1)

    np.random.seed(42)
    rgb = np.random.randint(0, 255, (args.height, args.width, 3), dtype=np.uint8)
    depth = np.random.rand(args.height, args.width).astype(np.float32)

    # Warmup
    engine.generate_hologram(rgb, depth)

    # Benchmark
    t0 = time.perf_counter()
    for i in range(args.frames):
        engine.generate_hologram(rgb, depth)
        if (i + 1) % 10 == 0:
            elapsed = (time.perf_counter() - t0) / (i + 1) * 1000
            fps = 1000.0 / elapsed
            print(f"  Frame {i+1}/{args.frames}: {elapsed:.2f} ms/frame ({fps:.1f} FPS)")

    total = time.perf_counter() - t0
    avg = total / args.frames * 1000
    fps = 1000.0 / avg

    print(f"\n=== Benchmark Results ===")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Frames:     {args.frames}")
    print(f"Total:      {total*1000:.1f} ms")
    print(f"Average:    {avg:.2f} ms/frame")
    print(f"FPS:        {fps:.1f}")

    engine.shutdown()


def cmd_quantize(args):
    """Quantize an ONNX model to INT8."""
    tools_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'tools')
    sys.path.insert(0, tools_dir)

    try:
        from quantize_model import quantize_model
    except ImportError:
        # Try direct import
        spec_path = os.path.join(tools_dir, 'quantize_model.py')
        if os.path.exists(spec_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location('quantize_model', spec_path)
            qm = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(qm)
            quantize_model = qm.quantize_model
        else:
            print("Error: quantize_model.py not found")
            sys.exit(1)

    print(f"Quantizing {args.input} -> {args.output}")
    print(f"Calibration samples: {args.samples}")

    quantize_model(
        input_path=args.input,
        output_path=args.output,
        calibration_samples=args.samples,
    )

    if os.path.exists(args.output):
        fp32_size = os.path.getsize(args.input) / 1024 / 1024
        int8_size = os.path.getsize(args.output) / 1024 / 1024
        print(f"\nFP32: {fp32_size:.2f} MB -> INT8: {int8_size:.2f} MB ({fp32_size/int8_size:.2f}x)")
        print("Quantization complete!")
    else:
        print("Quantization failed!")


def cmd_info(args):
    """Show engine information."""
    print("=" * 50)
    print("  AXCGH-Engine v1.0.0")
    print("  Deep Learning Holographic Rendering Engine")
    print("=" * 50)
    print()

    # Check available modules
    modules = {
        'onnxruntime': False,
        'numpy': False,
        'Pillow': False,
        'pygame': False,
        'screeninfo': False,
        'pyrealsense2': False,
        'pyk4a': False,
    }

    for mod in modules:
        try:
            __import__(mod)
            modules[mod] = True
        except ImportError:
            pass

    print("Dependencies:")
    for mod, available in modules.items():
        status = "OK" if available else "NOT INSTALLED"
        print(f"  {mod:20s} {status}")

    # Check C++ engine
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from deepcgh_engine import CppDeepCGHEngine
        print(f"\n  {'C++ Engine':20s} OK")
    except (ImportError, Exception):
        print(f"\n  {'C++ Engine':20s} NOT AVAILABLE")

    # Check GPU
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"\nONNX Runtime providers: {providers}")
        if 'CUDAExecutionProvider' in providers:
            print("  GPU acceleration: AVAILABLE")
        else:
            print("  GPU acceleration: NOT AVAILABLE (install onnxruntime-gpu)")
    except ImportError:
        pass

    print()
    print("Install optional dependencies:")
    print("  pip install axcgh-engine[gpu]      # GPU acceleration")
    print("  pip install axcgh-engine[display]  # SLM direct display")
    print("  pip install axcgh-engine[realsense] # RealSense camera")
    print("  pip install axcgh-engine[all]      # Everything")


def main():
    parser = argparse.ArgumentParser(
        prog='axcgh-demo',
        description='AXCGH-Engine — Deep Learning Holographic Rendering Engine'
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # generate
    p_gen = subparsers.add_parser('generate', help='Generate a hologram')
    p_gen.add_argument('--model', required=True, help='Path to ONNX model')
    p_gen.add_argument('--output', required=True, help='Output file path (.png/.npy)')
    p_gen.add_argument('--width', type=int, default=256, help='Frame width')
    p_gen.add_argument('--height', type=int, default=256, help='Frame height')
    p_gen.add_argument('--planes', type=int, default=5, help='Number of depth planes')

    # benchmark
    p_bench = subparsers.add_parser('benchmark', help='Run performance benchmark')
    p_bench.add_argument('--model', required=True, help='Path to ONNX model')
    p_bench.add_argument('--frames', type=int, default=30, help='Number of frames')
    p_bench.add_argument('--width', type=int, default=256, help='Frame width')
    p_bench.add_argument('--height', type=int, default=256, help='Frame height')
    p_bench.add_argument('--planes', type=int, default=5, help='Number of depth planes')

    # quantize
    p_quant = subparsers.add_parser('quantize', help='Quantize model to INT8')
    p_quant.add_argument('--input', required=True, help='Input ONNX model path')
    p_quant.add_argument('--output', required=True, help='Output INT8 model path')
    p_quant.add_argument('--samples', type=int, default=100, help='Calibration samples')

    # info
    subparsers.add_parser('info', help='Show engine information')

    args = parser.parse_args()

    if args.command == 'generate':
        cmd_generate(args)
    elif args.command == 'benchmark':
        cmd_benchmark(args)
    elif args.command == 'quantize':
        cmd_quantize(args)
    elif args.command == 'info':
        cmd_info(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
