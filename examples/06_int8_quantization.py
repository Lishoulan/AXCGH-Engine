#!/usr/bin/env python3
"""
示例 06: INT8 量化

演示如何将 FP32 ONNX 模型量化为 INT8，以减小模型体积并加速推理。
量化后的模型体积约为原来的 1/3，精度损失通常很小。

支持两种量化模式：
  - static: 静态量化（需要校准数据，精度更好）
  - dynamic: 动态量化（不需要校准数据，更简单）

运行方式:
    python examples/06_int8_quantization.py
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "DeepCGHEngine"))

from deepcgh_engine import EngineAPI, EngineConfig, Status


def quantize_model_static(input_path, output_path, calibration_samples=100):
    """使用静态量化将 FP32 模型转换为 INT8。

    静态量化需要校准数据来确定激活值的量化范围。
    这里使用合成数据作为校准样本。

    参数:
        input_path: FP32 ONNX 模型路径
        output_path: INT8 ONNX 模型输出路径
        calibration_samples: 校准样本数量
    """
    try:
        from onnxruntime.quantization import (
            QuantFormat, QuantType, CalibrationDataReader,
            quantize_static,
        )
        from onnxruntime.quantization.shape_inference import quant_pre_process
    except ImportError:
        print("[错误] 缺少依赖: pip install onnx onnxruntime")
        return False

    # 定义校准数据读取器
    class DeepCGHCalibrationReader(CalibrationDataReader):
        """生成合成校准数据。"""
        def __init__(self, model_path, num_samples):
            import onnxruntime as ort
            self.num_samples = num_samples
            self.current_idx = 0

            session = ort.InferenceSession(model_path)
            input_info = session.get_inputs()[0]
            self.input_name = input_info.name
            self.input_shape = [
                1 if isinstance(d, str) else d for d in input_info.shape
            ]

        def get_next(self):
            if self.current_idx >= self.num_samples:
                return None
            data = np.random.uniform(0.0, 1.0, self.input_shape).astype(np.float32)
            self.current_idx += 1
            return {self.input_name: data}

        def rewind(self):
            self.current_idx = 0

    print(f"[信息] 开始静态量化...")
    print(f"  输入: {input_path}")
    print(f"  输出: {output_path}")
    print(f"  校准样本数: {calibration_samples}")

    # 步骤 1: 预处理模型（形状推断）
    preprocessed_path = input_path + ".preprocessed.onnx"
    try:
        quant_pre_process(input_path, preprocessed_path)
    except Exception as e:
        print(f"  [警告] 预处理失败 ({e})，使用原始模型")
        preprocessed_path = input_path

    # 步骤 2: 执行静态量化
    cal_reader = DeepCGHCalibrationReader(preprocessed_path, calibration_samples)

    try:
        quantize_static(
            model_input=preprocessed_path,
            model_output=output_path,
            calibration_data_reader=cal_reader,
            quant_format=QuantFormat.QDQ,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
            per_channel=True,
            extra_options={
                "ActivationSymmetric": False,
                "WeightSymmetric": True,
            },
        )
        print("  [信息] 静态量化完成！")
    except Exception as e:
        print(f"  [错误] 量化失败: {e}")
        return False
    finally:
        # 清理临时文件
        if preprocessed_path != input_path and os.path.exists(preprocessed_path):
            os.remove(preprocessed_path)

    return True


def compare_engines(fp32_path, int8_path, H=256, W=256):
    """比较 FP32 和 INT8 模型的推理结果和性能。"""
    rgb = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    depth = np.random.rand(H, W).astype(np.float32)
    config = EngineConfig(height=H, width=W, num_planes=5)

    # FP32 引擎
    fp32_engine = EngineAPI()
    fp32_status = fp32_engine.init(fp32_path, config)

    # INT8 引擎
    int8_engine = EngineAPI()
    int8_status = int8_engine.init(int8_path, config)

    if fp32_status != Status.OK or int8_status != Status.OK:
        print("[错误] 引擎初始化失败")
        if fp32_status != Status.OK:
            print(f"  FP32: {fp32_engine.last_error}")
        if int8_status != Status.OK:
            print(f"  INT8: {int8_engine.last_error}")
        return

    # 生成全息图
    fp32_status, fp32_phase = fp32_engine.generate_hologram(rgb, depth)
    int8_status, int8_phase = int8_engine.generate_hologram(rgb, depth)

    if fp32_status == Status.OK and int8_status == Status.OK:
        # 比较结果
        diff = np.abs(fp32_phase - int8_phase)
        print()
        print("=" * 50)
        print("精度比较: FP32 vs INT8")
        print("=" * 50)
        print(f"  最大绝对误差: {diff.max():.6f}")
        print(f"  平均绝对误差: {diff.mean():.6f}")
        print(f"  中位绝对误差: {np.median(diff):.6f}")

    # 性能比较
    num_runs = 30
    print()
    print("性能比较:")
    for label, eng in [("FP32", fp32_engine), ("INT8", int8_engine)]:
        times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            eng.generate_hologram(rgb, depth)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        arr = np.array(times)
        print(f"  {label}: 平均 {arr.mean():.2f} ms, "
              f"最小 {arr.min():.2f} ms, FPS {1000.0/arr.mean():.1f}")

    fp32_engine.shutdown()
    int8_engine.shutdown()


def main():
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "DeepCGHEngine", "models", "deepcgh_unet.onnx"
    )
    model_path = os.path.abspath(model_path)

    if not os.path.exists(model_path):
        print(f"[错误] 模型文件不存在: {model_path}")
        return

    # INT8 模型输出路径
    int8_path = model_path.replace(".onnx", "_int8.onnx")

    # ---- 1. 查看原始模型大小 ----
    fp32_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"[信息] FP32 模型大小: {fp32_size:.2f} MB")

    # ---- 2. 执行量化 ----
    if os.path.exists(int8_path):
        print(f"[信息] INT8 模型已存在: {int8_path}")
        overwrite = input("是否重新量化？(y/N): ").strip().lower()
        if overwrite != 'y':
            print("[信息] 跳过量化，使用现有 INT8 模型")
        else:
            success = quantize_model_static(model_path, int8_path, calibration_samples=100)
            if not success:
                return
    else:
        success = quantize_model_static(model_path, int8_path, calibration_samples=100)
        if not success:
            return

    # ---- 3. 比较模型大小 ----
    int8_size = os.path.getsize(int8_path) / (1024 * 1024)
    reduction = (1.0 - int8_size / fp32_size) * 100

    print()
    print("=" * 50)
    print("模型大小比较")
    print("=" * 50)
    print(f"  FP32: {fp32_size:.2f} MB")
    print(f"  INT8: {int8_size:.2f} MB")
    print(f"  压缩率: {reduction:.1f}%")
    print(f"  压缩比: {fp32_size/int8_size:.2f}x")

    # ---- 4. 比较精度和性能 ----
    compare_engines(model_path, int8_path)

    print()
    print("[信息] 示例完成！")
    print()
    print("提示: 也可以使用命令行工具进行量化:")
    print(f"  python DeepCGHEngine/tools/quantize_model.py --input {model_path}")
    print(f"  python DeepCGHEngine/tools/quantize_model.py --mode dynamic  # 动态量化")
    print(f"  python DeepCGHEngine/tools/quantize_model.py --quantize-all  # 批量量化")


if __name__ == "__main__":
    main()
