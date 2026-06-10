#!/usr/bin/env python3
"""
示例 02: 使用 C++ 引擎

演示如何使用 C++ 引擎（通过 PyBind11 绑定）生成全息图。
C++ 引擎使用 FFTW3f 进行 IFFT 后处理，性能可能有所不同。

注意: C++ 引擎需要编译 _deepcgh_engine.pyd 扩展模块。
如果未编译，程序会自动回退到 Python 引擎。

运行方式:
    python examples/02_cpp_engine.py
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "DeepCGHEngine"))

from deepcgh_engine import EngineAPI, EngineConfig, Status

# 尝试导入 C++ 引擎
try:
    from deepcgh_engine import CppDeepCGHEngine
    CPP_AVAILABLE = CppDeepCGHEngine is not None
except ImportError:
    CPP_AVAILABLE = False


def benchmark_engine(engine, rgb, depth, num_runs=20, label="Engine"):
    """对引擎进行简单的基准测试。"""
    # 预热（第一次运行可能较慢）
    if hasattr(engine, 'generate_hologram'):
        if isinstance(engine, EngineAPI):
            engine.generate_hologram(rgb, depth)
        else:
            engine.generate_hologram(rgb, depth)

    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        if isinstance(engine, EngineAPI):
            status, phase = engine.generate_hologram(rgb, depth, benchmark=True)
        else:
            phase = engine.generate_hologram(rgb, depth)
            status = Status.OK
        t1 = time.perf_counter()

        if status == Status.OK or (not isinstance(engine, EngineAPI) and phase is not None):
            times.append((t1 - t0) * 1000)

    if times:
        arr = np.array(times)
        print(f"  [{label}] 平均延迟: {arr.mean():.2f} ms, "
              f"最小: {arr.min():.2f} ms, 最大: {arr.max():.2f} ms, "
              f"FPS: {1000.0/arr.mean():.1f}")
    else:
        print(f"  [{label}] 基准测试失败")

    return times


def main():
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "DeepCGHEngine", "models", "deepcgh_unet.onnx"
    )
    model_path = os.path.abspath(model_path)

    if not os.path.exists(model_path):
        print(f"[错误] 模型文件不存在: {model_path}")
        return

    H, W = 256, 256
    rgb = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    depth = np.random.rand(H, W).astype(np.float32)

    # ---- 方式 1: Python 引擎 ----
    print("=" * 50)
    print("Python 引擎 (NumPy FFT)")
    print("=" * 50)

    py_engine = EngineAPI()
    py_config = EngineConfig(height=H, width=W, num_planes=5)
    status = py_engine.init(model_path, py_config)

    if status == Status.OK:
        py_times = benchmark_engine(py_engine, rgb, depth, num_runs=20, label="Python")

        # 获取详细性能统计
        status, _ = py_engine.generate_hologram(rgb, depth, benchmark=True)
        stats = py_engine.get_perf_stats()
        for stage, vals in stats.items():
            print(f"    {stage}: mean={vals['mean_ms']:.2f}ms, "
                  f"min={vals['min_ms']:.2f}ms, max={vals['max_ms']:.2f}ms")

        py_engine.shutdown()
    else:
        print(f"  [错误] Python 引擎初始化失败: {py_engine.last_error}")

    # ---- 方式 2: C++ 引擎 ----
    print()
    print("=" * 50)
    if CPP_AVAILABLE:
        print("C++ 引擎 (FFTW3f + NumPy FFT)")
        print("=" * 50)

        cpp_engine = CppDeepCGHEngine()
        try:
            cpp_engine.init(
                model_path=model_path,
                height=H,
                width=W,
                num_planes=5,
                color_space="ycbcr",
                provider="cpu",
                phase_format="uint8",
                quantization_bits=8,
            )

            # 生成全息图
            phase = cpp_engine.generate_hologram(rgb, depth)
            print(f"  相位图形状: {phase.shape}, dtype: {phase.dtype}")
            print(f"  相位范围: [{phase.min():.4f}, {phase.max():.4f}]")

            # 生成量化全息图
            phase_q = cpp_engine.generate_hologram_quantized(rgb, depth)
            print(f"  量化相位图形状: {phase_q.shape}, dtype: {phase_q.dtype}")

            # 获取原始模型输出（amp_0, phi_0）
            amp, phi = cpp_engine.infer_raw(rgb, depth)
            print(f"  原始振幅形状: {amp.shape}, 原始相位形状: {phi.shape}")

            # 基准测试
            cpp_times = benchmark_engine(cpp_engine, rgb, depth, num_runs=20, label="C++")

            cpp_engine.shutdown()

        except RuntimeError as e:
            print(f"  [错误] C++ 引擎运行失败: {e}")
    else:
        print("C++ 引擎不可用")
        print("=" * 50)
        print("  C++ 引擎需要编译 PyBind11 扩展模块。")
        print("  请参考 README.md 中的构建说明。")
        print("  Python 引擎功能完全相同，可正常使用。")

    print()
    print("[信息] 示例完成！")


if __name__ == "__main__":
    main()
