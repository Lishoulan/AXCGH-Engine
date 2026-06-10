#!/usr/bin/env python3
"""
示例 04: SLM 显示

演示如何使用 SLMDriver 将全息相位图发送到 SLM（空间光调制器）。
支持三种后端：
  - DirectDisplayBackend: 通过 HDMI/DVI 在副屏全屏显示（需要 pygame 或 opencv）
  - FileBackend: 保存为图片文件（离线测试，不需要实际 SLM）
  - SDKBackend: 厂商 SDK 接口（需要子类化实现）

运行方式:
    python examples/04_slm_display.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "DeepCGHEngine"))

from deepcgh_engine import (
    EngineAPI, EngineConfig, Status,
    create_slm_driver,
)


def main():
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "DeepCGHEngine", "models", "deepcgh_unet.onnx"
    )
    model_path = os.path.abspath(model_path)

    if not os.path.exists(model_path):
        print(f"[错误] 模型文件不存在: {model_path}")
        return

    H, W = 256, 256

    # 生成全息相位图
    engine = EngineAPI()
    config = EngineConfig(height=H, width=W, num_planes=5)
    status = engine.init(model_path, config)

    if status != Status.OK:
        print(f"[错误] 引擎初始化失败: {engine.last_error}")
        return

    rgb = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    depth = np.random.rand(H, W).astype(np.float32)

    status, phase = engine.generate_hologram(rgb, depth)
    if status != Status.OK:
        print(f"[错误] 全息图生成失败: {engine.last_error}")
        engine.shutdown()
        return

    print(f"[信息] 相位图生成成功: shape={phase.shape}, range=[{phase.min():.4f}, {phase.max():.4f}]")
    engine.shutdown()

    # ---- 方式 1: 文件后端（推荐用于测试） ----
    print()
    print("=" * 50)
    print("方式 1: FileBackend — 保存为图片文件")
    print("=" * 50)

    output_dir = os.path.join(os.path.dirname(__file__), "..", "result", "slm_frames")
    os.makedirs(output_dir, exist_ok=True)

    file_slm = create_slm_driver(
        backend='file',
        resolution=(W, H),       # (width, height)
        bit_depth=8,             # 8-bit SLM
        output_dir=output_dir,
        format='png',            # png 或 bmp
    )

    # 显示（实际上是保存到文件）
    file_slm.display(phase)
    print(f"  已保存第 1 帧，总帧数: {file_slm.frame_count}")

    # 再保存几帧
    for i in range(3):
        # 模拟不同的相位图
        new_phase = phase + np.random.randn(H, W).astype(np.float32) * 0.1
        file_slm.display(new_phase)

    print(f"  已保存 {file_slm.frame_count} 帧到 {output_dir}")

    # 清除（保存空白帧）
    file_slm.clear()
    print(f"  已清除，总帧数: {file_slm.frame_count}")

    file_slm.close()

    # ---- 方式 2: 直接显示后端（需要副屏/SLM） ----
    print()
    print("=" * 50)
    print("方式 2: DirectDisplayBackend — 副屏全屏显示")
    print("=" * 50)
    print("  注意: 此模式需要连接副屏（SLM），如果没有将显示在主屏上。")
    print("  需要 pygame 或 opencv-python。")

    try:
        direct_slm = create_slm_driver(
            backend='direct',
            resolution=(1920, 1080),   # SLM 分辨率
            bit_depth=8,
            display_idx=1,             # 副屏索引（1=第二个显示器）
        )

        # 显示相位图（会自动缩放到 SLM 分辨率）
        direct_slm.display(phase)
        print("  [信息] 相位图已发送到 SLM 显示")

        # 等待用户观察
        import time
        print("  等待 2 秒...")
        time.sleep(2)

        # 清除 SLM
        direct_slm.clear()
        print("  [信息] SLM 已清除")

        direct_slm.close()

    except ImportError as e:
        print(f"  [跳过] 缺少依赖: {e}")
        print("  安装 pygame 或 opencv-python 以启用此功能")
    except Exception as e:
        print(f"  [跳过] 无法初始化直接显示: {e}")

    # ---- 方式 3: SDK 后端（需要厂商 SDK） ----
    print()
    print("=" * 50)
    print("方式 3: SDKBackend — 厂商 SDK 接口")
    print("=" * 50)
    print("  SDKBackend 是一个占位类，需要子类化实现厂商特定的通信逻辑。")
    print("  示例子类化代码:")
    print()
    print("  class HoloeyeBackend(SDKBackend):")
    print("      def __init__(self, resolution, bit_depth=8):")
    print("          super().__init__(resolution, bit_depth, vendor='holoeye')")
    print("          # 初始化 Holoeye SDK")
    print()
    print("      def display(self, phase_map):")
    print("          pixel_data = normalize_phase(phase_map, self.bit_depth)")
    print("          # 通过 SDK 发送 pixel_data 到 SLM")
    print()
    print("      def clear(self):")
    print("          # 通过 SDK 清除 SLM")
    print()
    print("      def close(self):")
    print("          # 通过 SDK 断开连接")

    # ---- 归一化工具函数演示 ----
    print()
    print("=" * 50)
    print("归一化工具函数")
    print("=" * 50)

    from deepcgh_engine.slm_driver import (
        normalize_phase_to_uint8,
        normalize_phase_to_uint16_10bit,
        normalize_phase,
    )

    # 8-bit 归一化
    phase_u8 = normalize_phase_to_uint8(phase)
    print(f"  8-bit 归一化: dtype={phase_u8.dtype}, range=[{phase_u8.min()}, {phase_u8.max()}]")

    # 10-bit 归一化（存储为 uint16）
    phase_u16 = normalize_phase_to_uint16_10bit(phase)
    print(f"  10-bit 归一化: dtype={phase_u16.dtype}, range=[{phase_u16.min()}, {phase_u16.max()}]")

    # 通用归一化
    phase_8 = normalize_phase(phase, bit_depth=8)
    phase_10 = normalize_phase(phase, bit_depth=10)
    print(f"  通用 8-bit: dtype={phase_8.dtype}")
    print(f"  通用 10-bit: dtype={phase_10.dtype}")

    print()
    print("[信息] 示例完成！")


if __name__ == "__main__":
    main()
