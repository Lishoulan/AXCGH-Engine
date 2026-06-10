#!/usr/bin/env python3
"""
示例 05: 实时预览

演示如何使用 RealtimeHologramDisplay 进行实时全息图预览。
支持三种摄像头源：
  - TestPatternSource: 合成测试图案（无需硬件）
  - RealSenseCamera: Intel RealSense 深度相机
  - KinectCamera: Azure Kinect 深度相机

键盘控制：
  q — 退出
  s — 保存截图
  p — 暂停/恢复

运行方式:
    python examples/05_realtime_preview.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "DeepCGHEngine"))

from deepcgh_engine import (
    EngineConfig, Status,
    RealtimeHologramDisplay, RealtimeConfig,
    CameraType, SLABackend,
)


def main():
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "DeepCGHEngine", "models", "deepcgh_unet.onnx"
    )
    model_path = os.path.abspath(model_path)

    if not os.path.exists(model_path):
        print(f"[错误] 模型文件不存在: {model_path}")
        return

    # ---- 配置实时预览 ----
    config = RealtimeConfig(
        # 摄像头源：'test'（合成数据）/ 'realsense' / 'kinect'
        camera_type=CameraType.TEST,

        # 性能参数
        target_fps=30.0,          # 目标帧率
        max_queue_size=2,         # 帧队列深度

        # 引擎配置
        model_path=model_path,
        engine_config=EngineConfig(
            height=256,
            width=256,
            num_planes=5,
            provider="cpu",       # 使用 CPU 推理
        ),

        # SLM 显示后端
        slm_backend=SLABackend.OPENCV,  # 'none' / 'sdl' / 'opencv' / 'sdk'
        slm_device_id=0,

        # 预览窗口
        preview_width=640,
        preview_height=480,
        show_preview=True,

        # 自动分辨率调整（当 FPS 低于目标时降低分辨率）
        auto_adjust_resolution=True,
        min_resolution=64,
        resolution_step=64,

        # 截图保存目录
        screenshot_dir="screenshots",
    )

    # ---- 创建并运行实时预览 ----
    display = RealtimeHologramDisplay(config)

    print("=" * 50)
    print("AXCGH-Engine 实时全息图预览")
    print("=" * 50)
    print()
    print("键盘控制:")
    print("  q — 退出")
    print("  s — 保存截图")
    print("  p — 暂停/恢复")
    print()
    print("启动中...")
    print()

    # 运行（阻塞直到用户按 q 退出）
    display.run()

    # 运行结束后打印性能统计
    print()
    print("=" * 50)
    print("性能统计")
    print("=" * 50)
    summary = display.perf_summary
    for key, val in summary.items():
        if isinstance(val, dict):
            print(f"  {key}: mean={val.get('mean_ms', 0):.1f}ms, "
                  f"max={val.get('max_ms', 0):.1f}ms")
        else:
            print(f"  {key}: {val:.1f}")


def demo_with_realsense():
    """使用 Intel RealSense 相机的示例配置。"""
    config = RealtimeConfig(
        camera_type=CameraType.REALSENSE,
        device_id=0,
        target_fps=30.0,
        model_path="models/deepcgh_unet.onnx",
        engine_config=EngineConfig(height=256, width=256, num_planes=5),
        slm_backend=SLABackend.OPENCV,
        show_preview=True,
    )

    display = RealtimeHologramDisplay(config)
    display.run()


def demo_with_kinect():
    """使用 Azure Kinect 相机的示例配置。"""
    config = RealtimeConfig(
        camera_type=CameraType.KINECT,
        device_id=0,
        target_fps=30.0,
        model_path="models/deepcgh_unet.onnx",
        engine_config=EngineConfig(height=256, width=256, num_planes=5),
        slm_backend=SLABackend.OPENCV,
        show_preview=True,
    )

    display = RealtimeHologramDisplay(config)
    display.run()


def demo_no_preview():
    """不显示预览窗口，仅输出到 SLM 的配置。"""
    config = RealtimeConfig(
        camera_type=CameraType.TEST,
        model_path="models/deepcgh_unet.onnx",
        engine_config=EngineConfig(height=256, width=256, num_planes=5),
        slm_backend=SLABackend.NONE,    # 不显示 SLM
        show_preview=False,              # 不显示预览
        target_fps=60.0,                # 更高的目标帧率
    )

    display = RealtimeHologramDisplay(config)

    # 非阻塞方式运行（在自己的线程中）
    import threading
    thread = threading.Thread(target=display.run, daemon=True)
    thread.start()

    # 主线程可以做其他事情
    import time
    for i in range(10):
        time.sleep(1)
        if display.current_phase is not None:
            print(f"  帧 {i}: 相位图形状={display.current_phase.shape}")

    display.stop()
    thread.join(timeout=3.0)


if __name__ == "__main__":
    main()
