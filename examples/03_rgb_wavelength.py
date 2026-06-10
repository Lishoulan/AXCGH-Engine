#!/usr/bin/env python3
"""
示例 03: 多波长 RGB 全息图

演示如何使用 RGBHologramEngine 生成三波长（红/绿/蓝）全息图，
并将三个通道的相位图合并为一张 SLM 相位图。

支持两种合并模式：
  - TimeDivision（时分复用）: 三个通道依次显示，合并图为平均值
  - SpatialMultiplex（空分复用）: 三个通道按棋盘格/条纹交错排列

运行方式:
    python examples/03_rgb_wavelength.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "DeepCGHEngine"))

from deepcgh_engine import (
    EngineAPI, EngineConfig, Status,
    RGBHologramEngine, RGBEngineConfig,
    CombineMode, SpatialPattern,
    WAVELENGTH_R, WAVELENGTH_G, WAVELENGTH_B,
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

    # 创建一个彩色测试图像
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    # 左半部分红色，右半部分蓝色，中间渐变
    rgb[:, :W//3, 0] = 255       # 红色区域
    rgb[:, W//3:2*W//3, 1] = 255  # 绿色区域
    rgb[:, 2*W//3:, 2] = 255     # 蓝色区域

    # 创建深度图
    yy, xx = np.mgrid[0:H, 0:W]
    depth = (0.5 + 0.3 * np.sin(xx * 0.03) * np.cos(yy * 0.03)).astype(np.float32)

    # ---- 方式 1: 时分复用 ----
    print("=" * 50)
    print("模式 1: 时分复用 (TimeDivision)")
    print("=" * 50)

    td_config = RGBEngineConfig(
        base_config=EngineConfig(height=H, width=W, num_planes=5),
        combine_mode=CombineMode.TimeDivision,
        # 标准激光波长（单位：mm）
        wavelength_r=WAVELENGTH_R,  # 633 nm (HeNe 红光)
        wavelength_g=WAVELENGTH_G,  # 532 nm (Nd:YAG 绿光)
        wavelength_b=WAVELENGTH_B,  # 450 nm (蓝光激光二极管)
    )

    td_engine = RGBHologramEngine()
    status = td_engine.init(model_path, td_config)

    if status == Status.OK:
        print("[信息] 时分复用引擎初始化成功")

        # 生成 RGB 全息图
        status, result = td_engine.generate_rgb_hologram(rgb, depth)
        if status == Status.OK:
            print(f"  红通道相位范围: [{result['phase_r'].min():.4f}, {result['phase_r'].max():.4f}]")
            print(f"  绿通道相位范围: [{result['phase_g'].min():.4f}, {result['phase_g'].max():.4f}]")
            print(f"  蓝通道相位范围: [{result['phase_b'].min():.4f}, {result['phase_b'].max():.4f}]")
            print(f"  合并相位范围:   [{result['phase_combined'].min():.4f}, {result['phase_combined'].max():.4f}]")
        else:
            print(f"  [错误] 生成失败: {td_engine.last_error}")

        td_engine.shutdown()
    else:
        print(f"[错误] 引擎初始化失败: {td_engine.last_error}")

    # ---- 方式 2: 空分复用（棋盘格） ----
    print()
    print("=" * 50)
    print("模式 2: 空分复用 - 棋盘格 (SpatialMultiplex + Checkerboard)")
    print("=" * 50)

    sm_config = RGBEngineConfig(
        base_config=EngineConfig(height=H, width=W, num_planes=5),
        combine_mode=CombineMode.SpatialMultiplex,
        spatial_pattern=SpatialPattern.Checkerboard,
    )

    sm_engine = RGBHologramEngine()
    status = sm_engine.init(model_path, sm_config)

    if status == Status.OK:
        print("[信息] 空分复用引擎初始化成功")

        status, result = sm_engine.generate_rgb_hologram(rgb, depth)
        if status == Status.OK:
            print(f"  合并相位范围: [{result['phase_combined'].min():.4f}, "
                  f"{result['phase_combined'].max():.4f}]")
        else:
            print(f"  [错误] 生成失败: {sm_engine.last_error}")

        sm_engine.shutdown()
    else:
        print(f"[错误] 引擎初始化失败: {sm_engine.last_error}")

    # ---- 方式 3: 空分复用（条纹） ----
    print()
    print("=" * 50)
    print("模式 3: 空分复用 - 条纹 (SpatialMultiplex + Stripe)")
    print("=" * 50)

    stripe_config = RGBEngineConfig(
        base_config=EngineConfig(height=H, width=W, num_planes=5),
        combine_mode=CombineMode.SpatialMultiplex,
        spatial_pattern=SpatialPattern.Stripe,
    )

    stripe_engine = RGBHologramEngine()
    status = stripe_engine.init(model_path, stripe_config)

    if status == Status.OK:
        print("[信息] 条纹空分复用引擎初始化成功")

        status, result = stripe_engine.generate_rgb_hologram(rgb, depth)
        if status == Status.OK:
            print(f"  合并相位范围: [{result['phase_combined'].min():.4f}, "
                  f"{result['phase_combined'].max():.4f}]")

            # 保存结果
            output_dir = os.path.join(os.path.dirname(__file__), "..", "result")
            os.makedirs(output_dir, exist_ok=True)

            try:
                from PIL import Image
                for ch_name in ['phase_r', 'phase_g', 'phase_b', 'phase_combined']:
                    phase_data = result[ch_name]
                    vis = ((phase_data + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
                    path = os.path.join(output_dir, f"example_03_{ch_name}.png")
                    Image.fromarray(vis).save(path)
                    print(f"  [信息] 已保存: {path}")
            except ImportError:
                pass

        stripe_engine.shutdown()
    else:
        print(f"[错误] 引擎初始化失败: {stripe_engine.last_error}")

    # ---- 生成量化版本 ----
    print()
    print("=" * 50)
    print("量化 RGB 全息图（用于 SLM 显示）")
    print("=" * 50)

    q_config = RGBEngineConfig(
        base_config=EngineConfig(
            height=H, width=W, num_planes=5,
            phase_format="uint8"  # 输出 uint8 格式
        ),
        combine_mode=CombineMode.TimeDivision,
    )

    q_engine = RGBHologramEngine()
    status = q_engine.init(model_path, q_config)

    if status == Status.OK:
        status, result = q_engine.generate_rgb_hologram_quantized(rgb, depth)
        if status == Status.OK:
            for ch_name in ['phase_r', 'phase_g', 'phase_b', 'phase_combined']:
                print(f"  {ch_name}: dtype={result[ch_name].dtype}, "
                      f"shape={result[ch_name].shape}")
        q_engine.shutdown()

    print()
    print("[信息] 示例完成！")


if __name__ == "__main__":
    main()
