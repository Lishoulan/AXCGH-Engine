#!/usr/bin/env python3
"""
示例 01: 基础全息图生成

演示如何使用 Python 引擎生成并保存一张全息相位图。
这是最简单的入门示例。

运行方式:
    python examples/01_basic_hologram.py
"""

import os
import sys
import numpy as np

# 添加项目路径（如果未安装为包）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "DeepCGHEngine"))

from deepcgh_engine import EngineAPI, EngineConfig, Status


def main():
    # ---- 1. 配置引擎参数 ----
    # height/width 必须与训练模型匹配，默认模型为 256x256
    # num_planes 是深度平面数，默认为 5
    config = EngineConfig(
        height=256,
        width=256,
        num_planes=5,
        color_space="ycbcr",      # 颜色空间转换：ycbcr / rgb / gray
        phase_format="uint8",     # 输出格式：uint8 / uint16 / float
        quantization_bits=8,      # SLM 量化位数
    )

    # ---- 2. 创建引擎并加载模型 ----
    engine = EngineAPI()

    # 查找模型文件
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "DeepCGHEngine", "models", "deepcgh_unet.onnx"
    )
    model_path = os.path.abspath(model_path)

    if not os.path.exists(model_path):
        print(f"[错误] 模型文件不存在: {model_path}")
        print("请先运行 export_onnx_v2.py 导出 ONNX 模型")
        return

    status = engine.init(model_path, config)
    if status != Status.OK:
        print(f"[错误] 引擎初始化失败: {engine.last_error}")
        return

    print("[信息] 引擎初始化成功")

    # ---- 3. 准备输入数据 ----
    H, W = config.height, config.width

    # 创建一个简单的测试图像（彩色条纹）
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    bar_width = W // 8
    colors = [
        [255, 0, 0], [0, 255, 0], [0, 0, 255],
        [255, 255, 0], [255, 0, 255], [0, 255, 255],
        [255, 255, 255], [128, 128, 128],
    ]
    for i, color in enumerate(colors):
        x0 = i * bar_width
        x1 = min(x0 + bar_width, W)
        rgb[:, x0:x1, :] = color

    # 创建深度图（中心凸起的曲面）
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W / 2, H / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    depth = np.clip(1.0 - dist / (max(H, W) / 2), 0, 1).astype(np.float32)

    print(f"[信息] 输入 RGB 形状: {rgb.shape}, dtype: {rgb.dtype}")
    print(f"[信息] 输入深度形状: {depth.shape}, dtype: {depth.dtype}")

    # ---- 4. 生成全息图 ----
    # 生成浮点相位图（范围 [-π, π]）
    status, phase = engine.generate_hologram(rgb, depth)
    if status != Status.OK:
        print(f"[错误] 全息图生成失败: {engine.last_error}")
        engine.shutdown()
        return

    print(f"[信息] 相位图形状: {phase.shape}, dtype: {phase.dtype}")
    print(f"[信息] 相位范围: [{phase.min():.4f}, {phase.max():.4f}]")

    # 生成量化相位图（uint8，范围 [0, 255]，可直接用于 SLM 显示）
    status, phase_u8 = engine.generate_hologram_quantized(rgb, depth)
    if status != Status.OK:
        print(f"[错误] 量化全息图生成失败: {engine.last_error}")
        engine.shutdown()
        return

    print(f"[信息] 量化相位图形状: {phase_u8.shape}, dtype: {phase_u8.dtype}")
    print(f"[信息] 量化范围: [{phase_u8.min()}, {phase_u8.max()}]")

    # ---- 5. 保存结果 ----
    output_dir = os.path.join(os.path.dirname(__file__), "..", "result")
    os.makedirs(output_dir, exist_ok=True)

    # 保存为图片
    try:
        from PIL import Image

        # 保存量化相位图
        phase_path = os.path.join(output_dir, "example_01_phase.png")
        Image.fromarray(phase_u8).save(phase_path)
        print(f"[信息] 量化相位图已保存: {phase_path}")

        # 保存浮点相位图的可视化（映射到 0-255）
        phase_vis = ((phase + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
        phase_vis_path = os.path.join(output_dir, "example_01_phase_vis.png")
        Image.fromarray(phase_vis).save(phase_vis_path)
        print(f"[信息] 相位可视化已保存: {phase_vis_path}")

        # 保存输入图像（用于对比）
        input_path = os.path.join(output_dir, "example_01_input.png")
        Image.fromarray(rgb).save(input_path)
        print(f"[信息] 输入图像已保存: {input_path}")

    except ImportError:
        # 没有 PIL，用 NumPy 保存
        npy_path = os.path.join(output_dir, "example_01_phase.npy")
        np.save(npy_path, phase)
        print(f"[信息] 相位图已保存为 NumPy 文件: {npy_path}")

    # ---- 6. 清理 ----
    engine.shutdown()
    print("[信息] 引擎已关闭，示例完成！")


if __name__ == "__main__":
    main()
