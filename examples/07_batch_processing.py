#!/usr/bin/env python3
"""
示例 07: 批量处理

演示如何批量处理一个文件夹中的所有图像，生成对应的全息相位图。
适用于离线生成大量全息图的场景。

功能：
  - 遍历输入文件夹中的所有图片
  - 自动生成深度图（如果没有提供）
  - 生成全息相位图并保存到输出文件夹
  - 支持自定义深度图文件夹

运行方式:
    python examples/07_batch_processing.py
    python examples/07_batch_processing.py --input_dir data/natural_images --output_dir result/batch
"""

import os
import sys
import argparse
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "DeepCGHEngine"))

from deepcgh_engine import EngineAPI, EngineConfig, Status


def generate_simple_depth(H, W, method="gradient"):
    """生成简单的合成深度图。

    参数:
        H, W: 图像尺寸
        method: 深度图生成方法
            - "gradient": 水平渐变
            - "center": 中心凸起
            - "random": 随机噪声
            - "flat": 平面

    返回:
        float32 数组 [H, W]
    """
    if method == "gradient":
        yy, xx = np.mgrid[0:H, 0:W]
        depth = (xx / W).astype(np.float32)
    elif method == "center":
        yy, xx = np.mgrid[0:H, 0:W]
        cx, cy = W / 2, H / 2
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        depth = np.clip(1.0 - dist / (max(H, W) / 2), 0, 1).astype(np.float32)
    elif method == "random":
        depth = np.random.rand(H, W).astype(np.float32)
    elif method == "flat":
        depth = np.full((H, W), 0.5, dtype=np.float32)
    else:
        depth = np.full((H, W), 0.5, dtype=np.float32)

    return depth


def load_image(path, target_h, target_w):
    """加载图像并调整大小。

    返回:
        rgb: uint8 [H, W, 3]
    """
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        img = img.resize((target_w, target_h), Image.LANCZOS)
        return np.array(img, dtype=np.uint8)
    except ImportError:
        # 使用 OpenCV 作为备选
        try:
            import cv2
            img = cv2.imread(path)
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (target_w, target_h))
            return img.astype(np.uint8)
        except ImportError:
            print(f"[错误] 需要 PIL 或 OpenCV 来加载图像")
            return None


def save_phase(phase, path, phase_format="uint8"):
    """保存相位图为图片。

    参数:
        phase: float32 [H, W] 相位图，范围 [-π, π]
        path: 保存路径
        phase_format: 保存格式
    """
    if phase_format == "uint8":
        vis = ((phase + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
    elif phase_format == "uint16":
        vis = ((phase + np.pi) / (2 * np.pi) * 65535).astype(np.uint16)
    else:
        vis = phase

    try:
        from PIL import Image
        if vis.dtype == np.uint16:
            # PIL 不直接支持 16-bit 灰度保存，转为 numpy
            np.save(path.replace('.png', '.npy'), vis)
            return
        Image.fromarray(vis).save(path)
    except ImportError:
        np.save(path.replace('.png', '.npy'), vis)


def main():
    parser = argparse.ArgumentParser(description="批量处理图像生成全息图")
    parser.add_argument("--input_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "data", "natural_images"),
                        help="输入图像文件夹")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "result", "batch"),
                        help="输出文件夹")
    parser.add_argument("--depth_method", type=str, default="center",
                        choices=["gradient", "center", "random", "flat"],
                        help="深度图生成方法")
    parser.add_argument("--model_path", type=str, default=None,
                        help="ONNX 模型路径")
    parser.add_argument("--height", type=int, default=256, help="模型输入高度")
    parser.add_argument("--width", type=int, default=256, help="模型输入宽度")
    args = parser.parse_args()

    # 查找模型
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "DeepCGHEngine", "models", "deepcgh_unet.onnx"
        )
    model_path = os.path.abspath(model_path)

    if not os.path.exists(model_path):
        print(f"[错误] 模型文件不存在: {model_path}")
        return

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isdir(input_dir):
        print(f"[错误] 输入文件夹不存在: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 查找输入图像
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    image_files = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]

    if not image_files:
        print(f"[错误] 输入文件夹中没有找到图像: {input_dir}")
        return

    print(f"[信息] 找到 {len(image_files)} 张图像")
    print(f"[信息] 输入目录: {input_dir}")
    print(f"[信息] 输出目录: {output_dir}")

    # 初始化引擎
    H, W = args.height, args.width
    config = EngineConfig(height=H, width=W, num_planes=5)
    engine = EngineAPI()

    status = engine.init(model_path, config)
    if status != Status.OK:
        print(f"[错误] 引擎初始化失败: {engine.last_error}")
        return

    print(f"[信息] 引擎初始化成功 (分辨率: {H}x{W})")

    # 批量处理
    total_time = 0
    success_count = 0

    for i, filename in enumerate(sorted(image_files)):
        input_path = os.path.join(input_dir, filename)
        basename = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{basename}_hologram.png")

        # 加载图像
        rgb = load_image(input_path, H, W)
        if rgb is None:
            print(f"  [{i+1}/{len(image_files)}] 跳过 {filename}（无法加载）")
            continue

        # 生成深度图
        depth = generate_simple_depth(H, W, method=args.depth_method)

        # 生成全息图
        t0 = time.perf_counter()
        status, phase = engine.generate_hologram(rgb, depth)
        t1 = time.perf_counter()

        if status == Status.OK:
            elapsed = (t1 - t0) * 1000
            total_time += elapsed
            success_count += 1

            # 保存结果
            save_phase(phase, output_path)

            print(f"  [{i+1}/{len(image_files)}] {filename} -> {basename}_hologram.png "
                  f"({elapsed:.1f} ms)")
        else:
            print(f"  [{i+1}/{len(image_files)}] {filename} 失败: {engine.last_error}")

    # 打印统计
    engine.shutdown()

    print()
    print("=" * 50)
    print("批量处理统计")
    print("=" * 50)
    print(f"  总图像数: {len(image_files)}")
    print(f"  成功处理: {success_count}")
    print(f"  总耗时: {total_time:.1f} ms")
    if success_count > 0:
        print(f"  平均耗时: {total_time / success_count:.1f} ms/帧")
        print(f"  平均 FPS: {1000.0 / (total_time / success_count):.1f}")

    print()
    print("[信息] 示例完成！")


if __name__ == "__main__":
    main()
