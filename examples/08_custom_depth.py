#!/usr/bin/env python3
"""
示例 08: 自定义深度图

演示如何创建自定义深度图，并用它生成全息图。
深度图决定了全息图中不同物体的聚焦距离。

深度图说明：
  - 值范围: 任意 float32（引擎会自动归一化到 [0, 1]）
  - 值的含义: 较小的值 = 较近的物体，较大的值 = 较远的物体
  - 形状: [H, W]，与输入图像相同

本示例展示多种深度图创建方式：
  1. 从文本/标签创建二值深度图
  2. 创建多平面深度图（不同区域不同深度）
  3. 从渐变创建连续深度图
  4. 从灰度图像生成深度图

运行方式:
    python examples/08_custom_depth.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "DeepCGHEngine"))

from deepcgh_engine import EngineAPI, EngineConfig, Status


def create_text_depth(H, W, text="CGH", font_scale=0.08):
    """创建包含文字的二值深度图。

    使用简单的点阵方式渲染文字，不需要字体库。
    """
    depth = np.zeros((H, W), dtype=np.float32)

    # 简单的 5x7 点阵字体（大写字母和数字）
    FONT_5x7 = {
        'A': [0x04,0x0A,0x11,0x1F,0x11,0x11,0x11],
        'B': [0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E],
        'C': [0x0E,0x11,0x10,0x10,0x10,0x11,0x0E],
        'D': [0x1C,0x12,0x11,0x11,0x11,0x12,0x1C],
        'E': [0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F],
        'F': [0x1F,0x10,0x10,0x1E,0x10,0x10,0x10],
        'G': [0x0E,0x11,0x10,0x17,0x11,0x11,0x0F],
        'H': [0x11,0x11,0x11,0x1F,0x11,0x11,0x11],
        'I': [0x0E,0x04,0x04,0x04,0x04,0x04,0x0E],
        'J': [0x01,0x01,0x01,0x01,0x01,0x11,0x0E],
        'K': [0x11,0x12,0x14,0x18,0x14,0x12,0x11],
        'L': [0x10,0x10,0x10,0x10,0x10,0x10,0x1F],
        'M': [0x11,0x1B,0x15,0x15,0x11,0x11,0x11],
        'N': [0x11,0x19,0x15,0x13,0x11,0x11,0x11],
        'O': [0x0E,0x11,0x11,0x11,0x11,0x11,0x0E],
        'P': [0x1E,0x11,0x11,0x1E,0x10,0x10,0x10],
        'Q': [0x0E,0x11,0x11,0x11,0x15,0x12,0x0D],
        'R': [0x1E,0x11,0x11,0x1E,0x14,0x12,0x11],
        'S': [0x0E,0x11,0x10,0x0E,0x01,0x11,0x0E],
        'T': [0x1F,0x04,0x04,0x04,0x04,0x04,0x04],
        'U': [0x11,0x11,0x11,0x11,0x11,0x11,0x0E],
        'V': [0x11,0x11,0x11,0x11,0x0A,0x0A,0x04],
        'W': [0x11,0x11,0x11,0x15,0x15,0x1B,0x11],
        'X': [0x11,0x11,0x0A,0x04,0x0A,0x11,0x11],
        'Y': [0x11,0x11,0x0A,0x04,0x04,0x04,0x04],
        'Z': [0x1F,0x01,0x02,0x04,0x08,0x10,0x1F],
    }

    # 计算字符大小和起始位置
    char_w = max(int(5 * font_scale * W / len(text)), 5)
    char_h = max(int(7 * font_scale * H), 7)
    total_w = len(text) * (char_w + 2)
    start_x = (W - total_w) // 2
    start_y = (H - char_h) // 2

    for ci, ch in enumerate(text.upper()):
        if ch not in FONT_5x7:
            continue
        pattern = FONT_5x7[ch]
        x_offset = start_x + ci * (char_w + 2)

        for row, bits in enumerate(pattern):
            y = start_y + int(row * char_h / 7)
            for col in range(5):
                if bits & (1 << (4 - col)):
                    x = x_offset + int(col * char_w / 5)
                    # 在深度图上绘制（值=1.0 表示近处）
                    y_end = min(y + char_h // 7, H)
                    x_end = min(x + char_w // 5, W)
                    depth[y:y_end, x:x_end] = 1.0

    return depth


def create_multi_plane_depth(H, W, num_zones=3):
    """创建多平面深度图（不同区域不同深度）。

    将图像分成多个区域，每个区域有不同的深度值。
    适用于模拟多个物体在不同距离的场景。
    """
    depth = np.zeros((H, W), dtype=np.float32)
    zone_h = H // num_zones

    for i in range(num_zones):
        y0 = i * zone_h
        y1 = min(y0 + zone_h, H)
        depth[y0:y1, :] = (i + 1) / num_zones

    return depth


def create_gradient_depth(H, W, direction="diagonal"):
    """创建渐变深度图。

    参数:
        direction: 渐变方向
            - "horizontal": 水平渐变（左近右远）
            - "vertical": 垂直渐变（上近下远）
            - "diagonal": 对角渐变
            - "radial": 径向渐变（中心近，边缘远）
    """
    yy, xx = np.mgrid[0:H, 0:W]

    if direction == "horizontal":
        depth = (xx / W).astype(np.float32)
    elif direction == "vertical":
        depth = (yy / H).astype(np.float32)
    elif direction == "diagonal":
        depth = ((xx / W + yy / H) / 2).astype(np.float32)
    elif direction == "radial":
        cx, cy = W / 2, H / 2
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        depth = (dist / max_dist).astype(np.float32)
    else:
        depth = np.full((H, W), 0.5, dtype=np.float32)

    return depth


def create_shape_depth(H, W):
    """创建包含几何形状的深度图。

    在不同深度放置圆形、矩形和三角形。
    """
    depth = np.full((H, W), 0.2, dtype=np.float32)  # 背景 = 远处

    yy, xx = np.mgrid[0:H, 0:W]

    # 圆形（近处，深度=0.8）
    cx1, cy1, r1 = W // 4, H // 4, min(H, W) // 6
    circle1 = (xx - cx1) ** 2 + (yy - cy1) ** 2 < r1 ** 2
    depth[circle1] = 0.8

    # 矩形（中等距离，深度=0.5）
    rect_x0, rect_y0 = W // 2, H // 6
    rect_x1, rect_y1 = 3 * W // 4, H // 3
    depth[rect_y0:rect_y1, rect_x0:rect_x1] = 0.5

    # 三角形（中等距离，深度=0.6）
    tri_cx, tri_cy = 3 * W // 4, 2 * H // 3
    tri_size = min(H, W) // 5
    for y in range(H):
        for x in range(W):
            # 简单的等腰三角形判断
            dy = y - (tri_cy - tri_size)
            dx = abs(x - tri_cx)
            if 0 <= dy <= 2 * tri_size and dx <= tri_size - dy * tri_size / (2 * tri_size):
                depth[y, x] = 0.6

    return depth


def main():
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "DeepCGHEngine", "models", "deepcgh_unet.onnx"
    )
    model_path = os.path.abspath(model_path)

    if not os.path.exists(model_path):
        print(f"[错误] 模型文件不存在: {model_path}")
        return

    H, W = 256, 256
    config = EngineConfig(height=H, width=W, num_planes=5)
    engine = EngineAPI()

    status = engine.init(model_path, config)
    if status != Status.OK:
        print(f"[错误] 引擎初始化失败: {engine.last_error}")
        return

    # 创建统一的输入图像（白色背景）
    rgb = np.full((H, W, 3), 200, dtype=np.uint8)

    # 定义不同的深度图
    depth_maps = {
        "text_cgh": create_text_depth(H, W, text="CGH"),
        "multi_plane": create_multi_plane_depth(H, W, num_zones=3),
        "gradient_horizontal": create_gradient_depth(H, W, "horizontal"),
        "gradient_vertical": create_gradient_depth(H, W, "vertical"),
        "gradient_diagonal": create_gradient_depth(H, W, "diagonal"),
        "gradient_radial": create_gradient_depth(H, W, "radial"),
        "shapes": create_shape_depth(H, W),
    }

    output_dir = os.path.join(os.path.dirname(__file__), "..", "result")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 50)
    print("自定义深度图全息图生成")
    print("=" * 50)

    for name, depth in depth_maps.items():
        print(f"\n--- 深度图: {name} ---")
        print(f"  值范围: [{depth.min():.4f}, {depth.max():.4f}]")
        print(f"  唯一值数量: {len(np.unique(depth))}")

        # 生成全息图
        status, phase = engine.generate_hologram(rgb, depth)
        if status == Status.OK:
            print(f"  相位范围: [{phase.min():.4f}, {phase.max():.4f}]")

            # 保存结果
            try:
                from PIL import Image

                # 保存深度图可视化
                depth_vis = (depth * 255).astype(np.uint8)
                depth_path = os.path.join(output_dir, f"example_08_depth_{name}.png")
                Image.fromarray(depth_vis).save(depth_path)

                # 保存相位图可视化
                phase_vis = ((phase + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
                phase_path = os.path.join(output_dir, f"example_08_phase_{name}.png")
                Image.fromarray(phase_vis).save(phase_path)

                print(f"  已保存: {depth_path}")
                print(f"  已保存: {phase_path}")
            except ImportError:
                np.save(os.path.join(output_dir, f"example_08_depth_{name}.npy"), depth)
                np.save(os.path.join(output_dir, f"example_08_phase_{name}.npy"), phase)
                print(f"  已保存为 NumPy 文件")
        else:
            print(f"  [错误] 生成失败: {engine.last_error}")

    # ---- 使用自然图像 + 自定义深度 ----
    print()
    print("=" * 50)
    print("自然图像 + 自定义深度图")
    print("=" * 50)

    # 尝试加载项目自带的测试图像
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "natural_images")
    if os.path.isdir(data_dir):
        try:
            from PIL import Image as PILImage

            for img_name in ["astronaut.png", "camera.png", "chelsea.png"]:
                img_path = os.path.join(data_dir, img_name)
                if not os.path.exists(img_path):
                    continue

                img = PILImage.open(img_path).convert("RGB").resize((W, H))
                rgb_natural = np.array(img, dtype=np.uint8)

                # 使用径向渐变深度图
                depth_radial = create_gradient_depth(H, W, "radial")

                status, phase = engine.generate_hologram(rgb_natural, depth_radial)
                if status == Status.OK:
                    basename = os.path.splitext(img_name)[0]
                    phase_vis = ((phase + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
                    phase_path = os.path.join(output_dir, f"example_08_natural_{basename}.png")
                    PILImage.fromarray(phase_vis).save(phase_path)
                    print(f"  {img_name} -> example_08_natural_{basename}.png")

        except ImportError:
            print("  [跳过] 需要 PIL 加载自然图像")

    engine.shutdown()
    print()
    print("[信息] 示例完成！")


if __name__ == "__main__":
    main()
