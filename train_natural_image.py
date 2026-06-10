
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from deepcgh import DeepCGH, DeepCGH_Datasets
from utils import (
    get_propagate, display_results, make_diag_point_volume,
    make_text_gaussian_dotlattice_volume, ensure_4d,
    make_random_points_volume, make_text_solid_volume,
    load_natural_image
)
import skimage.data

VIS_ARGS = dict(stretch=True, p=97)

# ---------------------------
# 全局配置
# ---------------------------
H, W, C = 512, 512, 5
WAVELENGTH = 5.32e-7
PIXEL_PITCH = 8e-6
PLANE_DISTANCE = 0.01

# 训练配置
DO_TRAIN = True
EPOCHS = 150  # 增加训练轮数
BATCH_SIZE = 4

# 模型配置
MODEL_PATH = "DeepCGH_Models"
NEW_TOKEN = "Natural_Improved_v1"  # 新的模型 Token
SOURCE_TOKEN = "CNTFalse_Noise_1022_0157"  # 使用现有模型作为预训练

np.random.seed(42)
tf.random.set_seed(42)
for g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

# ---------------------------
# 学习率调度器
# ---------------------------
def lr_schedule(epoch):
    initial_lr = 3e-5
    if epoch &lt; 50:
        return initial_lr
    elif epoch &lt; 100:
        return initial_lr * 0.5
    else:
        return initial_lr * 0.2

# ---------------------------
# 1. 数据集参数与准备
# ---------------------------
data_params = {
    'shape': (H, W, C),
    'data_path': 'dd/Natural_Improved_Dataset',
    'N': 8000,  # 增加数据集大小
    'train_ratio': 0.95,
    'object_type': 'Natural',
    'object_size': [3, 8],
    'intensity': [0.7, 1.0],
    'object_count': [10, 25],
    'name': 'Natural_Improved',
    'compression': 'GZIP',
}

print("[SETUP] Preparing improved natural dataset...")
deepcgh_dataset = DeepCGH_Datasets(data_params)
deepcgh_dataset.getDataset()
print("[SETUP] Dataset is ready.")

# ---------------------------
# 2. 模型参数与训练
# ---------------------------
model_params = {
    'model_path': MODEL_PATH,
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'shuffle': 4,
    'lr': 3e-5,
    'int_factor': 32,
    'quantization': 8,
    'n_kernels': [24, 48, 96],
    'plane_distance': PLANE_DISTANCE,
    'wavelength': WAVELENGTH,
    'pixel_size': PIXEL_PITCH,
    'focal_point': 0.1,
    'token': NEW_TOKEN,
    'shape': (H, W, C),
}

dcgh = DeepCGH(data_params, model_params)

# 尝试加载源模型进行微调
if SOURCE_TOKEN:
    source_model_params = model_params.copy()
    source_model_params['token'] = SOURCE_TOKEN
    dcgh_source = DeepCGH(data_params, source_model_params)
    
    if dcgh_source.checkpoint_manager.latest_checkpoint:
        print(f"[INFO] Loading source model from {dcgh_source.checkpoint_manager.latest_checkpoint}")
        dcgh_source.checkpoint.restore(dcgh_source.checkpoint_manager.latest_checkpoint).expect_partial()
        dcgh.model.set_weights(dcgh_source.model.get_weights())
        print("[SUCCESS] Source model weights loaded for finetuning!")

train_history, val_history = None, None 
if DO_TRAIN:
    print("\n" + "="*60 + "\n" + " " * 15 + "STARTING TRAINING FOR NATURAL IMAGES" + "\n" + "="*60)
    train_history, val_history = dcgh.train(deepcgh_dataset, epochs=model_params['epochs'], lr_schedule=lr_schedule)
    print("\n" + "="*60 + "\n" + " " * 22 + "TRAINING FINISHED" + "\n" + "="*60 + "\n")
else:
    print("[INFO] DO_TRAIN is False, skipping training.")

# ---------------------------
# 3. 绘制训练历史曲线
# ---------------------------
if DO_TRAIN and train_history and val_history:
    print("\n[PLOT] Generating training loss curve...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), train_history, label='Training Loss', marker='o')
    plt.plot(range(1, EPOCHS + 1), val_history, label='Validation Loss', marker='x')
    plt.title('Model Loss During Training (Natural Images)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(1, EPOCHS + 1, max(1, EPOCHS // 10)))
    plt.tight_layout()
    loss_curve_filename = 'training_loss_curve_natural.png'
    plt.savefig(loss_curve_filename)
    plt.close()
    print(f"[PLOT] Training loss curve saved to '{loss_curve_filename}'")

# ---------------------------
# 4. 评估与验证
# ---------------------------
propagate = get_propagate(data_params, {**model_params, "HMatrix": dcgh.Hs,}, unitary=False)

def evaluate_and_save(test_name, test_volume, dcgh_model, propagate_fn):
    print(f"\n[EVAL] Evaluating on '{test_name}' target...")
    test_batch = ensure_4d(test_volume)
    slm_phase = dcgh_model.get_hologram(test_batch)
    reconstructions = propagate_fn(tf.convert_to_tensor(slm_phase, tf.float32)).numpy()
    
    # 保存 .png 图像
    hologram_filename_png = f'natural_hologram_{test_name}.png'
    phase_normalized = (np.squeeze(slm_phase) + np.pi) / (2 * np.pi)
    plt.imsave(hologram_filename_png, phase_normalized, cmap='gray', format='png')
    print(f"[SAVE] Grayscale hologram image saved to '{hologram_filename_png}'")

    try:
        results_filename = f'natural_results_{test_name}.png'
        display_results(test_batch, slm_phase, reconstructions, 0.0, filename=results_filename, **VIS_ARGS)
        print(f"[EVAL] Results for '{test_name}' saved to '{results_filename}'")
    except Exception as e:
        print(f"[WARN] display_results for '{test_name}' failed: {e}")

# --- 测试自然图像 ---
print("\n" + "="*60)
print(" " * 10 + "EVALUATING ON NATURAL IMAGES")
print("="*60)

# 宇航员图像
print("\n[EVAL] Testing on Natural Image (Astronaut)...")
astro = skimage.data.astronaut() 
temp_img_path = "temp_astro_natural.png"
plt.imsave(temp_img_path, astro)
test_vol_natural = load_natural_image(temp_img_path, H, W, C)
evaluate_and_save("real_astronaut", test_vol_natural, dcgh, propagate)
if os.path.exists(temp_img_path):
    os.remove(temp_img_path)

# 摄影师图像
print("\n[EVAL] Testing on Natural Image (Cameraman)...")
cam = skimage.data.camera()
temp_cam_path = "temp_cam_natural.png"
plt.imsave(temp_cam_path, cam, cmap='gray')
test_vol_cam = load_natural_image(temp_cam_path, H, W, C)
evaluate_and_save("real_cameraman", test_vol_cam, dcgh, propagate)
if os.path.exists(temp_cam_path):
    os.remove(temp_cam_path)

# 其他测试
test_vol_diag = make_diag_point_volume(H=H, W=W, C=C, margin=0.2, dot_sigma=1.5)
evaluate_and_save("diag_points", test_vol_diag, dcgh, propagate)

texts_axcgh = ['A','X','C','G','H'][:C]
test_vol_axcgh = make_text_gaussian_dotlattice_volume(H=H, W=W, C=C, texts=texts_axcgh, dot_sigma=2.5, grid_step=25, accept_radius=10, letter_height_ratio=0.6)
evaluate_and_save("letters_gaussdots", test_vol_axcgh, dcgh, propagate)

print("\n" + "="*60)
print(" " * 18 + "EVALUATION COMPLETE")
print("="*60)
print(f"\n[DONE] Natural image training and evaluation complete.")
print(f" - New model saved with token: {NEW_TOKEN}")
print(f" - Check the generated PNG files for results!")

