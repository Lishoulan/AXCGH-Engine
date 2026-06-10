#!/usr/bin/env python3
"""
train_1080p.py — Train a DeepCGH model at 1920x1080 resolution and export to ONNX.

Configuration:
  - Resolution: 1920x1080, 5 depth planes
  - int_factor: 4 (needed for 1080p to fit in GPU memory)
  - n_kernels: [32, 64, 128]
  - Training: 30 epochs, batch_size=1, lr=1e-4
  - Dataset: 2000 samples, mixed types (Disk, Line, RandomNoise, StructuredText)
  - ONNX export: U-Net subgraph only (amp_0 + phi_0)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
H, W, C = 1080, 1920, 5
IF = 4          # int_factor (4 for 1080p to fit GPU memory)
QUANTIZATION = 256  # 2^8
EPOCHS = 30
BATCH_SIZE = 1
LR = 1e-4
N_SAMPLES = 2000
N_KERNELS = [32, 64, 128]

WAVELENGTH = 5.32e-7
PIXEL_PITCH = 8e-6
PLANE_DISTANCE = 0.01

ONNX_PATH = os.path.join('DeepCGHEngine', 'models', 'deepcgh_unet_1080p.onnx')

# ---------------------------------------------------------------------------
# Seed & GPU setup
# ---------------------------------------------------------------------------
np.random.seed(42)
tf.random.set_seed(42)
for g in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Step 1: Dataset
# ---------------------------------------------------------------------------
print("[1/4] Preparing mixed dataset (1920x1080, 5 planes, 2000 samples)...")

from deepcgh import DeepCGH, DeepCGH_Datasets

data_params = {
    'shape': (H, W, C),
    'data_path': 'dd/Train_1080p',
    'N': N_SAMPLES,
    'train_ratio': 0.9,
    'object_type': 'Mixed',
    'object_size': [3, 8],
    'intensity': [0.7, 1.0],
    'object_count': [5, 15],
    'name': 'Mixed_1080p',
    'compression': 'GZIP',
}

deepcgh_dataset = DeepCGH_Datasets(data_params)
deepcgh_dataset.getDataset()

# ---------------------------------------------------------------------------
# Step 2: Model & Training
# ---------------------------------------------------------------------------
print("[2/4] Building model and training...")

model_params = {
    'model_path': 'DeepCGH_Models_1080p',
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'shuffle': 4,
    'lr': LR,
    'int_factor': IF,
    'quantization': 8,
    'n_kernels': N_KERNELS,
    'plane_distance': PLANE_DISTANCE,
    'wavelength': WAVELENGTH,
    'pixel_size': PIXEL_PITCH,
    'focal_point': 0.1,
    'token': 'train_1080p',
    'shape': (H, W, C),
}

dcgh = DeepCGH(data_params, model_params)
train_history, val_history = dcgh.train(deepcgh_dataset, epochs=model_params['epochs'])

print("\n" + "=" * 60)
print(" " * 20 + "TRAINING FINISHED")
print("=" * 60)

# ---------------------------------------------------------------------------
# Step 3: Build U-Net sub-model and export to ONNX
# ---------------------------------------------------------------------------
print("[3/4] Building U-Net sub-model (amp_0 + phi_0) and exporting to ONNX...")

full_model = dcgh.model

amp_layer = full_model.get_layer('amp_0')
phi_layer = full_model.get_layer('phi_0')

sub_model = tf.keras.Model(
    inputs=full_model.input,
    outputs=[amp_layer.output, phi_layer.output]
)

print(f"  Sub-model input shape:  {sub_model.input_shape}")
print(f"  Sub-model output shapes: {sub_model.output_shape}")

os.makedirs(os.path.dirname(ONNX_PATH), exist_ok=True)

import tf2onnx

input_signature = [tf.TensorSpec(shape=(1, H, W, C), dtype=tf.float32, name='target')]

@tf.function(input_signature=input_signature)
def unet_inference(target):
    amp, phi = sub_model(target, training=False)
    return {'amp_0': amp, 'phi_0': phi}

try:
    onnx_model, _ = tf2onnx.convert.from_function(
        unet_inference,
        input_signature=input_signature,
        opset=13,
        output_path=ONNX_PATH,
    )
    print(f"[OK] ONNX model saved to: {ONNX_PATH}")
except Exception as e:
    print(f"[WARN] from_function failed: {e}")
    print("[INFO] Trying from_keras fallback...")
    onnx_model, _ = tf2onnx.convert.from_keras(
        sub_model,
        input_signature=input_signature,
        opset=13,
        output_path=ONNX_PATH,
    )
    print(f"[OK] ONNX model saved to: {ONNX_PATH}")

# ---------------------------------------------------------------------------
# Step 4: Verify ONNX model
# ---------------------------------------------------------------------------
print("[4/4] Verifying ONNX model against TensorFlow output...")

import onnxruntime as ort

sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])

for inp in sess.get_inputs():
    print(f"  Input:  name='{inp.name}', shape={inp.shape}, dtype={inp.type}")
for out in sess.get_outputs():
    print(f"  Output: name='{out.name}', shape={out.shape}, dtype={out.type}")

test_input = np.random.rand(1, H, W, C).astype(np.float32)
input_name = sess.get_inputs()[0].name
outputs = sess.run(None, {input_name: test_input})

amp_0_onnx = outputs[0]
phi_0_onnx = outputs[1]

print(f"  ONNX amp_0 shape: {amp_0_onnx.shape}, range: [{amp_0_onnx.min():.4f}, {amp_0_onnx.max():.4f}]")
print(f"  ONNX phi_0 shape: {phi_0_onnx.shape}, range: [{phi_0_onnx.min():.4f}, {phi_0_onnx.max():.4f}]")

# Compare with TF sub-model output
tf_amp, tf_phi = sub_model(test_input, training=False).numpy()
print(f"  TF   amp_0 shape: {tf_amp.shape}, range: [{tf_amp.min():.4f}, {tf_amp.max():.4f}]")
print(f"  TF   phi_0 shape: {tf_phi.shape}, range: [{tf_phi.min():.4f}, {tf_phi.max():.4f}]")

# Compute max absolute error
amp_mae = np.max(np.abs(amp_0_onnx - tf_amp))
phi_mae = np.max(np.abs(phi_0_onnx - tf_phi))
print(f"  amp_0 max abs error (ONNX vs TF): {amp_mae:.6e}")
print(f"  phi_0 max abs error (ONNX vs TF): {phi_mae:.6e}")

# Also verify the full pipeline (ONNX amp/phi + IFFT postprocess vs TF full model)
def ifft_amph_postprocess(amp_0, phi_0, quantization=QUANTIZATION):
    amp = np.squeeze(amp_0, axis=-1)
    phi = np.squeeze(phi_0, axis=-1)
    complex_field = amp * np.exp(1j * phi)
    shifted = np.fft.ifftshift(complex_field, axes=[1, 2])
    slm_field = np.fft.ifft2(shifted)
    modulation = np.angle(slm_field)
    quantized = np.round((modulation + np.pi) * (quantization - 1) / (2 * np.pi))
    quantized = quantized / (quantization - 1) * 2 * np.pi - np.pi
    return quantized[..., np.newaxis]

phi_slm_onnx = ifft_amph_postprocess(amp_0_onnx, phi_0_onnx)
tf_output = full_model(test_input, training=False).numpy()

tf_squeezed = np.squeeze(tf_output)
ort_squeezed = np.squeeze(phi_slm_onnx)
correlation = np.corrcoef(tf_squeezed.flatten(), ort_squeezed.flatten())[0, 1]
print(f"  Correlation (ONNX+post vs TF full model): {correlation:.6f}")

if amp_mae < 1e-4 and phi_mae < 1e-4:
    print("[PASS] ONNX output matches TF output within tolerance.")
else:
    print(f"[WARN] ONNX/TF difference exceeds 1e-4 — amp_mae={amp_mae:.6e}, phi_mae={phi_mae:.6e}")

print(f"\n[DONE] 1920x1080 training and ONNX export complete!")
print(f"  Model path: {os.path.abspath(ONNX_PATH)}")
print(f"  Model outputs: amp_0 [1,{H},{W},1] + phi_0 [1,{H},{W},1]")
