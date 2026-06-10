#!/usr/bin/env python3
"""
export_onnx.py — Train a lightweight DeepCGH model and export it to ONNX format.

This script creates a small-scale DeepCGH U-Net model, trains it briefly
on synthetic data, and exports the inference subgraph to ONNX for use
with the DeepCGHEngine.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from deepcgh import DeepCGH, DeepCGH_Datasets

# ---------------------------------------------------------------------------
# Configuration — small model for fast training & export
# ---------------------------------------------------------------------------
H, W, C = 256, 256, 5  # Smaller resolution for quick iteration

data_params = {
    'shape': (H, W, C),
    'data_path': 'dd/Export_ONNX',
    'N': 500,
    'train_ratio': 0.9,
    'object_type': 'Disk',
    'object_size': [3, 6],
    'intensity': [0.7, 1.0],
    'object_count': [5, 15],
    'name': 'Disk',
    'compression': 'GZIP',
}

model_params = {
    'model_path': 'DeepCGH_Models_Export',
    'epochs': 5,
    'batch_size': 2,
    'shuffle': 4,
    'lr': 1e-3,
    'int_factor': 2,
    'quantization': 8,
    'n_kernels': [16, 32, 64],
    'plane_distance': 0.01,
    'wavelength': 5.32e-7,
    'pixel_size': 8e-6,
    'focal_point': 0.1,
    'token': 'export_onnx',
    'shape': (H, W, C),
}

ONNX_PATH = os.path.join('DeepCGHEngine', 'models', 'deepcgh_unet.onnx')

# ---------------------------------------------------------------------------
# Step 1: Prepare dataset
# ---------------------------------------------------------------------------
print("[1/4] Preparing dataset...")
deepcgh_dataset = DeepCGH_Datasets(data_params)
deepcgh_dataset.getDataset()

# ---------------------------------------------------------------------------
# Step 2: Train model
# ---------------------------------------------------------------------------
print("[2/4] Training model...")
np.random.seed(42)
tf.random.set_seed(42)
for g in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

dcgh = DeepCGH(data_params, model_params)
dcgh.train(deepcgh_dataset, epochs=model_params['epochs'])

# ---------------------------------------------------------------------------
# Step 3: Extract the U-Net subgraph (input -> phi_slm output)
# ---------------------------------------------------------------------------
print("[3/4] Extracting inference subgraph for ONNX export...")

# The DeepCGH model's Keras Model has:
#   Input:  'target'  shape [H, W, C]
#   Output: phi_slm   shape [H, W, 1]
#
# We extract just the U-Net part (without the custom _ifft_AmPh Lambda
# which contains FFT operations that are tricky for ONNX export).
# Instead, we export the full model and let ONNX handle the FFT ops.

keras_model = dcgh.model

# Define a concrete function for the export
class ExportModule(tf.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, H, W, C], dtype=tf.float32, name='target')
    ])
    def __call__(self, target):
        return self.model(target, training=False)

export_module = ExportModule(keras_model)

# ---------------------------------------------------------------------------
# Step 4: Export to ONNX via tf2onnx
# ---------------------------------------------------------------------------
print("[4/4] Converting to ONNX...")

os.makedirs(os.path.dirname(ONNX_PATH), exist_ok=True)

try:
    import tf2onnx

    # Use tf2onnx.convert.from_function to convert the tf.function directly
    input_signature = [tf.TensorSpec(shape=(1, H, W, C), dtype=tf.float32, name='target')]

    @tf.function(input_signature=input_signature)
    def inference_fn(target):
        return keras_model(target, training=False)

    onnx_model, _ = tf2onnx.convert.from_function(
        inference_fn,
        input_signature=input_signature,
        opset=13,
        output_path=ONNX_PATH,
    )
    print(f"[OK] ONNX model saved to: {ONNX_PATH}")

except ImportError:
    print("[WARN] tf2onnx not available, trying manual export...")
    raise RuntimeError("tf2onnx is required for ONNX export. Install with: pip install tf2onnx")
except Exception as e:
    # Fallback: try from_keras
    print(f"[WARN] from_function failed ({e}), trying from_keras...")
    try:
        onnx_model, _ = tf2onnx.convert.from_keras(
            keras_model,
            input_signature=input_signature,
            opset=13,
            output_path=ONNX_PATH,
        )
        print(f"[OK] ONNX model saved to: {ONNX_PATH}")
    except Exception as e2:
        print(f"[ERROR] ONNX export failed: {e2}")
        raise

# ---------------------------------------------------------------------------
# Verify the exported model
# ---------------------------------------------------------------------------
print("\n[VERIFY] Loading ONNX model and running inference test...")
import onnxruntime as ort

sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape
output_name = sess.get_outputs()[0].name
output_shape = sess.get_outputs()[0].shape

print(f"  Input:  name='{input_name}', shape={input_shape}")
print(f"  Output: name='{output_name}', shape={output_shape}")

# Run a test inference
test_input = np.random.rand(1, H, W, C).astype(np.float32)
result = sess.run([output_name], {input_name: test_input})[0]

print(f"  Output shape: {result.shape}")
print(f"  Output range: [{result.min():.4f}, {result.max():.4f}]")
print(f"  Output dtype: {result.dtype}")

print("\n[DONE] ONNX export and verification complete!")
print(f"  Model path: {os.path.abspath(ONNX_PATH)}")
