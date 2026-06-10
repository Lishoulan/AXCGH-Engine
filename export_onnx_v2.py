#!/usr/bin/env python3
"""
export_onnx_v2.py — Export the DeepCGH U-Net (without FFT post-processing) to ONNX.

The DeepCGH model has this structure:
  Input [H, W, C] -> U-Net -> (amp_0, phi_0) -> IFFT -> phi_slm

The FFT/Complex/Angle ops in the IFFT post-processing are not well supported
in ONNX opset < 20. We export only the U-Net subgraph (up to amp_0 and phi_0),
and implement the IFFT post-processing in the engine itself.

This approach is actually better for production because:
1. The IFFT post-processing is deterministic and can be optimized separately
2. The ONNX model is smaller and more portable
3. We can use optimized FFT libraries (FFTW, cuFFT) in the C++ engine
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Rebuild the model (same as deepcgh.py but export the subgraph)
# ---------------------------------------------------------------------------
H, W, C = 256, 256, 5
IF = 2  # int_factor
QUANTIZATION = 256  # 2^8

from deepcgh import DeepCGH, DeepCGH_Datasets

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
    'int_factor': IF,
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
# Step 1: Train or load model
# ---------------------------------------------------------------------------
print("[1/4] Preparing dataset and model...")
np.random.seed(42)
tf.random.set_seed(42)

deepcgh_dataset = DeepCGH_Datasets(data_params)
deepcgh_dataset.getDataset()

dcgh = DeepCGH(data_params, model_params)
dcgh.train(deepcgh_dataset, epochs=model_params['epochs'])

# ---------------------------------------------------------------------------
# Step 2: Build a sub-model that outputs amp_0 and phi_0 (before IFFT)
# ---------------------------------------------------------------------------
print("[2/4] Building U-Net sub-model (amp_0 + phi_0 outputs)...")

full_model = dcgh.model

# Find the intermediate layers by name
amp_layer = full_model.get_layer('amp_0')
phi_layer = full_model.get_layer('phi_0')

# Create a new model that outputs both amp_0 and phi_0
sub_model = tf.keras.Model(
    inputs=full_model.input,
    outputs=[amp_layer.output, phi_layer.output]
)

print(f"  Sub-model input shape:  {sub_model.input_shape}")
print(f"  Sub-model output shapes: {sub_model.output_shape}")

# ---------------------------------------------------------------------------
# Step 3: Export to ONNX
# ---------------------------------------------------------------------------
print("[3/4] Converting U-Net sub-model to ONNX...")

os.makedirs(os.path.dirname(ONNX_PATH), exist_ok=True)

import tf2onnx

input_signature = [tf.TensorSpec(shape=(1, H, W, C), dtype=tf.float32, name='target')]

@tf.function(input_signature=input_signature)
def unet_inference(target):
    amp, phi = sub_model(target, training=False)
    # Return as a dict for named outputs
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
# Step 4: Verify ONNX model and implement post-processing in Python
# ---------------------------------------------------------------------------
print("[4/4] Verifying ONNX model with IFFT post-processing...")

import onnxruntime as ort

sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])

# Print model I/O info
for inp in sess.get_inputs():
    print(f"  Input:  name='{inp.name}', shape={inp.shape}, dtype={inp.type}")
for out in sess.get_outputs():
    print(f"  Output: name='{out.name}', shape={out.shape}, dtype={out.type}")

# Run inference
test_input = np.random.rand(1, H, W, C).astype(np.float32)
input_name = sess.get_inputs()[0].name
outputs = sess.run(None, {input_name: test_input})

amp_0 = outputs[0]  # Amplitude [1, H, W, 1]
phi_0 = outputs[1]  # Initial phase [1, H, W, 1]

print(f"  amp_0 shape: {amp_0.shape}, range: [{amp_0.min():.4f}, {amp_0.max():.4f}]")
print(f"  phi_0 shape: {phi_0.shape}, range: [{phi_0.min():.4f}, {phi_0.max():.4f}]")

# Implement the IFFT post-processing (same as _ifft_AmPh in deepcgh.py)
def ifft_amph_postprocess(amp_0, phi_0, quantization=QUANTIZATION):
    """
    Replicate the _ifft_AmPh Lambda from DeepCGH:
    1. Construct complex field: amp * exp(j * phi)
    2. IFFT2D
    3. Extract angle (phase)
    4. Quantize to SLM levels
    """
    # Squeeze the last dimension
    amp = np.squeeze(amp_0, axis=-1)    # [1, H, W]
    phi = np.squeeze(phi_0, axis=-1)    # [1, H, W]

    # Construct complex field
    complex_field = amp * np.exp(1j * phi)

    # IFFT shift + IFFT2D
    shifted = np.fft.ifftshift(complex_field, axes=[1, 2])
    slm_field = np.fft.ifft2(shifted)

    # Extract phase angle
    modulation = np.angle(slm_field)

    # Quantize: map [-pi, pi] -> discrete levels -> back to [-pi, pi]
    quantized = np.round((modulation + np.pi) * (quantization - 1) / (2 * np.pi))
    quantized = quantized / (quantization - 1) * 2 * np.pi - np.pi

    return quantized[..., np.newaxis]  # [1, H, W, 1]

# Run post-processing
phi_slm = ifft_amph_postprocess(amp_0, phi_0)
print(f"  phi_slm shape: {phi_slm.shape}, range: [{phi_slm.min():.4f}, {phi_slm.max():.4f}]")

# Compare with TensorFlow model output
tf_output = full_model(test_input, training=False).numpy()
print(f"  TF phi_slm shape: {tf_output.shape}, range: [{tf_output.min():.4f}, {tf_output.max():.4f}]")

# Compute correlation between ONNX+postprocess and TF outputs
tf_squeezed = np.squeeze(tf_output)
ort_squeezed = np.squeeze(phi_slm)
correlation = np.corrcoef(tf_squeezed.flatten(), ort_squeezed.flatten())[0, 1]
print(f"  Correlation (ONNX+post vs TF): {correlation:.6f}")

# Save a test phase map image
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    phase_vis = (ort_squeezed + np.pi) / (2 * np.pi)
    plt.imsave(os.path.join('DeepCGHEngine', 'models', 'test_phase_onnx.png'),
               phase_vis, cmap='hsv')
    print(f"  Test phase map saved to DeepCGHEngine/models/test_phase_onnx.png")
except Exception as e:
    print(f"  Could not save test image: {e}")

print(f"\n[DONE] ONNX export and verification complete!")
print(f"  Model path: {os.path.abspath(ONNX_PATH)}")
print(f"  Model outputs: amp_0 [1,H,W,1] + phi_0 [1,H,W,1]")
print(f"  Post-processing: IFFT2D + angle + quantize (implemented in engine)")
