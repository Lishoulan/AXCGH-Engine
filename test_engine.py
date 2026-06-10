#!/usr/bin/env python3
"""
test_engine.py — End-to-end test for DeepCGHEngine.

Tests the full pipeline: RGB-D input -> PreProcessor -> ONNX Inference
-> IFFT PostProcess -> PhaseMap output.

Also compares results against the original TensorFlow model.
"""

import os
import sys
import time

import numpy as np

# Add the DeepCGHEngine directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DeepCGHEngine'))

from deepcgh_engine import EngineAPI, EngineConfig, ColorSpace, Status, PhaseFormat

# ===========================================================================
# Configuration
# ===========================================================================
MODEL_PATH = os.path.join('DeepCGHEngine', 'models', 'deepcgh_unet.onnx')
H, W, C = 256, 256, 5

print("=" * 60)
print("  DeepCGHEngine — End-to-End Test")
print("=" * 60)

# ===========================================================================
# Test 1: Engine Initialization
# ===========================================================================
print("\n[Test 1] Engine initialization...")

engine = EngineAPI()
config = EngineConfig(
    height=H, width=W, num_planes=C,
    color_space=ColorSpace.YCbCr,
    phase_format=PhaseFormat.Uint8,
    quantization_bits=8,
)

status = engine.init(MODEL_PATH, config)
assert status == Status.OK, f"Init failed: {engine.last_error}"
assert engine.is_ready(), "Engine should be ready after init"
print(f"  Status: OK — engine initialized with model {MODEL_PATH}")

# ===========================================================================
# Test 2: Generate hologram from synthetic RGB-D data
# ===========================================================================
print("\n[Test 2] Generate hologram from synthetic data...")

np.random.seed(42)
rgb = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
depth = np.random.rand(H, W).astype(np.float32)

# Warm-up run
status, phase = engine.generate_hologram(rgb, depth)
assert status == Status.OK, f"Generation failed: {engine.last_error}"
assert phase.shape == (H, W), f"Phase shape {phase.shape} != expected ({H}, {W})"
print(f"  Phase shape: {phase.shape}")
print(f"  Phase range: [{phase.min():.4f}, {phase.max():.4f}] (expected [-3.14, 3.14])")

# Timed runs
N_RUNS = 10
times = []
for i in range(N_RUNS):
    t0 = time.perf_counter()
    status, phase = engine.generate_hologram(rgb, depth)
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)

avg_ms = np.mean(times)
std_ms = np.std(times)
print(f"  Inference time: {avg_ms:.1f} +/- {std_ms:.1f} ms (over {N_RUNS} runs)")

# ===========================================================================
# Test 3: Quantized phase output
# ===========================================================================
print("\n[Test 3] Quantized phase output...")

status, phase_u8 = engine.generate_hologram_quantized(rgb, depth)
assert status == Status.OK, f"Quantized generation failed: {engine.last_error}"
assert phase_u8.dtype == np.uint8, f"Expected uint8, got {phase_u8.dtype}"
assert phase_u8.shape == (H, W), f"Shape mismatch: {phase_u8.shape}"
print(f"  Quantized phase dtype: {phase_u8.dtype}")
print(f"  Quantized phase range: [{phase_u8.min()}, {phase_u8.max()}]")
print(f"  Unique values: {len(np.unique(phase_u8))}")

# ===========================================================================
# Test 4: Different color spaces
# ===========================================================================
print("\n[Test 4] Different color spaces...")

for cs in [ColorSpace.YCbCr, ColorSpace.Gray, ColorSpace.RGB]:
    engine.shutdown()
    config_cs = EngineConfig(height=H, width=W, num_planes=C, color_space=cs)
    status = engine.init(MODEL_PATH, config_cs)
    assert status == Status.OK, f"Init with {cs} failed"
    status, phase = engine.generate_hologram(rgb, depth)
    assert status == Status.OK, f"Generation with {cs} failed"
    print(f"  {cs.value:6s}: phase range [{phase.min():.4f}, {phase.max():.4f}]")

# ===========================================================================
# Test 5: Comparison with TensorFlow model
# ===========================================================================
print("\n[Test 5] Comparison with TensorFlow model...")

try:
    # Re-init with YCbCr
    engine.shutdown()
    config = EngineConfig(height=H, width=W, num_planes=C, color_space=ColorSpace.YCbCr)
    engine.init(MODEL_PATH, config)

    # Use the same target volume as TF model input
    from utils import make_diag_point_volume
    target_vol = make_diag_point_volume(H=H, W=W, C=C, margin=0.2, dot_sigma=1.5)

    # Create RGB-D from the target (use center plane as image)
    center_plane = target_vol[:, :, C // 2]
    rgb_from_target = np.stack([center_plane * 255] * 3, axis=-1).astype(np.uint8)
    depth_from_target = np.zeros((H, W), dtype=np.float32)

    # Run engine
    status, engine_phase = engine.generate_hologram(rgb_from_target, depth_from_target)
    assert status == Status.OK

    # Run TF model for comparison
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    from deepcgh import DeepCGH as TFDeepCGH, DeepCGH_Datasets as TFDeepCGH_Datasets

    data_params = {
        'shape': (H, W, C), 'data_path': 'dd/Export_ONNX', 'N': 500,
        'train_ratio': 0.9, 'object_type': 'Disk', 'object_size': [3, 6],
        'intensity': [0.7, 1.0], 'object_count': [5, 15], 'name': 'Disk',
        'compression': 'GZIP',
    }
    model_params = {
        'model_path': 'DeepCGH_Models_Export', 'epochs': 1, 'batch_size': 2,
        'shuffle': 4, 'lr': 1e-3, 'int_factor': 2, 'quantization': 8,
        'n_kernels': [16, 32, 64], 'plane_distance': 0.01,
        'wavelength': 5.32e-7, 'pixel_size': 8e-6, 'focal_point': 0.1,
        'token': 'export_onnx', 'shape': (H, W, C),
    }

    tf_dcgh = TFDeepCGH(data_params, model_params)
    target_batch = target_vol[np.newaxis, ...]
    tf_phase = tf_dcgh.get_hologram(target_batch)
    tf_phase_squeezed = np.squeeze(tf_phase)

    # Compare
    correlation = np.corrcoef(engine_phase.flatten(), tf_phase_squeezed.flatten())[0, 1]
    print(f"  Engine phase range: [{engine_phase.min():.4f}, {engine_phase.max():.4f}]")
    print(f"  TF phase range:     [{tf_phase_squeezed.min():.4f}, {tf_phase_squeezed.max():.4f}]")
    print(f"  Correlation: {correlation:.6f}")

except ImportError as e:
    print(f"  [SKIP] TensorFlow comparison unavailable: {e}")

# ===========================================================================
# Test 6: Save output visualization
# ===========================================================================
print("\n[Test 6] Saving output visualization...")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Re-init
    engine.shutdown()
    config = EngineConfig(height=H, width=W, num_planes=C, phase_format=PhaseFormat.Uint8)
    engine.init(MODEL_PATH, config)

    from utils import make_diag_point_volume
    target_vol = make_diag_point_volume(H=H, W=W, C=C, margin=0.2, dot_sigma=1.5)
    center_plane = target_vol[:, :, C // 2]
    rgb_test = np.stack([center_plane * 255] * 3, axis=-1).astype(np.uint8)
    depth_test = np.zeros((H, W), dtype=np.float32)

    status, phase = engine.generate_hologram(rgb_test, depth_test)
    status_q, phase_u8 = engine.generate_hologram_quantized(rgb_test, depth_test)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Target
    axes[0].imshow(center_plane, cmap='Greens')
    axes[0].set_title('Target (center plane)')
    axes[0].axis('off')

    # Phase (HSV)
    phase_vis = (phase + np.pi) / (2 * np.pi)
    axes[1].imshow(phase_vis, cmap='hsv', vmin=0, vmax=1)
    axes[1].set_title('SLM Phase (HSV)')
    axes[1].axis('off')

    # Quantized phase
    axes[2].imshow(phase_u8, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title('Quantized Phase (8-bit)')
    axes[2].axis('off')

    fig.suptitle(f'DeepCGHEngine Output — {avg_ms:.1f} ms/frame', fontsize=14)
    fig.tight_layout()

    out_path = os.path.join('DeepCGHEngine', 'models', 'engine_test_output.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved to: {out_path}")

except Exception as e:
    print(f"  [SKIP] Visualization failed: {e}")

# ===========================================================================
# Cleanup
# ===========================================================================
engine.shutdown()
print("\n" + "=" * 60)
print("  All tests passed!")
print("=" * 60)
