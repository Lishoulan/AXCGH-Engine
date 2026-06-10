#!/usr/bin/env python3
"""
benchmark_engine.py — Performance benchmark for DeepCGHEngine.

Measures per-stage latency and throughput for the full hologram generation pipeline.
"""

import os
import sys
import time
import json

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DeepCGHEngine'))
from deepcgh_engine import EngineAPI, EngineConfig, ColorSpace, Status, PhaseFormat

# ===========================================================================
# Configuration
# ===========================================================================
MODEL_PATH = os.path.join('DeepCGHEngine', 'models', 'deepcgh_unet.onnx')
H, W, C = 256, 256, 5
N_WARMUP = 5
N_BENCHMARK = 50

print("=" * 60)
print("  DeepCGHEngine — Performance Benchmark")
print("=" * 60)

# ===========================================================================
# Initialize engine
# ===========================================================================
print("\n[Init] Loading model...")
engine = EngineAPI()
config = EngineConfig(
    height=H, width=W, num_planes=C,
    color_space=ColorSpace.YCbCr,
    phase_format=PhaseFormat.Uint8,
    quantization_bits=8,
    intra_op_threads=4,
)

status = engine.init(MODEL_PATH, config)
assert status == Status.OK, f"Init failed: {engine.last_error}"
print(f"  Model loaded: {MODEL_PATH}")
print(f"  GPU acceleration: {engine.gpu_enabled}")

# ===========================================================================
# Generate test data
# ===========================================================================
np.random.seed(42)
rgb = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
depth = np.random.rand(H, W).astype(np.float32)

# ===========================================================================
# Warm-up runs
# ===========================================================================
print(f"\n[Warm-up] Running {N_WARMUP} warm-up iterations...")
for i in range(N_WARMUP):
    status, phase = engine.generate_hologram(rgb, depth)
    assert status == Status.OK

# ===========================================================================
# Benchmark runs
# ===========================================================================
print(f"[Benchmark] Running {N_BENCHMARK} timed iterations...")
for i in range(N_BENCHMARK):
    status, phase = engine.generate_hologram(rgb, depth, benchmark=True)
    assert status == Status.OK

# ===========================================================================
# Results
# ===========================================================================
stats = engine.get_perf_stats()

print("\n" + "-" * 60)
print("  Performance Results")
print("-" * 60)

for stage in ['preprocess', 'inference', 'postprocess', 'total']:
    if stage in stats:
        s = stats[stage]
        print(f"  {stage:15s}: {s['mean_ms']:7.2f} +/- {s['std_ms']:5.2f} ms  "
              f"(min={s['min_ms']:.2f}, max={s['max_ms']:.2f})")

total_mean = stats.get('total', {}).get('mean_ms', 0)
if total_mean > 0:
    fps = 1000.0 / total_mean
    print(f"\n  Throughput: {fps:.1f} FPS ({total_mean:.2f} ms/frame)")

# ===========================================================================
# Breakdown
# ===========================================================================
if all(k in stats for k in ['preprocess', 'inference', 'postprocess']):
    total = stats['total']['mean_ms']
    print(f"\n  Stage breakdown:")
    for stage in ['preprocess', 'inference', 'postprocess']:
        pct = stats[stage]['mean_ms'] / total * 100
        print(f"    {stage:15s}: {pct:5.1f}%")

# ===========================================================================
# Quantized output benchmark
# ===========================================================================
print(f"\n[Quantized] Benchmarking quantized output...")
for i in range(N_BENCHMARK):
    status, phase_u8 = engine.generate_hologram_quantized(rgb, depth, benchmark=True)

q_stats = engine.get_perf_stats()
if 'total' in q_stats:
    q_total = q_stats['total']['mean_ms']
    q_fps = 1000.0 / q_total
    print(f"  Quantized throughput: {q_fps:.1f} FPS ({q_total:.2f} ms/frame)")

# ===========================================================================
# Save results
# ===========================================================================
results = {
    'resolution': f'{H}x{W}',
    'num_planes': C,
    'n_runs': N_BENCHMARK,
    'gpu_enabled': engine.gpu_enabled,
    'stages': stats,
    'fps': 1000.0 / stats.get('total', {}).get('mean_ms', 1),
}

out_path = os.path.join('DeepCGHEngine', 'models', 'benchmark_results.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved to: {out_path}")

engine.shutdown()
print("\n" + "=" * 60)
print("  Benchmark complete!")
print("=" * 60)
