#!/usr/bin/env python3
"""
benchmark_gpu.py — GPU benchmark script for DeepCGHEngine.

Compares CPU vs CUDA vs TensorRT performance across:
  - Different resolutions (256, 512, 1024)
  - Different batch sizes (1, 4, 8, 16)
  - FP16 vs FP32 precision

Outputs results as a formatted table and saves to JSON.

Usage:
    # Default: benchmark all available providers
    py -3.10 benchmark_gpu.py

    # Custom model path
    py -3.10 benchmark_gpu.py --model path/to/model.onnx

    # Specific resolutions and batch sizes
    py -3.10 benchmark_gpu.py --resolutions 256 512 --batch-sizes 1 4

    # Skip TensorRT
    py -3.10 benchmark_gpu.py --skip-trt

    # More warmup/runs for stable results
    py -3.10 benchmark_gpu.py --warmup 10 --runs 100
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepcgh_engine.engine import EngineAPI, EngineConfig, Status
from deepcgh_engine.gpu_engine import (
    GPUEngineAPI,
    GPUConfig,
    detect_available_providers,
)


# ===========================================================================
# Benchmark Runner
# ===========================================================================

class GPUBenchmarkRunner:
    """Runs comprehensive GPU benchmarks for DeepCGHEngine."""

    def __init__(self, model_path: str, num_warmup: int = 5, num_runs: int = 50):
        self.model_path = model_path
        self.num_warmup = num_warmup
        self.num_runs = num_runs
        self.results: List[Dict[str, Any]] = []

    def benchmark_single(self, resolution: int, provider_priority: List[str],
                         batch_size: int = 1, fp16: bool = False) -> Dict[str, Any]:
        """Benchmark a single configuration.

        Args:
            resolution: Spatial resolution (H = W = resolution).
            provider_priority: Provider priority list.
            batch_size: Batch size for inference.
            fp16: Enable FP16 inference.

        Returns:
            Dict with benchmark results.
        """
        config = EngineConfig(height=resolution, width=resolution, num_planes=5)
        gpu_config = GPUConfig(
            provider_priority=provider_priority,
            enable_fp16=fp16,
            max_batch_size=max(batch_size, 1),
            preallocate_io_bindings=True,
        )

        engine = GPUEngineAPI(gpu_config)
        status = engine.init(self.model_path, config)

        if status != Status.OK:
            return {
                'resolution': resolution,
                'provider': provider_priority[0],
                'batch_size': batch_size,
                'fp16': fp16,
                'status': 'failed',
                'error': engine.last_error,
            }

        # Generate test data
        H, W = resolution, resolution
        rgb_list = [np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
                    for _ in range(batch_size)]
        depth_list = [np.random.rand(H, W).astype(np.float32)
                      for _ in range(batch_size)]

        # Warmup
        for _ in range(self.num_warmup):
            if batch_size == 1:
                engine.generate_hologram(rgb_list[0], depth_list[0])
            else:
                engine.generate_hologram_batch(rgb_list, depth_list)

        # Benchmark
        latencies = []
        for _ in range(self.num_runs):
            t0 = time.perf_counter()
            if batch_size == 1:
                status, _ = engine.generate_hologram(rgb_list[0], depth_list[0])
            else:
                status, _ = engine.generate_hologram_batch(rgb_list, depth_list)
            t1 = time.perf_counter()

            if status == Status.OK:
                latencies.append((t1 - t0) * 1000)

        engine.shutdown()

        if not latencies:
            return {
                'resolution': resolution,
                'provider': provider_priority[0],
                'batch_size': batch_size,
                'fp16': fp16,
                'status': 'no_valid_runs',
            }

        arr = np.array(latencies)
        result = {
            'resolution': resolution,
            'provider': provider_priority[0],
            'batch_size': batch_size,
            'fp16': fp16,
            'status': 'ok',
            'mean_ms': float(np.mean(arr)),
            'std_ms': float(np.std(arr)),
            'min_ms': float(np.min(arr)),
            'max_ms': float(np.max(arr)),
            'median_ms': float(np.median(arr)),
            'p95_ms': float(np.percentile(arr, 95)),
            'p99_ms': float(np.percentile(arr, 99)),
            'fps': float(1000.0 / np.mean(arr)),
            'fps_per_frame': float(1000.0 * batch_size / np.mean(arr)) if batch_size > 1 else float(1000.0 / np.mean(arr)),
            'n_runs': len(latencies),
        }

        return result

    def benchmark_cpu(self, resolution: int, batch_size: int = 1) -> Dict[str, Any]:
        """Benchmark CPU-only inference for comparison."""
        config = EngineConfig(height=resolution, width=resolution, num_planes=5)
        engine = EngineAPI()
        status = engine.init(self.model_path, config)

        if status != Status.OK:
            return {
                'resolution': resolution,
                'provider': 'cpu',
                'batch_size': batch_size,
                'fp16': False,
                'status': 'failed',
                'error': engine.last_error,
            }

        H, W = resolution, resolution
        rgb = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        depth = np.random.rand(H, W).astype(np.float32)

        # Warmup
        for _ in range(self.num_warmup):
            engine.generate_hologram(rgb, depth)

        # Benchmark
        latencies = []
        for _ in range(self.num_runs):
            t0 = time.perf_counter()
            status, _ = engine.generate_hologram(rgb, depth)
            t1 = time.perf_counter()
            if status == Status.OK:
                latencies.append((t1 - t0) * 1000)

        engine.shutdown()

        if not latencies:
            return {
                'resolution': resolution,
                'provider': 'cpu',
                'batch_size': batch_size,
                'fp16': False,
                'status': 'no_valid_runs',
            }

        arr = np.array(latencies)
        return {
            'resolution': resolution,
            'provider': 'cpu',
            'batch_size': batch_size,
            'fp16': False,
            'status': 'ok',
            'mean_ms': float(np.mean(arr)),
            'std_ms': float(np.std(arr)),
            'min_ms': float(np.min(arr)),
            'max_ms': float(np.max(arr)),
            'median_ms': float(np.median(arr)),
            'p95_ms': float(np.percentile(arr, 95)),
            'p99_ms': float(np.percentile(arr, 99)),
            'fps': float(1000.0 / np.mean(arr)),
            'fps_per_frame': float(1000.0 / np.mean(arr)),
            'n_runs': len(latencies),
        }

    def run_all(self, resolutions: List[int], batch_sizes: List[int],
                skip_cpu: bool = False, skip_cuda: bool = False,
                skip_trt: bool = False, skip_fp16: bool = False) -> List[Dict[str, Any]]:
        """Run all benchmark configurations."""
        available = detect_available_providers()
        results = []
        total = 0
        current = 0

        # Count total benchmarks
        for res in resolutions:
            for bs in batch_sizes:
                if not skip_cpu:
                    total += 1
                if not skip_cuda and 'cuda' in available:
                    total += 1  # FP32
                    if not skip_fp16:
                        total += 1  # FP16
                if not skip_trt and 'tensorrt' in available:
                    total += 1  # FP32
                    if not skip_fp16:
                        total += 1  # FP16

        print(f"\n{'=' * 80}")
        print(f"DeepCGHEngine GPU Benchmark")
        print(f"{'=' * 80}")
        print(f"  Model:      {self.model_path}")
        print(f"  Warmup:     {self.num_warmup} iterations")
        print(f"  Runs:       {self.num_runs} iterations")
        print(f"  Available providers: {', '.join(available)}")
        print(f"  Total benchmarks: {total}")
        print(f"{'=' * 80}\n")

        for res in resolutions:
            for bs in batch_sizes:
                # CPU baseline
                if not skip_cpu:
                    current += 1
                    print(f"[{current}/{total}] CPU | {res}x{res} | batch={bs} | FP32")
                    result = self.benchmark_cpu(res, bs)
                    results.append(result)
                    self._print_result(result)

                # CUDA FP32
                if not skip_cuda and 'cuda' in available:
                    current += 1
                    print(f"[{current}/{total}] CUDA | {res}x{res} | batch={bs} | FP32")
                    result = self.benchmark_single(res, ['cuda', 'cpu'], bs, fp16=False)
                    results.append(result)
                    self._print_result(result)

                    # CUDA FP16
                    if not skip_fp16:
                        current += 1
                        print(f"[{current}/{total}] CUDA | {res}x{res} | batch={bs} | FP16")
                        result = self.benchmark_single(res, ['cuda', 'cpu'], bs, fp16=True)
                        results.append(result)
                        self._print_result(result)

                # TensorRT FP32
                if not skip_trt and 'tensorrt' in available:
                    current += 1
                    print(f"[{current}/{total}] TensorRT | {res}x{res} | batch={bs} | FP32")
                    result = self.benchmark_single(res, ['tensorrt', 'cuda', 'cpu'], bs, fp16=False)
                    results.append(result)
                    self._print_result(result)

                    # TensorRT FP16
                    if not skip_fp16:
                        current += 1
                        print(f"[{current}/{total}] TensorRT | {res}x{res} | batch={bs} | FP16")
                        result = self.benchmark_single(res, ['tensorrt', 'cuda', 'cpu'], bs, fp16=True)
                        results.append(result)
                        self._print_result(result)

        self.results = results
        return results

    def _print_result(self, result: Dict[str, Any]):
        """Print a single benchmark result."""
        if result['status'] != 'ok':
            print(f"  FAILED: {result.get('error', 'unknown error')}")
            return
        print(f"  Mean: {result['mean_ms']:.2f} ms | "
              f"Median: {result['median_ms']:.2f} ms | "
              f"P95: {result['p95_ms']:.2f} ms | "
              f"FPS: {result['fps']:.1f}")

    def print_summary_table(self):
        """Print a formatted summary table of all results."""
        if not self.results:
            print("No benchmark results to display.")
            return

        print(f"\n{'=' * 110}")
        print(f"{'DeepCGHEngine GPU Benchmark Summary':^110}")
        print(f"{'=' * 110}")

        header = (f"{'Provider':<12} {'Resolution':<12} {'Batch':<8} {'Precision':<10} "
                  f"{'Mean(ms)':<10} {'Median(ms)':<12} {'P95(ms)':<10} "
                  f"{'FPS':<10} {'Status':<10}")
        print(header)
        print('-' * 110)

        for r in self.results:
            if r['status'] == 'ok':
                precision = 'FP16' if r['fp16'] else 'FP32'
                line = (f"{r['provider']:<12} {r['resolution']}x{r['resolution']:<7} "
                        f"{r['batch_size']:<8} {precision:<10} "
                        f"{r['mean_ms']:<10.2f} {r['median_ms']:<12.2f} "
                        f"{r['p95_ms']:<10.2f} {r['fps']:<10.1f} "
                        f"{'OK':<10}")
            else:
                line = (f"{r.get('provider', '?'):<12} {r['resolution']}x{r['resolution']:<7} "
                        f"{r['batch_size']:<8} {'FP16' if r['fp16'] else 'FP32':<10} "
                        f"{'-':<10} {'-':<12} {'-':<10} {'-':<10} "
                        f"{r['status']:<10}")
            print(line)

        print('=' * 110)

        # Speedup comparison
        self._print_speedup_table()

    def _print_speedup_table(self):
        """Print speedup comparison vs CPU baseline."""
        cpu_results = {f"{r['resolution']}_{r['batch_size']}": r
                       for r in self.results
                       if r['provider'] == 'cpu' and r['status'] == 'ok'}

        gpu_results = [r for r in self.results
                       if r['provider'] != 'cpu' and r['status'] == 'ok']

        if not cpu_results or not gpu_results:
            return

        print(f"\n{'=' * 80}")
        print(f"{'Speedup vs CPU Baseline':^80}")
        print(f"{'=' * 80}")

        header = f"{'Provider':<12} {'Resolution':<12} {'Batch':<8} {'Precision':<10} {'Speedup':<10}"
        print(header)
        print('-' * 80)

        for r in gpu_results:
            key = f"{r['resolution']}_{r['batch_size']}"
            cpu_r = cpu_results.get(key)
            if cpu_r:
                speedup = cpu_r['mean_ms'] / r['mean_ms']
                precision = 'FP16' if r['fp16'] else 'FP32'
                line = (f"{r['provider']:<12} {r['resolution']}x{r['resolution']:<7} "
                        f"{r['batch_size']:<8} {precision:<10} {speedup:<10.2f}x")
                print(line)

        print('=' * 80)

    def save_results(self, output_path: str):
        """Save benchmark results to JSON."""
        output = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'num_warmup': self.num_warmup,
            'num_runs': self.num_runs,
            'available_providers': detect_available_providers(),
            'results': self.results,
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {output_path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GPU benchmark for DeepCGHEngine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default benchmark
  python benchmark_gpu.py

  # Custom model and output
  python benchmark_gpu.py --model model.onnx --output results.json

  # Specific configurations
  python benchmark_gpu.py --resolutions 256 512 --batch-sizes 1 4 8

  # Skip TensorRT
  python benchmark_gpu.py --skip-trt

  # More iterations for stable results
  python benchmark_gpu.py --warmup 10 --runs 100
""",
    )

    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to ONNX model (default: auto-detect in models/)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save JSON results (default: models/gpu_benchmark_results.json)",
    )
    parser.add_argument(
        "--resolutions", type=int, nargs='+', default=[256, 512, 1024],
        help="Resolutions to test (default: 256 512 1024)",
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs='+', default=[1, 4, 8, 16],
        help="Batch sizes to test (default: 1 4 8 16)",
    )
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="Number of warmup iterations (default: 5)",
    )
    parser.add_argument(
        "--runs", type=int, default=50,
        help="Number of benchmark iterations (default: 50)",
    )
    parser.add_argument(
        "--skip-cpu", action="store_true", default=False,
        help="Skip CPU benchmark",
    )
    parser.add_argument(
        "--skip-cuda", action="store_true", default=False,
        help="Skip CUDA benchmark",
    )
    parser.add_argument(
        "--skip-trt", action="store_true", default=False,
        help="Skip TensorRT benchmark",
    )
    parser.add_argument(
        "--skip-fp16", action="store_true", default=False,
        help="Skip FP16 benchmarks",
    )

    args = parser.parse_args()

    # Determine model path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)

    if args.model:
        model_path = args.model
    else:
        default_model = os.path.join(base_dir, "models", "deepcgh_unet.onnx")
        if not os.path.exists(default_model):
            print(f"[ERROR] Default model not found: {default_model}")
            print("Use --model to specify the model path.")
            sys.exit(1)
        model_path = default_model

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(base_dir, "models", "gpu_benchmark_results.json")

    # Print system info
    print("\nSystem Information:")
    print(f"  ONNX Runtime version: {ort.__version__}")
    available = detect_available_providers()
    print(f"  Available providers: {', '.join(available)}")

    try:
        import onnxruntime as ort_
        print(f"  ORT build info: {ort_.get_build_info()}")
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
    except ImportError:
        pass

    # Run benchmarks
    runner = GPUBenchmarkRunner(model_path, args.warmup, args.runs)
    runner.run_all(
        resolutions=args.resolutions,
        batch_sizes=args.batch_sizes,
        skip_cpu=args.skip_cpu,
        skip_cuda=args.skip_cuda,
        skip_trt=args.skip_trt,
        skip_fp16=args.skip_fp16,
    )

    # Print summary
    runner.print_summary_table()

    # Save results
    runner.save_results(output_path)


if __name__ == "__main__":
    try:
        import onnxruntime as ort
    except ImportError:
        print("[ERROR] onnxruntime is required: pip install onnxruntime")
        sys.exit(1)
    main()
