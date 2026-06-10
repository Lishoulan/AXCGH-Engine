#!/usr/bin/env python3
"""
optimize_tensorrt.py — TensorRT optimization for DeepCGH ONNX models.

Converts ONNX FP32 models to TensorRT engines with support for:
  - FP16 precision
  - INT8 precision with calibration
  - Dynamic batch size
  - Workspace size configuration
  - Engine caching

Saves optimized engine as .trt file and benchmarks TRT vs ONNX.

Usage:
    # Default: optimize with FP16
    py -3.10 optimize_tensorrt.py

    # INT8 with calibration
    py -3.10 optimize_tensorrt.py --precision int8 --calibration-samples 200

    # Dynamic batch size
    py -3.10 optimize_tensorrt.py --min-batch 1 --opt-batch 4 --max-batch 16

    # Custom workspace and output
    py -3.10 optimize_tensorrt.py --workspace 4096 --output model_fp16.trt

    # Benchmark only (use existing .trt engine)
    py -3.10 optimize_tensorrt.py --benchmark-only --trt-engine model_fp16.trt
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("[ERROR] onnxruntime is required: pip install onnxruntime")
    sys.exit(1)


# ===========================================================================
# TensorRT Builder (via ONNX Runtime TRT EP or direct tensorrt bindings)
# ===========================================================================

class TensorRTOptimizer:
    """Convert ONNX models to TensorRT engines.

    Supports two paths:
      1. Direct TensorRT Python API (if tensorrt package is installed)
      2. ONNX Runtime TensorRT Execution Provider (fallback)
    """

    def __init__(self, model_path: str, device_id: int = 0):
        self.model_path = model_path
        self.device_id = device_id
        self._trt_available = False
        self._ort_trt_available = False

        # Check direct TensorRT availability
        try:
            import tensorrt as trt
            self._trt = trt
            self._trt_available = True
            self._logger = trt.Logger(trt.Logger.WARNING)
        except ImportError:
            self._trt = None
            self._logger = None

        # Check ORT TensorRT EP availability
        try:
            providers = ort.get_available_providers()
            self._ort_trt_available = 'TensorrtExecutionProvider' in providers
        except Exception:
            self._ort_trt_available = False

    @property
    def is_available(self) -> bool:
        return self._trt_available or self._ort_trt_available

    def optimize_direct(self, output_path: str,
                        precision: str = 'fp16',
                        workspace_mb: int = 1024,
                        min_batch: int = 1,
                        opt_batch: int = 1,
                        max_batch: int = 1,
                        calibration_samples: int = 100,
                        int8_cache_path: str = '') -> bool:
        """Optimize using direct TensorRT Python API.

        Args:
            output_path: Path to save .trt engine file.
            precision: 'fp32', 'fp16', or 'int8'.
            workspace_mb: Max workspace size in MB.
            min_batch: Minimum batch size for dynamic shape.
            opt_batch: Optimal batch size for dynamic shape.
            max_batch: Maximum batch size for dynamic shape.
            calibration_samples: Number of calibration samples for INT8.
            int8_cache_path: Path for INT8 calibration cache.

        Returns:
            True if optimization succeeded.
        """
        if not self._trt_available:
            print("[ERROR] TensorRT Python bindings not available.")
            print("Install with: pip install tensorrt")
            return False

        import tensorrt as trt

        print(f"\n{'=' * 70}")
        print(f"TensorRT Direct Optimization")
        print(f"{'=' * 70}")
        print(f"  Input:       {self.model_path}")
        print(f"  Output:      {output_path}")
        print(f"  Precision:   {precision.upper()}")
        print(f"  Workspace:   {workspace_mb} MB")
        print(f"  Batch:       min={min_batch}, opt={opt_batch}, max={max_batch}")

        try:
            # Create builder
            builder = trt.Builder(self._logger)
            network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            network = builder.create_network(network_flags)
            parser = trt.OnnxParser(network, self._logger)

            # Parse ONNX model
            print("\n[1/5] Parsing ONNX model...")
            with open(self.model_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        print(f"  [ERROR] {parser.get_error(i)}")
                    return False
            print("  ONNX model parsed successfully.")

            # Configure builder
            print("\n[2/5] Configuring builder...")
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)

            # Set precision flags
            if precision == 'fp16':
                config.set_flag(trt.BuilderFlag.FP16)
                print("  FP16 precision enabled.")
            elif precision == 'int8':
                config.set_flag(trt.BuilderFlag.INT8)
                print("  INT8 precision enabled.")

                # Set up INT8 calibrator
                if calibration_samples > 0:
                    calibrator = DeepCGHInt8Calibrator(
                        model_path=self.model_path,
                        num_samples=calibration_samples,
                        cache_path=int8_cache_path,
                    )
                    config.int8_calibrator = calibrator
                    print(f"  INT8 calibrator configured with {calibration_samples} samples.")

            # Set dynamic batch profile
            input_tensor = network.get_input(0)
            input_name = input_tensor.name
            input_shape = input_tensor.shape

            has_dynamic = any(isinstance(d, trt.TensorDimension) or d == -1 for d in input_shape)

            if has_dynamic or min_batch != max_batch:
                print(f"  Dynamic batch: min={min_batch}, opt={opt_batch}, max={max_batch}")
                profile = builder.create_optimization_profile()

                # Determine spatial dims
                spatial = [d if d > 0 else 256 for d in input_shape[1:]]

                min_shape = (min_batch,) + tuple(spatial)
                opt_shape = (opt_batch,) + tuple(spatial)
                max_shape = (max_batch,) + tuple(spatial)

                profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                config.add_optimization_profile(profile)
            else:
                print(f"  Static batch size: {opt_batch}")

            # Build engine
            print("\n[3/5] Building TensorRT engine (this may take a while)...")
            serialized_engine = builder.build_serialized_network(network, config)

            if serialized_engine is None:
                print("  [ERROR] Failed to build TensorRT engine.")
                return False

            # Save engine
            print(f"\n[4/5] Saving engine to {output_path}...")
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(serialized_engine)

            engine_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  Engine saved: {engine_size_mb:.2f} MB")

            # Report
            print(f"\n[5/5] Optimization complete.")
            onnx_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
            print(f"  ONNX model size:   {onnx_size_mb:.2f} MB")
            print(f"  TRT engine size:   {engine_size_mb:.2f} MB")
            if engine_size_mb < onnx_size_mb:
                reduction = (1.0 - engine_size_mb / onnx_size_mb) * 100
                print(f"  Size reduction:    {reduction:.1f}%")

            return True

        except Exception as e:
            print(f"[ERROR] TensorRT optimization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def optimize_via_ort(self, output_dir: str = '',
                         precision: str = 'fp16',
                         workspace_mb: int = 1024,
                         min_batch: int = 1,
                         opt_batch: int = 1,
                         max_batch: int = 1,
                         calibration_samples: int = 100) -> bool:
        """Optimize using ONNX Runtime TensorRT Execution Provider.

        This approach lets ORT build and cache the TRT engine automatically.

        Args:
            output_dir: Directory for TRT engine cache.
            precision: 'fp32', 'fp16', or 'int8'.
            workspace_mb: Max workspace size in MB.
            min_batch: Minimum batch size.
            opt_batch: Optimal batch size.
            max_batch: Maximum batch size.
            calibration_samples: Number of calibration samples for INT8.

        Returns:
            True if optimization succeeded.
        """
        if not self._ort_trt_available:
            print("[ERROR] TensorRT Execution Provider not available in ONNX Runtime.")
            print("Install with: pip install onnxruntime-gpu")
            return False

        print(f"\n{'=' * 70}")
        print(f"TensorRT Optimization via ONNX Runtime EP")
        print(f"{'=' * 70}")
        print(f"  Input:       {self.model_path}")
        print(f"  Precision:   {precision.upper()}")
        print(f"  Workspace:   {workspace_mb} MB")
        print(f"  Batch:       min={min_batch}, opt={opt_batch}, max={max_batch}")

        try:
            # Configure TRT EP options
            trt_options = {
                'device_id': self.device_id,
                'trt_max_workspace_size': workspace_mb * 1024 * 1024,
                'trt_fp16_enable': precision in ('fp16', 'int8'),
                'trt_int8_enable': precision == 'int8',
            }

            if output_dir:
                trt_options['trt_engine_cache_enable'] = True
                trt_options['trt_engine_cache_path'] = output_dir
                os.makedirs(output_dir, exist_ok=True)

            # Create session with TRT EP
            print("\n[1/3] Creating ONNX Runtime session with TensorRT EP...")
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            providers = [
                ('TensorrtExecutionProvider', trt_options),
                ('CUDAExecutionProvider', {
                    'device_id': self.device_id,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                }),
                'CPUExecutionProvider',
            ]

            session = ort.InferenceSession(self.model_path, opts, providers=providers)

            # Warm up to trigger TRT engine build
            print("\n[2/3] Building TensorRT engine (warmup inference)...")
            input_info = session.get_inputs()[0]
            input_name = input_info.name
            input_shape = [opt_batch if isinstance(d, str) or d <= 0 else d
                          for d in input_info.shape]

            dummy_input = np.random.uniform(0.0, 1.0, input_shape).astype(np.float32)
            output_names = [o.name for o in session.get_outputs()]

            # First run triggers TRT engine build
            t0 = time.perf_counter()
            session.run(output_names, {input_name: dummy_input})
            t1 = time.perf_counter()
            print(f"  First inference (engine build): {(t1 - t0) * 1000:.1f} ms")

            # Second run uses cached engine
            t0 = time.perf_counter()
            session.run(output_names, {input_name: dummy_input})
            t1 = time.perf_counter()
            print(f"  Second inference (cached): {(t1 - t0) * 1000:.1f} ms")

            # Test different batch sizes if dynamic
            if min_batch != max_batch:
                print("\n[3/3] Testing dynamic batch sizes...")
                for bs in [min_batch, opt_batch, max_batch]:
                    shape = list(input_shape)
                    shape[0] = bs
                    dummy = np.random.uniform(0.0, 1.0, shape).astype(np.float32)
                    t0 = time.perf_counter()
                    session.run(output_names, {input_name: dummy})
                    t1 = time.perf_counter()
                    print(f"  Batch {bs}: {(t1 - t0) * 1000:.1f} ms")
            else:
                print("\n[3/3] Static batch size, no dynamic shape testing needed.")

            # Report cache location
            if output_dir:
                cache_files = [f for f in os.listdir(output_dir) if f.endswith('.engine') or f.endswith('.trt')]
                if cache_files:
                    print(f"\n  TRT engine cache files in {output_dir}:")
                    for f in cache_files:
                        size_mb = os.path.getsize(os.path.join(output_dir, f)) / (1024 * 1024)
                        print(f"    {f} ({size_mb:.2f} MB)")

            print("\n  Optimization complete.")
            return True

        except Exception as e:
            print(f"[ERROR] ORT TRT optimization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def benchmark_trt(self, trt_engine_path: str = '',
                      num_warmup: int = 5, num_runs: int = 50,
                      batch_size: int = 1) -> Dict[str, Any]:
        """Benchmark TensorRT engine vs ONNX model.

        Args:
            trt_engine_path: Path to .trt engine file (empty = use ORT TRT EP).
            num_warmup: Number of warmup iterations.
            num_runs: Number of benchmark iterations.
            batch_size: Batch size for inference.

        Returns:
            Dict with benchmark results.
        """
        results = {}

        # Benchmark ONNX model (CPU baseline)
        print(f"\n{'=' * 70}")
        print(f"Benchmark: ONNX Model (CPU)")
        print(f"{'=' * 70}")

        try:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 4

            session = ort.InferenceSession(self.model_path, opts, providers=['CPUExecutionProvider'])
            input_info = session.get_inputs()[0]
            input_name = input_info.name
            input_shape = [batch_size if isinstance(d, str) or d <= 0 else d
                          for d in input_info.shape]
            output_names = [o.name for o in session.get_outputs()]

            dummy = np.random.uniform(0.0, 1.0, input_shape).astype(np.float32)

            for _ in range(num_warmup):
                session.run(output_names, {input_name: dummy})

            latencies = []
            for _ in range(num_runs):
                t0 = time.perf_counter()
                session.run(output_names, {input_name: dummy})
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)

            arr = np.array(latencies)
            results['onnx_cpu'] = {
                'mean_ms': float(np.mean(arr)),
                'median_ms': float(np.median(arr)),
                'min_ms': float(np.min(arr)),
                'max_ms': float(np.max(arr)),
                'p95_ms': float(np.percentile(arr, 95)),
                'fps': float(1000.0 / np.mean(arr)),
            }
            print(f"  Mean: {results['onnx_cpu']['mean_ms']:.2f} ms | "
                  f"Median: {results['onnx_cpu']['median_ms']:.2f} ms | "
                  f"FPS: {results['onnx_cpu']['fps']:.1f}")

            del session
        except Exception as e:
            print(f"  [ERROR] ONNX CPU benchmark failed: {e}")
            results['onnx_cpu'] = {'error': str(e)}

        # Benchmark ONNX model (CUDA)
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            print(f"\n{'=' * 70}")
            print(f"Benchmark: ONNX Model (CUDA)")
            print(f"{'=' * 70}")

            try:
                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                session = ort.InferenceSession(
                    self.model_path, opts,
                    providers=[
                        ('CUDAExecutionProvider', {
                            'device_id': self.device_id,
                            'arena_extend_strategy': 'kNextPowerOfTwo',
                            'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        }),
                        'CPUExecutionProvider',
                    ]
                )

                for _ in range(num_warmup):
                    session.run(output_names, {input_name: dummy})

                latencies = []
                for _ in range(num_runs):
                    t0 = time.perf_counter()
                    session.run(output_names, {input_name: dummy})
                    t1 = time.perf_counter()
                    latencies.append((t1 - t0) * 1000)

                arr = np.array(latencies)
                results['onnx_cuda'] = {
                    'mean_ms': float(np.mean(arr)),
                    'median_ms': float(np.median(arr)),
                    'min_ms': float(np.min(arr)),
                    'max_ms': float(np.max(arr)),
                    'p95_ms': float(np.percentile(arr, 95)),
                    'fps': float(1000.0 / np.mean(arr)),
                }
                print(f"  Mean: {results['onnx_cuda']['mean_ms']:.2f} ms | "
                      f"Median: {results['onnx_cuda']['median_ms']:.2f} ms | "
                      f"FPS: {results['onnx_cuda']['fps']:.1f}")

                del session
            except Exception as e:
                print(f"  [ERROR] ONNX CUDA benchmark failed: {e}")
                results['onnx_cuda'] = {'error': str(e)}

        # Benchmark TensorRT engine
        if trt_engine_path and os.path.exists(trt_engine_path):
            print(f"\n{'=' * 70}")
            print(f"Benchmark: TensorRT Engine ({trt_engine_path})")
            print(f"{'=' * 70}")

            if self._trt_available:
                try:
                    results['trt'] = self._benchmark_trt_direct(
                        trt_engine_path, num_warmup, num_runs, batch_size
                    )
                except Exception as e:
                    print(f"  [ERROR] TRT direct benchmark failed: {e}")
                    results['trt'] = {'error': str(e)}
            else:
                print("  [WARN] Direct TRT API not available, using ORT TRT EP.")

        # Benchmark via ORT TRT EP
        if self._ort_trt_available:
            print(f"\n{'=' * 70}")
            print(f"Benchmark: ONNX Runtime + TensorRT EP")
            print(f"{'=' * 70}")

            try:
                trt_cache = os.path.join(os.path.dirname(self.model_path), 'trt_cache')
                os.makedirs(trt_cache, exist_ok=True)

                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                session = ort.InferenceSession(
                    self.model_path, opts,
                    providers=[
                        ('TensorrtExecutionProvider', {
                            'device_id': self.device_id,
                            'trt_max_workspace_size': 1024 * 1024 * 1024,
                            'trt_fp16_enable': True,
                            'trt_engine_cache_enable': True,
                            'trt_engine_cache_path': trt_cache,
                        }),
                        ('CUDAExecutionProvider', {
                            'device_id': self.device_id,
                        }),
                        'CPUExecutionProvider',
                    ]
                )

                input_info = session.get_inputs()[0]
                input_name = input_info.name
                input_shape = [batch_size if isinstance(d, str) or d <= 0 else d
                              for d in input_info.shape]
                output_names = [o.name for o in session.get_outputs()]
                dummy = np.random.uniform(0.0, 1.0, input_shape).astype(np.float32)

                # Warmup (first run builds TRT engine)
                print("  Building TRT engine (warmup)...")
                session.run(output_names, {input_name: dummy})

                for _ in range(num_warmup - 1):
                    session.run(output_names, {input_name: dummy})

                latencies = []
                for _ in range(num_runs):
                    t0 = time.perf_counter()
                    session.run(output_names, {input_name: dummy})
                    t1 = time.perf_counter()
                    latencies.append((t1 - t0) * 1000)

                arr = np.array(latencies)
                results['ort_trt'] = {
                    'mean_ms': float(np.mean(arr)),
                    'median_ms': float(np.median(arr)),
                    'min_ms': float(np.min(arr)),
                    'max_ms': float(np.max(arr)),
                    'p95_ms': float(np.percentile(arr, 95)),
                    'fps': float(1000.0 / np.mean(arr)),
                }
                print(f"  Mean: {results['ort_trt']['mean_ms']:.2f} ms | "
                      f"Median: {results['ort_trt']['median_ms']:.2f} ms | "
                      f"FPS: {results['ort_trt']['fps']:.1f}")

                del session
            except Exception as e:
                print(f"  [ERROR] ORT TRT benchmark failed: {e}")
                results['ort_trt'] = {'error': str(e)}

        # Print speedup summary
        self._print_speedup_summary(results)

        return results

    def _benchmark_trt_direct(self, engine_path: str,
                              num_warmup: int, num_runs: int,
                              batch_size: int) -> Dict[str, Any]:
        """Benchmark a .trt engine file using direct TensorRT API."""
        import tensorrt as trt
        import cuda as cuda

        # Load engine
        runtime = trt.Runtime(self._logger)
        with open(engine_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()

        # Allocate buffers
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            shape = engine.get_tensor_shape(name)
            shape = tuple(batch_size if d == -1 else d for d in shape)
            size = np.prod(shape)

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
                context.set_tensor_address(name, int(device_mem))
            else:
                outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
                context.set_tensor_address(name, int(device_mem))

        # Fill input with random data
        for inp in inputs:
            np.copyto(inp['host'], np.random.uniform(0.0, 1.0, inp['shape']).ravel())

        # Warmup
        for inp in inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
        context.execute_async_v3(stream_handle=stream.handle)
        for out in outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
        stream.synchronize()

        # Benchmark
        latencies = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            for inp in inputs:
                cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
            context.execute_async_v3(stream_handle=stream.handle)
            for out in outputs:
                cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
            stream.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

        arr = np.array(latencies)
        result = {
            'mean_ms': float(np.mean(arr)),
            'median_ms': float(np.median(arr)),
            'min_ms': float(np.min(arr)),
            'max_ms': float(np.max(arr)),
            'p95_ms': float(np.percentile(arr, 95)),
            'fps': float(1000.0 / np.mean(arr)),
        }
        print(f"  Mean: {result['mean_ms']:.2f} ms | "
              f"Median: {result['median_ms']:.2f} ms | "
              f"FPS: {result['fps']:.1f}")

        return result

    def _print_speedup_summary(self, results: Dict[str, Any]):
        """Print speedup comparison summary."""
        cpu_result = results.get('onnx_cpu', {})
        if 'error' in cpu_result or 'mean_ms' not in cpu_result:
            return

        cpu_ms = cpu_result['mean_ms']

        print(f"\n{'=' * 70}")
        print(f"{'Speedup vs CPU Baseline':^70}")
        print(f"{'=' * 70}")

        header = f"{'Backend':<25} {'Mean (ms)':<12} {'FPS':<10} {'Speedup':<10}"
        print(header)
        print('-' * 70)

        print(f"{'ONNX CPU (baseline)':<25} {cpu_ms:<12.2f} {cpu_result['fps']:<10.1f} {'1.00x':<10}")

        for key in ['onnx_cuda', 'ort_trt', 'trt']:
            r = results.get(key, {})
            if 'mean_ms' in r:
                speedup = cpu_ms / r['mean_ms']
                label_map = {
                    'onnx_cuda': 'ONNX CUDA',
                    'ort_trt': 'ORT + TensorRT',
                    'trt': 'TensorRT Direct',
                }
                label = label_map.get(key, key)
                print(f"{label:<25} {r['mean_ms']:<12.2f} {r['fps']:<10.1f} {speedup:<10.2f}x")

        print('=' * 70)


# ===========================================================================
# INT8 Calibrator
# ===========================================================================

class DeepCGHInt8Calibrator:
    """INT8 calibrator for DeepCGH models using TensorRT direct API.

    Generates synthetic calibration data matching the model input shape.
    """

    def __init__(self, model_path: str, num_samples: int = 100,
                 cache_path: str = '', input_range: tuple = (0.0, 1.0)):
        self.num_samples = num_samples
        self.cache_path = cache_path
        self.input_range = input_range
        self.current_idx = 0

        # Read input shape from ONNX model
        session = ort.InferenceSession(model_path)
        input_info = session.get_inputs()[0]
        self.input_name = input_info.name
        self.input_shape = [
            1 if isinstance(d, str) else d for d in input_info.shape
        ]
        del session

        # Pre-generate calibration data
        lo, hi = self.input_range
        self._data = [
            np.random.uniform(lo, hi, self.input_shape).astype(np.float32)
            for _ in range(num_samples)
        ]

        # Try to load cache
        self._cache = None
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    self._cache = f.read()
            except Exception:
                self._cache = None

    def get_batch_size(self):
        return self.input_shape[0]

    def get_batch(self, names):
        if self.current_idx >= self.num_samples:
            return None

        batch = self._data[self.current_idx]
        self.current_idx += 1
        return [batch.ravel()]

    def read_calibration_cache(self):
        return self._cache

    def write_calibration_cache(self, cache):
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path) or '.', exist_ok=True)
            with open(self.cache_path, 'wb') as f:
                f.write(cache)


# ===========================================================================
# Model Size Helper
# ===========================================================================

def get_model_size_mb(path: str) -> float:
    """Return file size in MB."""
    return os.path.getsize(path) / (1024 * 1024)


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TensorRT optimization for DeepCGH ONNX models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: FP16 optimization
  python optimize_tensorrt.py

  # INT8 with calibration
  python optimize_tensorrt.py --precision int8 --calibration-samples 200

  # Dynamic batch size
  python optimize_tensorrt.py --min-batch 1 --opt-batch 4 --max-batch 16

  # Benchmark only
  python optimize_tensorrt.py --benchmark-only --trt-engine model_fp16.trt

  # Use ORT TRT EP instead of direct TRT
  python optimize_tensorrt.py --use-ort-ep
""",
    )

    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to FP32 ONNX model (default: auto-detect in models/)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save .trt engine file (default: <model>_<precision>.trt)",
    )
    parser.add_argument(
        "--precision", type=str, choices=['fp32', 'fp16', 'int8'], default='fp16',
        help="Precision: fp32, fp16, or int8 (default: fp16)",
    )
    parser.add_argument(
        "--workspace", type=int, default=1024,
        help="Max workspace size in MB (default: 1024)",
    )
    parser.add_argument(
        "--min-batch", type=int, default=1,
        help="Minimum batch size for dynamic shape (default: 1)",
    )
    parser.add_argument(
        "--opt-batch", type=int, default=1,
        help="Optimal batch size for dynamic shape (default: 1)",
    )
    parser.add_argument(
        "--max-batch", type=int, default=1,
        help="Maximum batch size for dynamic shape (default: 1)",
    )
    parser.add_argument(
        "--calibration-samples", type=int, default=100,
        help="Number of calibration samples for INT8 (default: 100)",
    )
    parser.add_argument(
        "--int8-cache", type=str, default='',
        help="Path for INT8 calibration cache file",
    )
    parser.add_argument(
        "--device-id", type=int, default=0,
        help="GPU device ID (default: 0)",
    )
    parser.add_argument(
        "--use-ort-ep", action="store_true", default=False,
        help="Use ONNX Runtime TensorRT EP instead of direct TRT API",
    )
    parser.add_argument(
        "--benchmark-only", action="store_true", default=False,
        help="Only benchmark, skip optimization",
    )
    parser.add_argument(
        "--trt-engine", type=str, default='',
        help="Path to existing .trt engine for benchmarking",
    )
    parser.add_argument(
        "--benchmark-warmup", type=int, default=5,
        help="Benchmark warmup iterations (default: 5)",
    )
    parser.add_argument(
        "--benchmark-runs", type=int, default=50,
        help="Benchmark iterations (default: 50)",
    )
    parser.add_argument(
        "--save-results", type=str, default='',
        help="Path to save benchmark results as JSON",
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
        base, _ = os.path.splitext(model_path)
        output_path = f"{base}_{args.precision}.trt"

    # Create optimizer
    optimizer = TensorRTOptimizer(model_path, device_id=args.device_id)

    if not optimizer.is_available:
        print("[ERROR] No TensorRT support available.")
        print("Install one of:")
        print("  - pip install tensorrt          (direct TRT API)")
        print("  - pip install onnxruntime-gpu   (ORT TRT EP)")
        sys.exit(1)

    # Print system info
    print("\nSystem Information:")
    print(f"  ONNX Runtime version: {ort.__version__}")
    available = ort.get_available_providers()
    print(f"  Available providers: {', '.join(available)}")
    print(f"  Direct TRT API: {'available' if optimizer._trt_available else 'not available'}")
    print(f"  ORT TRT EP: {'available' if optimizer._ort_trt_available else 'not available'}")

    if optimizer._trt_available:
        print(f"  TensorRT version: {optimizer._trt.__version__}")

    # Run optimization or benchmark
    if not args.benchmark_only:
        if args.use_ort_ep:
            cache_dir = os.path.join(os.path.dirname(model_path), 'trt_cache')
            success = optimizer.optimize_via_ort(
                output_dir=cache_dir,
                precision=args.precision,
                workspace_mb=args.workspace,
                min_batch=args.min_batch,
                opt_batch=args.opt_batch,
                max_batch=args.max_batch,
                calibration_samples=args.calibration_samples,
            )
        else:
            success = optimizer.optimize_direct(
                output_path=output_path,
                precision=args.precision,
                workspace_mb=args.workspace,
                min_batch=args.min_batch,
                opt_batch=args.opt_batch,
                max_batch=args.max_batch,
                calibration_samples=args.calibration_samples,
                int8_cache_path=args.int8_cache,
            )

        if not success:
            print("\n[FAIL] Optimization failed.")
            sys.exit(1)

    # Run benchmark
    results = optimizer.benchmark_trt(
        trt_engine_path=args.trt_engine or (output_path if not args.benchmark_only else ''),
        num_warmup=args.benchmark_warmup,
        num_runs=args.benchmark_runs,
    )

    # Save results
    if args.save_results:
        output = {
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'precision': args.precision,
            'results': results,
        }
        with open(args.save_results, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.save_results}")
    else:
        default_results_path = os.path.join(
            os.path.dirname(model_path),
            f'trt_benchmark_{args.precision}.json'
        )
        output = {
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'precision': args.precision,
            'results': results,
        }
        with open(default_results_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {default_results_path}")


if __name__ == "__main__":
    main()
