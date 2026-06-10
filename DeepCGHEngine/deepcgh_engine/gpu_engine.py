"""
deepcgh_engine/gpu_engine.py — GPU-accelerated DeepCGH Engine.

Extends the CPU-based EngineAPI with CUDA, TensorRT, and DML support:
  - Auto-detect available ORT providers (TensorRT > CUDA > DML > CPU)
  - FP16 inference on GPU (convert input to float16, convert output back to float32)
  - Batch inference (process multiple frames at once)
  - GPU memory management (pre-allocate IO bindings on GPU)
  - Benchmark mode comparing CPU vs GPU vs TensorRT
  - CuPy-based IFFT for GPU-accelerated post-processing (if cupy available)
  - Automatic fallback to CPU if GPU not available

Pipeline:
  RGB-D Input -> PreProcessor -> ONNX/TRT Inference -> IFFT PostProcess -> PhaseMap
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Any

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("onnxruntime is required: pip install onnxruntime")

# Optional CuPy for GPU acceleration
try:
    import cupy as cp
    import cupy.fft as cpfft
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False

from .engine import (
    EngineAPI,
    EngineConfig,
    PreProcessor,
    IFFTPostProcessor,
    PhaseFormat,
    ColorSpace,
    ExecutionProvider,
    Status,
)


# ===========================================================================
# GPU Configuration
# ===========================================================================

@dataclass
class GPUConfig:
    """GPU-specific configuration for GPUEngineAPI."""
    # Provider priority: which GPU provider to prefer
    # Options: 'tensorrt', 'cuda', 'dml', 'cpu'
    provider_priority: List[str] = field(default_factory=lambda: ['tensorrt', 'cuda', 'dml', 'cpu'])

    # FP16 inference
    enable_fp16: bool = True

    # Batch inference
    max_batch_size: int = 1

    # GPU memory management
    preallocate_io_bindings: bool = True
    gpu_mem_limit_mb: int = 0          # 0 = no limit

    # TensorRT-specific
    trt_max_workspace_size_mb: int = 1024
    trt_cache_path: str = ""           # TRT engine cache directory
    trt_fp16: bool = True
    trt_int8: bool = False

    # Device
    device_id: int = 0

    # CuPy IFFT acceleration
    use_cupy_ifft: bool = True


# ===========================================================================
# Provider Detection
# ===========================================================================

def detect_available_providers() -> List[str]:
    """Detect available ONNX Runtime execution providers.

    Returns list of provider names in priority order:
      TensorRT > CUDA > DML > CPU
    """
    available = []
    try:
        registered = ort.get_available_providers()
    except Exception:
        registered = ['CPUExecutionProvider']

    if 'TensorrtExecutionProvider' in registered:
        available.append('tensorrt')
    if 'CUDAExecutionProvider' in registered:
        available.append('cuda')
    if 'DmlExecutionProvider' in registered:
        available.append('dml')
    available.append('cpu')  # CPU is always available

    return available


def select_provider(gpu_config: GPUConfig) -> Tuple[str, str]:
    """Select the best available provider based on priority.

    Returns:
        (provider_key, ort_provider_name) e.g. ('cuda', 'CUDAExecutionProvider')
    """
    available = detect_available_providers()

    for preferred in gpu_config.provider_priority:
        if preferred in available:
            mapping = {
                'tensorrt': ('tensorrt', 'TensorrtExecutionProvider'),
                'cuda': ('cuda', 'CUDAExecutionProvider'),
                'dml': ('dml', 'DmlExecutionProvider'),
                'cpu': ('cpu', 'CPUExecutionProvider'),
            }
            return mapping[preferred]

    return ('cpu', 'CPUExecutionProvider')


# ===========================================================================
# GPU InferenceCore
# ===========================================================================

class GPUInferenceCore:
    """ONNX Runtime inference with GPU support, FP16, batch, and IO binding."""

    def __init__(self, config: EngineConfig, gpu_config: GPUConfig):
        self.config = config
        self.gpu_config = gpu_config
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_names: List[str] = []
        self._io_binding = None
        self._provider_key: str = 'cpu'
        self._ort_provider: str = 'CPUExecutionProvider'
        self._input_shape: Optional[List[int]] = None
        self._output_shapes: Optional[List[List[int]]] = None
        self._gpu_input_buffer: Optional[np.ndarray] = None
        self._gpu_output_buffers: Optional[List[np.ndarray]] = None

    def load_model(self, model_path: str) -> Status:
        """Load ONNX model with GPU provider selection."""
        if not os.path.exists(model_path):
            return Status.MODEL_LOAD_FAILED

        try:
            # Select best provider
            self._provider_key, self._ort_provider = select_provider(self.gpu_config)

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = self.config.intra_op_threads
            opts.inter_op_num_threads = self.config.inter_op_threads
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.enable_mem_pattern = True
            opts.enable_mem_reuse = True

            providers = self._build_provider_list()

            self.session = ort.InferenceSession(model_path, opts, providers=providers)

            # Cache I/O metadata
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]

            input_meta = self.session.get_inputs()[0]
            self._input_shape = [
                self.gpu_config.max_batch_size if isinstance(d, str) or d == 1 else d
                for d in input_meta.shape
            ]

            self._output_shapes = []
            for o in self.session.get_outputs():
                shape = [
                    self.gpu_config.max_batch_size if isinstance(d, str) or d == 1 else d
                    for d in o.shape
                ]
                self._output_shapes.append(shape)

            # Create IO binding for zero-copy GPU inference
            if self._provider_key in ('cuda', 'tensorrt') and self.gpu_config.preallocate_io_bindings:
                try:
                    self._io_binding = self.session.io_binding()
                    self._preallocate_gpu_buffers()
                except Exception:
                    self._io_binding = None

            return Status.OK

        except Exception as e:
            print(f"[ERROR] GPU model load failed: {e}")
            # Fallback to CPU
            try:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = self.config.intra_op_threads
                opts.inter_op_num_threads = self.config.inter_op_threads
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.session = ort.InferenceSession(model_path, opts, providers=['CPUExecutionProvider'])
                self.input_name = self.session.get_inputs()[0].name
                self.output_names = [o.name for o in self.session.get_outputs()]
                self._provider_key = 'cpu'
                self._ort_provider = 'CPUExecutionProvider'
                self._io_binding = None
                return Status.OK
            except Exception as e2:
                print(f"[ERROR] CPU fallback also failed: {e2}")
                return Status.MODEL_LOAD_FAILED

    def _build_provider_list(self) -> List[Any]:
        """Build the ORT provider list with options."""
        providers = []
        device_id = self.gpu_config.device_id

        if self._ort_provider == 'TensorrtExecutionProvider':
            trt_options = {
                'device_id': device_id,
                'trt_max_workspace_size': self.gpu_config.trt_max_workspace_size_mb * 1024 * 1024,
                'trt_fp16_enable': self.gpu_config.trt_fp16,
                'trt_int8_enable': self.gpu_config.trt_int8,
            }
            if self.gpu_config.trt_cache_path:
                trt_options['trt_engine_cache_enable'] = True
                trt_options['trt_engine_cache_path'] = self.gpu_config.trt_cache_path
            providers.append(('TensorrtExecutionProvider', trt_options))
            # CUDA as fallback within TRT provider
            providers.append(('CUDAExecutionProvider', {
                'device_id': device_id,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': self.gpu_config.gpu_mem_limit_mb * 1024 * 1024 if self.gpu_config.gpu_mem_limit_mb > 0 else 0,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }))

        elif self._ort_provider == 'CUDAExecutionProvider':
            providers.append(('CUDAExecutionProvider', {
                'device_id': device_id,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': self.gpu_config.gpu_mem_limit_mb * 1024 * 1024 if self.gpu_config.gpu_mem_limit_mb > 0 else 0,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }))

        elif self._ort_provider == 'DmlExecutionProvider':
            providers.append(('DmlExecutionProvider', {
                'device_id': device_id,
            }))

        # CPU as final fallback
        providers.append('CPUExecutionProvider')
        return providers

    def _preallocate_gpu_buffers(self):
        """Pre-allocate GPU IO buffers for zero-copy inference."""
        if self._input_shape is None:
            return
        try:
            dtype = np.float16 if self.gpu_config.enable_fp16 else np.float32
            self._gpu_input_buffer = np.empty(self._input_shape, dtype=dtype)
            self._gpu_output_buffers = [
                np.empty(shape, dtype=np.float32) for shape in self._output_shapes
            ]
        except Exception as e:
            print(f"[WARN] GPU buffer pre-allocation failed: {e}")
            self._gpu_input_buffer = None
            self._gpu_output_buffers = None

    def forward(self, input_tensor: np.ndarray) -> Tuple[Status, Optional[List[np.ndarray]]]:
        """Execute forward pass with GPU acceleration."""
        if self.session is None:
            return Status.NOT_INITIALIZED, None

        try:
            # FP16 conversion if enabled and on GPU
            if self.gpu_config.enable_fp16 and self._provider_key in ('cuda', 'tensorrt'):
                input_tensor = input_tensor.astype(np.float16)

            # Use IO binding for zero-copy GPU inference
            if self._io_binding is not None and self._provider_key in ('cuda', 'tensorrt'):
                return self._forward_io_binding(input_tensor)

            # Standard inference path
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

            # Convert FP16 outputs back to FP32
            if self.gpu_config.enable_fp16 and self._provider_key in ('cuda', 'tensorrt'):
                outputs = [o.astype(np.float32) if o.dtype == np.float16 else o for o in outputs]

            return Status.OK, outputs

        except Exception as e:
            print(f"[ERROR] GPU inference failed: {e}")
            return Status.INFERENCE_FAILED, None

    def _forward_io_binding(self, input_tensor: np.ndarray) -> Tuple[Status, Optional[List[np.ndarray]]]:
        """Forward pass using IO binding for zero-copy GPU inference."""
        try:
            self._io_binding.clear_binding_inputs()
            self._io_binding.clear_binding_outputs()

            # Bind input
            x = ort.OrtValue.ortvalue_from_numpy(input_tensor)
            self._io_binding.bind_ortvalue_input(self.input_name, x)

            # Bind outputs
            for name in self.output_names:
                self._io_binding.bind_output(name)

            self.session.run_with_iobinding(self._io_binding)

            outputs = self._io_binding.copy_outputs_to_cpu()

            # Convert FP16 outputs back to FP32
            if self.gpu_config.enable_fp16 and self._provider_key in ('cuda', 'tensorrt'):
                outputs = [o.astype(np.float32) if o.dtype == np.float16 else o for o in outputs]

            return Status.OK, outputs

        except Exception as e:
            # Fallback to standard run
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            if self.gpu_config.enable_fp16 and self._provider_key in ('cuda', 'tensorrt'):
                outputs = [o.astype(np.float32) if o.dtype == np.float16 else o for o in outputs]
            return Status.OK, outputs

    def forward_batch(self, input_tensors: List[np.ndarray]) -> Tuple[Status, Optional[List[np.ndarray]]]:
        """Execute batched forward pass.

        Args:
            input_tensors: List of input arrays, each [1, H, W, C]

        Returns:
            (Status, list of output arrays) or (Status, None)
        """
        if self.session is None:
            return Status.NOT_INITIALIZED, None

        if not input_tensors:
            return Status.INVALID_INPUT, None

        try:
            # Stack inputs into a batch
            batch = np.concatenate(input_tensors, axis=0)  # [B, H, W, C]

            # FP16 conversion
            if self.gpu_config.enable_fp16 and self._provider_key in ('cuda', 'tensorrt'):
                batch = batch.astype(np.float16)

            outputs = self.session.run(self.output_names, {self.input_name: batch})

            # Convert FP16 outputs back to FP32
            if self.gpu_config.enable_fp16 and self._provider_key in ('cuda', 'tensorrt'):
                outputs = [o.astype(np.float32) if o.dtype == np.float16 else o for o in outputs]

            return Status.OK, outputs

        except Exception as e:
            print(f"[ERROR] Batch inference failed: {e}")
            return Status.INFERENCE_FAILED, None

    def is_loaded(self) -> bool:
        return self.session is not None

    @property
    def provider_key(self) -> str:
        return self._provider_key

    @property
    def is_gpu(self) -> bool:
        return self._provider_key in ('cuda', 'tensorrt', 'dml')


# ===========================================================================
# CuPy IFFT Post-Processor
# ===========================================================================

class CuPyIFFTPostProcessor:
    """CuPy-accelerated IFFT post-processing.

    Replicates the _ifft_AmPh Lambda from DeepCGH with GPU acceleration.
    Falls back to NumPy if CuPy is not available.
    """

    def __init__(self, quantization_bits: int = 8, use_gpu: bool = False):
        self.quantization = 2 ** quantization_bits
        self.use_gpu = use_gpu and _CUPY_AVAILABLE
        self._xp = cp if self.use_gpu else np

        self._q_scale = (self.quantization - 1) / (2 * np.pi)
        self._q_inv = 2 * np.pi / (self.quantization - 1)

        if self.use_gpu:
            self._q_scale = cp.float32(self._q_scale)
            self._q_inv = cp.float32(self._q_inv)

    def process(self, amp_0: np.ndarray, phi_0: np.ndarray) -> np.ndarray:
        """Process amplitude and phase through IFFT.

        Args:
            amp_0: [B, H, W, 1] or [1, H, W, 1] amplitude
            phi_0: [B, H, W, 1] or [1, H, W, 1] initial phase

        Returns:
            phi_slm: same shape as input, quantized SLM phase in [-pi, pi]
        """
        xp = self._xp
        fft2 = cpfft.ifft2 if self.use_gpu else np.fft.ifft2
        ifftshift = cpfft.ifftshift if self.use_gpu else np.fft.ifftshift

        if self.use_gpu:
            amp = cp.asarray(np.squeeze(amp_0, axis=-1))
            phi = cp.asarray(np.squeeze(phi_0, axis=-1))
        else:
            amp = np.squeeze(amp_0, axis=-1)
            phi = np.squeeze(phi_0, axis=-1)

        complex_field = amp * xp.exp(1j * phi)

        # Determine spatial axes based on input dimensions
        ndim = complex_field.ndim
        if ndim == 3:
            axes = [1, 2]
        elif ndim == 2:
            axes = [0, 1]
        else:
            axes = [ndim - 2, ndim - 1]

        shifted = ifftshift(complex_field, axes=axes)
        slm_field = fft2(shifted, axes=axes)

        modulation = xp.angle(slm_field)

        PI = xp.float32(np.pi)
        quantized = xp.round((modulation + PI) * self._q_scale)
        quantized = quantized * self._q_inv - PI

        if self.use_gpu:
            result = cp.asnumpy(quantized[..., xp.newaxis])
        else:
            result = quantized[..., np.newaxis]

        return result

    def process_batch(self, amp_batch: np.ndarray, phi_batch: np.ndarray) -> np.ndarray:
        """Process a batch of amplitude/phase pairs through IFFT.

        Args:
            amp_batch: [B, H, W, 1] amplitude
            phi_batch: [B, H, W, 1] initial phase

        Returns:
            phi_slm_batch: [B, H, W, 1] quantized SLM phase
        """
        return self.process(amp_batch, phi_batch)


# ===========================================================================
# GPUEngineAPI
# ===========================================================================

class GPUEngineAPI:
    """
    GPU-accelerated hologram generation engine.

    Extends EngineAPI with:
      - Auto-detect GPU providers (TensorRT > CUDA > DML > CPU)
      - FP16 inference on GPU
      - Batch inference
      - GPU memory management with IO binding
      - CuPy IFFT acceleration
      - Benchmark mode (CPU vs GPU vs TensorRT)

    Usage:
        engine = GPUEngineAPI()
        engine.init("model.onnx", EngineConfig(height=256, width=256))
        phase = engine.generate_hologram(rgb, depth)
        engine.shutdown()
    """

    def __init__(self, gpu_config: Optional[GPUConfig] = None):
        self._gpu_config = gpu_config or GPUConfig()
        self._preprocessor: Optional[PreProcessor] = None
        self._inference: Optional[GPUInferenceCore] = None
        self._postprocessor: Optional[CuPyIFFTPostProcessor] = None
        self._config: Optional[EngineConfig] = None
        self._initialized = False
        self._last_error = ""
        self._use_gpu = False
        self._use_cupy_ifft = False
        self._perf_stats: Dict[str, list] = {
            'preprocess': [], 'inference': [], 'postprocess': [], 'total': [],
            'transfer_in': [], 'transfer_out': [],
        }

    def init(self, model_path: str, config: Optional[EngineConfig] = None) -> Status:
        """Initialize the GPU engine with model path and configuration."""
        if config is None:
            config = EngineConfig()

        try:
            config.validate()
        except AssertionError as e:
            self._last_error = str(e)
            return Status.INVALID_INPUT

        self._config = config

        # Determine GPU usage
        available = detect_available_providers()
        self._use_gpu = any(p in available for p in ['cuda', 'tensorrt', 'dml'])

        # Determine CuPy IFFT availability
        self._use_cupy_ifft = (
            self._gpu_config.use_cupy_ifft
            and _CUPY_AVAILABLE
            and self._use_gpu
        )

        # Update config provider to match detected GPU
        if self._use_gpu:
            _, ort_provider = select_provider(self._gpu_config)
            provider_map = {
                'CUDAExecutionProvider': ExecutionProvider.CUDA,
                'DmlExecutionProvider': ExecutionProvider.DML,
            }
            config.provider = provider_map.get(ort_provider, ExecutionProvider.CPU)

        # Initialize components
        self._preprocessor = PreProcessor(config, use_gpu=self._use_cupy_ifft)
        self._inference = GPUInferenceCore(config, self._gpu_config)
        self._postprocessor = CuPyIFFTPostProcessor(
            config.quantization_bits, use_gpu=self._use_cupy_ifft
        )

        status = self._inference.load_model(model_path)
        if status != Status.OK:
            self._last_error = f"Model load failed: {model_path}"
            return status

        self._initialized = True
        self._last_error = ""
        self._perf_stats = {
            'preprocess': [], 'inference': [], 'postprocess': [], 'total': [],
            'transfer_in': [], 'transfer_out': [],
        }

        # Print provider info
        provider_name = self._inference.provider_key.upper()
        fp16_str = "FP16" if self.gpu_uses_fp16 else "FP32"
        cupy_str = "CuPy" if self._use_cupy_ifft else "NumPy"
        print(f"[GPUEngine] Provider: {provider_name} | Precision: {fp16_str} | IFFT: {cupy_str}")

        return Status.OK

    def is_ready(self) -> bool:
        return self._initialized and self._inference and self._inference.is_loaded()

    def shutdown(self):
        """Release all resources."""
        self._inference = None
        self._preprocessor = None
        self._postprocessor = None
        self._initialized = False

    def generate_hologram(self, rgb: np.ndarray, depth: np.ndarray,
                          benchmark: bool = False) -> Tuple[Status, Optional[np.ndarray]]:
        """Generate hologram phase map from RGB-D input.

        Args:
            rgb:   uint8 [H, W, 3]
            depth: float32 [H, W]
            benchmark: If True, record per-stage timing.

        Returns:
            (Status, float32 [H, W] phase in [-pi, pi]) or (Status, None)
        """
        if not self.is_ready():
            self._last_error = "Engine not initialized"
            return Status.NOT_INITIALIZED, None

        try:
            t0 = time.perf_counter()

            # Step 1: Preprocess
            input_tensor = self._preprocessor.process(rgb, depth)
            t1 = time.perf_counter()

            # Step 2: Inference
            status, outputs = self._inference.forward(input_tensor)
            if status != Status.OK:
                self._last_error = "Inference failed"
                return status, None

            amp_0, phi_0 = outputs[0], outputs[1]
            t2 = time.perf_counter()

            # Step 3: IFFT post-processing
            phi_slm = self._postprocessor.process(amp_0, phi_0)
            t3 = time.perf_counter()

            # Step 4: Extract spatial phase [H, W]
            phase = np.squeeze(phi_slm, axis=(0, 3))  # [H, W]

            if benchmark:
                self._perf_stats['preprocess'].append((t1 - t0) * 1000)
                self._perf_stats['inference'].append((t2 - t1) * 1000)
                self._perf_stats['postprocess'].append((t3 - t2) * 1000)
                self._perf_stats['total'].append((t3 - t0) * 1000)

            return Status.OK, phase

        except Exception as e:
            self._last_error = str(e)
            return Status.INFERENCE_FAILED, None

    def generate_hologram_batch(self, rgb_list: List[np.ndarray],
                                depth_list: List[np.ndarray],
                                benchmark: bool = False) -> Tuple[Status, Optional[List[np.ndarray]]]:
        """Generate holograms for a batch of RGB-D inputs.

        Args:
            rgb_list:   List of uint8 [H, W, 3] arrays
            depth_list: List of float32 [H, W] arrays
            benchmark:  If True, record timing.

        Returns:
            (Status, list of float32 [H, W] phase arrays) or (Status, None)
        """
        if not self.is_ready():
            self._last_error = "Engine not initialized"
            return Status.NOT_INITIALIZED, None

        if len(rgb_list) != len(depth_list):
            self._last_error = "rgb_list and depth_list must have same length"
            return Status.INVALID_INPUT, None

        batch_size = len(rgb_list)
        if batch_size == 0:
            return Status.INVALID_INPUT, None

        try:
            t0 = time.perf_counter()

            # Step 1: Preprocess all inputs
            input_tensors = []
            for rgb, depth in zip(rgb_list, depth_list):
                tensor = self._preprocessor.process(rgb, depth)
                input_tensors.append(tensor)
            t1 = time.perf_counter()

            # Step 2: Batch inference
            if batch_size == 1:
                status, outputs = self._inference.forward(input_tensors[0])
            else:
                status, outputs = self._inference.forward_batch(input_tensors)

            if status != Status.OK:
                self._last_error = "Batch inference failed"
                return status, None

            amp_0, phi_0 = outputs[0], outputs[1]
            t2 = time.perf_counter()

            # Step 3: Batch IFFT post-processing
            phi_slm = self._postprocessor.process_batch(amp_0, phi_0)
            t3 = time.perf_counter()

            # Step 4: Split batch results
            results = []
            for i in range(batch_size):
                if phi_slm.ndim == 4:
                    phase = phi_slm[i, :, :, 0]  # [H, W]
                else:
                    phase = np.squeeze(phi_slm, axis=-1)
                results.append(phase)

            if benchmark:
                self._perf_stats['preprocess'].append((t1 - t0) * 1000)
                self._perf_stats['inference'].append((t2 - t1) * 1000)
                self._perf_stats['postprocess'].append((t3 - t2) * 1000)
                self._perf_stats['total'].append((t3 - t0) * 1000)

            return Status.OK, results

        except Exception as e:
            self._last_error = str(e)
            return Status.INFERENCE_FAILED, None

    def generate_hologram_quantized(self, rgb: np.ndarray, depth: np.ndarray,
                                    benchmark: bool = False) -> Tuple[Status, Optional[np.ndarray]]:
        """Generate quantized phase map for SLM display."""
        status, phase = self.generate_hologram(rgb, depth, benchmark=benchmark)
        if status != Status.OK:
            return status, None

        if self._config.phase_format == PhaseFormat.Uint8:
            normalized = (phase + np.pi) / (2 * np.pi)
            quantized = np.clip(normalized * 255, 0, 255).astype(np.uint8)
        elif self._config.phase_format == PhaseFormat.Uint16:
            normalized = (phase + np.pi) / (2 * np.pi)
            quantized = np.clip(normalized * 65535, 0, 65535).astype(np.uint16)
        else:
            quantized = phase

        return Status.OK, quantized

    # -----------------------------------------------------------------------
    # Benchmark
    # -----------------------------------------------------------------------

    def benchmark(self, model_path: str, config: Optional[EngineConfig] = None,
                  num_warmup: int = 5, num_runs: int = 50) -> Dict[str, Any]:
        """Benchmark CPU vs GPU vs TensorRT performance.

        Args:
            model_path: Path to ONNX model.
            config: Engine configuration.
            num_warmup: Number of warmup iterations.
            num_runs: Number of benchmark iterations.

        Returns:
            Dict with benchmark results for each provider.
        """
        if config is None:
            config = EngineConfig()

        results = {}
        available = detect_available_providers()

        H, W = config.height, config.width
        rgb = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        depth = np.random.rand(H, W).astype(np.float32)

        # Test each available provider
        provider_configs = {
            'cpu': (['cpu'], False),
            'cuda': (['cuda', 'cpu'], False),
            'tensorrt': (['tensorrt', 'cuda', 'cpu'], False),
        }

        for provider_name, (priority, fp16) in provider_configs.items():
            if provider_name not in available:
                results[provider_name] = {'status': 'unavailable'}
                continue

            # Test FP32
            gpu_cfg = GPUConfig(
                provider_priority=priority,
                enable_fp16=False,
                preallocate_io_bindings=True,
            )
            result = self._benchmark_provider(model_path, config, gpu_cfg, rgb, depth,
                                              num_warmup, num_runs)
            results[f"{provider_name}_fp32"] = result

            # Test FP16 (GPU only)
            if provider_name in ('cuda', 'tensorrt'):
                gpu_cfg_fp16 = GPUConfig(
                    provider_priority=priority,
                    enable_fp16=True,
                    preallocate_io_bindings=True,
                )
                result_fp16 = self._benchmark_provider(model_path, config, gpu_cfg_fp16,
                                                       rgb, depth, num_warmup, num_runs)
                results[f"{provider_name}_fp16"] = result_fp16

        return results

    def _benchmark_provider(self, model_path: str, config: EngineConfig,
                            gpu_config: GPUConfig, rgb: np.ndarray, depth: np.ndarray,
                            num_warmup: int, num_runs: int) -> Dict[str, Any]:
        """Run benchmark with a specific provider configuration."""
        engine = GPUEngineAPI(gpu_config)
        status = engine.init(model_path, EngineConfig(
            height=config.height, width=config.width,
            num_planes=config.num_planes,
        ))

        if status != Status.OK:
            return {'status': 'init_failed', 'error': engine.last_error}

        # Warmup
        for _ in range(num_warmup):
            engine.generate_hologram(rgb, depth)

        # Benchmark
        for _ in range(num_runs):
            engine.generate_hologram(rgb, depth, benchmark=True)

        stats = engine.get_perf_stats()
        engine.shutdown()

        return {
            'status': 'ok',
            'provider': engine._inference.provider_key if engine._inference else 'unknown',
            'stats': stats,
        }

    # -----------------------------------------------------------------------
    # Properties & Utilities
    # -----------------------------------------------------------------------

    def get_perf_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics from benchmarked runs."""
        stats = {}
        for key, values in self._perf_stats.items():
            if values:
                arr = np.array(values)
                stats[key] = {
                    'mean_ms': float(np.mean(arr)),
                    'std_ms': float(np.std(arr)),
                    'min_ms': float(np.min(arr)),
                    'max_ms': float(np.max(arr)),
                    'median_ms': float(np.median(arr)),
                    'n_runs': len(values),
                }
        return stats

    @property
    def last_error(self) -> str:
        return self._last_error

    @property
    def config(self) -> Optional[EngineConfig]:
        return self._config

    @property
    def gpu_config(self) -> GPUConfig:
        return self._gpu_config

    @property
    def gpu_enabled(self) -> bool:
        return self._use_gpu

    @property
    def gpu_uses_fp16(self) -> bool:
        return self._gpu_config.enable_fp16 and self._use_gpu

    @property
    def active_provider(self) -> str:
        if self._inference:
            return self._inference.provider_key
        return 'none'

    @property
    def cupy_ifft_enabled(self) -> bool:
        return self._use_cupy_ifft

    @staticmethod
    def list_available_providers() -> List[str]:
        """List available ORT execution providers."""
        return detect_available_providers()

    @staticmethod
    def is_cupy_available() -> bool:
        """Check if CuPy is available for GPU IFFT."""
        return _CUPY_AVAILABLE
