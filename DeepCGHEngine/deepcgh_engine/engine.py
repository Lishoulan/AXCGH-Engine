"""
deepcgh_engine/engine.py — High-performance Python implementation of DeepCGHEngine.

This module mirrors the C++ architecture (Types, PreProcessor, InferenceCore,
EngineAPI) using Python + ONNX Runtime + NumPy/CuPy, providing a fully functional
engine that can be used immediately without C++ compilation.

Pipeline:
  RGB-D Input -> PreProcessor -> ONNX Inference -> IFFT PostProcess -> PhaseMap

Performance optimizations:
  - CuPy GPU acceleration for IFFT and preprocessing (when CUDA is available)
  - ONNX Runtime CUDA execution provider for neural network inference
  - Pre-allocated memory pools to avoid per-frame allocation
  - Vectorized NumPy operations for color-space conversion and volume assembly
"""

import os
import enum
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

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


# ===========================================================================
# Types (mirrors C++ Types.h)
# ===========================================================================

class PhaseFormat(enum.Enum):
    Uint8 = "uint8"
    Uint16 = "uint16"
    Float = "float"


class ColorSpace(enum.Enum):
    RGB = "rgb"
    YCbCr = "ycbcr"
    Gray = "gray"


class ExecutionProvider(enum.Enum):
    CPU = "cpu"
    CUDA = "cuda"
    DML = "dml"


class Status(enum.IntEnum):
    OK = 0
    NOT_INITIALIZED = 1
    MODEL_LOAD_FAILED = 2
    INVALID_INPUT = 3
    INFERENCE_FAILED = 4
    SIZE_MISMATCH = 5
    PROVIDER_UNAVAIL = 6
    ALLOCATION_FAILED = 7


@dataclass
class EngineConfig:
    """Engine configuration (mirrors C++ EngineConfig)."""
    height: int = 256
    width: int = 256
    num_planes: int = 5

    # Preprocessing
    color_space: ColorSpace = ColorSpace.YCbCr
    norm_mean: float = 0.0
    norm_std: float = 1.0

    # Inference
    provider: ExecutionProvider = ExecutionProvider.CPU
    device_id: int = 0
    enable_memory_pool: bool = True
    intra_op_threads: int = 4
    inter_op_threads: int = 1

    # Output
    phase_format: PhaseFormat = PhaseFormat.Uint8
    wavelength: float = 532e-6      # mm
    pixel_size: float = 8e-3        # mm
    focal_length: float = 200.0     # mm
    plane_distance: float = 10.0    # mm
    quantization_bits: int = 8
    int_factor: int = 2

    # GPU acceleration
    use_cupy: bool = True           # Auto-detect CuPy availability

    def validate(self):
        assert self.height > 0 and self.width > 0, "Dimensions must be positive"
        assert self.num_planes > 0, "num_planes must be positive"
        assert self.height % self.int_factor == 0, "height must be divisible by int_factor"
        assert self.width % self.int_factor == 0, "width must be divisible by int_factor"


# ===========================================================================
# PreProcessor (mirrors C++ PreProcessor)
# ===========================================================================

class PreProcessor:
    """RGB-D preprocessing: color-space conversion, depth normalization,
    multi-plane volume assembly. Supports CuPy GPU acceleration."""

    # ITU-R BT.601 coefficients (pre-computed for vectorized operations)
    _KR = 0.299
    _KG = 0.587
    _KB = 0.114

    def __init__(self, config: EngineConfig, use_gpu: bool = False):
        config.validate()
        self.config = config
        self.use_gpu = use_gpu and _CUPY_AVAILABLE
        self._xp = cp if self.use_gpu else np

        # Pre-allocate volume assembly weights
        C = config.num_planes
        center = C // 2
        self._color_weights = np.zeros(C, dtype=np.float32)
        self._depth_weights = np.zeros(C, dtype=np.float32)
        for p in range(C):
            if p == center:
                self._color_weights[p] = 1.0
                self._depth_weights[p] = 0.0
            else:
                offset = abs(p - center)
                weight = offset / (C // 2)
                self._color_weights[p] = 1.0 - weight
                self._depth_weights[p] = weight

        if self.use_gpu:
            self._color_weights = cp.asarray(self._color_weights)
            self._depth_weights = cp.asarray(self._depth_weights)

    def process(self, rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """
        Process RGB-D data into model input tensor.

        Args:
            rgb:   uint8 array [H, W, 3]
            depth: float32 array [H, W]

        Returns:
            float32 tensor [1, H, W, num_planes] in NHWC layout
        """
        H, W = self.config.height, self.config.width
        C = self.config.num_planes

        if rgb.shape != (H, W, 3):
            raise ValueError(f"RGB shape {rgb.shape} != expected ({H}, {W}, 3)")
        if depth.shape != (H, W):
            raise ValueError(f"Depth shape {depth.shape} != expected ({H}, {W})")

        xp = self._xp

        # Transfer to GPU if needed
        if self.use_gpu:
            rgb_gpu = cp.asarray(rgb)
            depth_gpu = cp.asarray(depth)
        else:
            rgb_gpu = rgb
            depth_gpu = depth

        # Step 1: Color-space conversion (vectorized)
        r = rgb_gpu[:, :, 0].astype(xp.float32)
        g = rgb_gpu[:, :, 1].astype(xp.float32)
        b = rgb_gpu[:, :, 2].astype(xp.float32)

        if self.config.color_space in (ColorSpace.YCbCr, ColorSpace.Gray):
            color = (self._KR * r + self._KG * g + self._KB * b) / 255.0
        else:  # RGB
            color = r / 255.0

        # Step 2: Depth normalization
        d_min, d_max = depth_gpu.min(), depth_gpu.max()
        rng = d_max - d_min
        if rng < 1e-6:
            depth_norm = xp.zeros_like(depth_gpu)
        else:
            depth_norm = (depth_gpu - d_min) / rng

        # Step 3: Vectorized volume assembly [H, W, C]
        # Use broadcasting: color [H,W,1] * weights [C] + depth [H,W,1] * weights [C]
        color_3d = color[:, :, xp.newaxis]       # [H, W, 1]
        depth_3d = depth_norm[:, :, xp.newaxis]   # [H, W, 1]
        volume = (color_3d * self._color_weights +
                  depth_3d * self._depth_weights)   # [H, W, C]

        # Step 4: Mean/std normalization
        if self.config.norm_mean != 0.0 or self.config.norm_std != 1.0:
            volume = (volume - self.config.norm_mean) / self.config.norm_std

        # Transfer back to CPU if needed (ORT runs on CPU/GPU depending on provider)
        if self.use_gpu:
            result = cp.asnumpy(volume[np.newaxis, ...].astype(np.float32))
        else:
            result = volume[np.newaxis, ...].astype(np.float32)

        return result  # [1, H, W, C]


# ===========================================================================
# InferenceCore (mirrors C++ InferenceCore)
# ===========================================================================

class InferenceCore:
    """ONNX Runtime inference with memory pooling and CUDA support."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_names: list = []
        self._io_binding = None

    def load_model(self, model_path: str) -> Status:
        """Load ONNX model and initialize session."""
        if not os.path.exists(model_path):
            return Status.MODEL_LOAD_FAILED

        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = self.config.intra_op_threads
            opts.inter_op_num_threads = self.config.inter_op_threads
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # Enable memory pattern optimization for fixed-size inputs
            opts.enable_mem_pattern = True
            opts.enable_mem_reuse = True

            # Select provider
            if self.config.provider == ExecutionProvider.CUDA:
                providers = [
                    ('CUDAExecutionProvider', {
                        'device_id': self.config.device_id,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 0,
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True,
                    }),
                    'CPUExecutionProvider'
                ]
            elif self.config.provider == ExecutionProvider.DML:
                providers = [
                    ('DmlExecutionProvider', {'device_id': self.config.device_id}),
                    'CPUExecutionProvider'
                ]
            else:
                providers = ['CPUExecutionProvider']

            self.session = ort.InferenceSession(model_path, opts, providers=providers)

            # Cache I/O names
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]

            # Create IO binding for zero-copy GPU inference
            if self.config.provider == ExecutionProvider.CUDA:
                try:
                    self._io_binding = self.session.io_binding()
                except Exception:
                    self._io_binding = None

            return Status.OK

        except Exception as e:
            print(f"[ERROR] Model load failed: {e}")
            return Status.MODEL_LOAD_FAILED

    def forward(self, input_tensor: np.ndarray) -> Tuple[Status, Optional[list]]:
        """Execute single forward pass."""
        if self.session is None:
            return Status.NOT_INITIALIZED, None

        try:
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            return Status.OK, outputs
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            return Status.INFERENCE_FAILED, None

    def is_loaded(self) -> bool:
        return self.session is not None


# ===========================================================================
# IFFT Post-Processor (replaces the TF Lambda in the ONNX model)
# ===========================================================================

class IFFTPostProcessor:
    """
    Replicates the _ifft_AmPh Lambda from DeepCGH:
      amp_0, phi_0 -> complex field -> IFFT2D -> angle -> quantize -> phi_slm

    Supports both NumPy (CPU) and CuPy (GPU) backends.
    """

    def __init__(self, quantization_bits: int = 8, use_gpu: bool = False):
        self.quantization = 2 ** quantization_bits
        self.use_gpu = use_gpu and _CUPY_AVAILABLE
        self._xp = cp if self.use_gpu else np

        # Pre-compute quantization constants
        self._q_scale = (self.quantization - 1) / (2 * np.pi)
        self._q_inv = 2 * np.pi / (self.quantization - 1)

        if self.use_gpu:
            self._q_scale = cp.float32(self._q_scale)
            self._q_inv = cp.float32(self._q_inv)

    def process(self, amp_0: np.ndarray, phi_0: np.ndarray) -> np.ndarray:
        """
        Args:
            amp_0: [1, H, W, 1] amplitude
            phi_0: [1, H, W, 1] initial phase

        Returns:
            phi_slm: [1, H, W, 1] quantized SLM phase in [-pi, pi]
        """
        xp = self._xp
        fft2 = cpfft.ifft2 if self.use_gpu else np.fft.ifft2
        ifftshift = cpfft.ifftshift if self.use_gpu else np.fft.ifftshift

        # Transfer to GPU if needed
        if self.use_gpu:
            amp = cp.asarray(np.squeeze(amp_0, axis=-1))    # [1, H, W]
            phi = cp.asarray(np.squeeze(phi_0, axis=-1))    # [1, H, W]
        else:
            amp = np.squeeze(amp_0, axis=-1)
            phi = np.squeeze(phi_0, axis=-1)

        # Construct complex field: amp * exp(j * phi)
        complex_field = amp * xp.exp(1j * phi)

        # IFFT shift + IFFT2D
        shifted = ifftshift(complex_field, axes=[1, 2])
        slm_field = fft2(shifted)

        # Extract phase angle
        modulation = xp.angle(slm_field)

        # Quantize: [-pi, pi] -> discrete levels -> [-pi, pi]
        PI = xp.float32(np.pi)
        quantized = xp.round((modulation + PI) * self._q_scale)
        quantized = quantized * self._q_inv - PI

        # Transfer back to CPU
        if self.use_gpu:
            result = cp.asnumpy(quantized[..., xp.newaxis])
        else:
            result = quantized[..., np.newaxis]

        return result  # [1, H, W, 1]


# ===========================================================================
# EngineAPI (mirrors C++ EngineAPI)
# ===========================================================================

class EngineAPI:
    """
    Top-level facade: RGB-D -> PreProcessor -> InferenceCore -> IFFT -> PhaseMap

    Usage:
        engine = EngineAPI()
        engine.init("model.onnx", EngineConfig(height=256, width=256, num_planes=5))
        phase = engine.generate_hologram(rgb, depth)
        engine.shutdown()
    """

    def __init__(self):
        self._preprocessor: Optional[PreProcessor] = None
        self._inference: Optional[InferenceCore] = None
        self._postprocessor: Optional[IFFTPostProcessor] = None
        self._config: Optional[EngineConfig] = None
        self._initialized = False
        self._last_error = ""
        self._use_gpu = False
        self._perf_stats: Dict[str, list] = {
            'preprocess': [], 'inference': [], 'postprocess': [], 'total': []
        }

    def init(self, model_path: str, config: Optional[EngineConfig] = None) -> Status:
        """Initialize the engine with model path and configuration."""
        if config is None:
            config = EngineConfig()

        try:
            config.validate()
        except AssertionError as e:
            self._last_error = str(e)
            return Status.INVALID_INPUT

        self._config = config

        # Determine GPU availability
        self._use_gpu = (config.use_cupy and _CUPY_AVAILABLE and
                         config.provider == ExecutionProvider.CUDA)

        self._preprocessor = PreProcessor(config, use_gpu=self._use_gpu)
        self._inference = InferenceCore(config)
        self._postprocessor = IFFTPostProcessor(
            config.quantization_bits, use_gpu=self._use_gpu)

        status = self._inference.load_model(model_path)
        if status != Status.OK:
            self._last_error = f"Model load failed: {model_path}"
            return status

        self._initialized = True
        self._last_error = ""
        self._perf_stats = {'preprocess': [], 'inference': [], 'postprocess': [], 'total': []}
        return Status.OK

    def is_ready(self) -> bool:
        return self._initialized and self._inference and self._inference.is_loaded()

    def shutdown(self):
        self._inference = None
        self._preprocessor = None
        self._postprocessor = None
        self._initialized = False

    def generate_hologram(self, rgb: np.ndarray, depth: np.ndarray,
                          benchmark: bool = False) -> Tuple[Status, Optional[np.ndarray]]:
        """
        Generate hologram phase map from RGB-D input.

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

            # Step 2: Inference (U-Net forward pass)
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
    def gpu_enabled(self) -> bool:
        return self._use_gpu
