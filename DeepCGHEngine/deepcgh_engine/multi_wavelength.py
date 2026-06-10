"""
deepcgh_engine/multi_wavelength.py — RGB three-wavelength hologram generation.

For color holography, separate phase maps are generated for three wavelengths
(Red: 633nm, Green: 532nm, Blue: 450nm) and then combined into a single SLM
phase pattern using time-division or spatial multiplexing.

Pipeline per channel:
  Single-channel RGB-D -> EngineAPI -> PhaseMap

Combination:
  PhaseR + PhaseG + PhaseB -> CombinedPhase (TimeDivision or SpatialMultiplex)
"""

import enum
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple

import numpy as np

from .engine import EngineAPI, EngineConfig, Status, PhaseFormat, ColorSpace, ExecutionProvider

# Try to import C++ engine
try:
    from ._deepcgh_engine import DeepCGHEngine as CppDeepCGHEngine
    _CPP_ENGINE_AVAILABLE = True
except ImportError:
    CppDeepCGHEngine = None
    _CPP_ENGINE_AVAILABLE = False


# ===========================================================================
# Types
# ===========================================================================

class CombineMode(enum.Enum):
    """Phase map combination mode for multi-wavelength holography."""
    TimeDivision = "time_division"
    SpatialMultiplex = "spatial_multiplex"


class SpatialPattern(enum.Enum):
    """Spatial multiplexing pattern type."""
    Checkerboard = "checkerboard"
    Stripe = "stripe"


# Standard RGB wavelengths in mm
WAVELENGTH_R = 633e-6   # 633 nm (HeNe red)
WAVELENGTH_G = 532e-6   # 532 nm (frequency-doubled Nd:YAG green)
WAVELENGTH_B = 450e-6   # 450 nm (blue laser diode)


@dataclass
class RGBEngineConfig:
    """Configuration for RGBHologramEngine."""
    # Base engine config (shared settings, wavelength will be overridden per channel)
    base_config: EngineConfig = field(default_factory=EngineConfig)

    # Wavelengths in mm
    wavelength_r: float = WAVELENGTH_R
    wavelength_g: float = WAVELENGTH_G
    wavelength_b: float = WAVELENGTH_B

    # Combination settings
    combine_mode: CombineMode = CombineMode.TimeDivision
    spatial_pattern: SpatialPattern = SpatialPattern.Checkerboard

    # Use C++ engine instead of Python engine
    use_cpp_engine: bool = False


# ===========================================================================
# Spatial Multiplexing Helpers
# ===========================================================================

def _checkerboard_mask(H: int, W: int) -> np.ndarray:
    """Generate a 3-channel checkerboard mask of shape [3, H, W].
    Each pixel is assigned to exactly one channel (R=0, G=1, B=2)."""
    mask = np.zeros((3, H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            ch = (i + j) % 3
            mask[ch, i, j] = 1.0
    return mask


def _stripe_mask(H: int, W: int) -> np.ndarray:
    """Generate a 3-channel stripe mask of shape [3, H, W].
    Rows are assigned to channels in R, G, B order."""
    mask = np.zeros((3, H, W), dtype=np.float32)
    for i in range(H):
        ch = i % 3
        mask[ch, i, :] = 1.0
    return mask


# ===========================================================================
# RGBHologramEngine
# ===========================================================================

class RGBHologramEngine:
    """
    RGB three-wavelength hologram generation engine.

    Wraps three EngineAPI (or CppDeepCGHEngine) instances, one per wavelength,
    and combines the resulting phase maps into a single SLM pattern.

    Usage:
        engine = RGBHologramEngine()
        engine.init("model.onnx", RGBEngineConfig())
        result = engine.generate_rgb_hologram(rgb, depth)
        engine.shutdown()
    """

    def __init__(self):
        self._engines: Dict[str, Any] = {}  # 'r', 'g', 'b'
        self._configs: Dict[str, EngineConfig] = {}
        self._config: Optional[RGBEngineConfig] = None
        self._initialized = False
        self._last_error = ""
        self._spatial_masks: Optional[np.ndarray] = None  # [3, H, W]
        self._perf_stats: Dict[str, list] = {
            'channel_r': [], 'channel_g': [], 'channel_b': [],
            'combine': [], 'total': []
        }

    def init(self, model_path: str, config: Optional[RGBEngineConfig] = None) -> Status:
        """Initialize three sub-engines with respective wavelengths.

        Args:
            model_path: Path to the ONNX model file (shared across all channels).
            config: RGBEngineConfig. If None, uses defaults.

        Returns:
            Status.OK on success.
        """
        if config is None:
            config = RGBEngineConfig()

        self._config = config
        base = config.base_config

        # Create per-channel configs with different wavelengths
        wavelengths = {
            'r': config.wavelength_r,
            'g': config.wavelength_g,
            'b': config.wavelength_b,
        }

        for channel, wl in wavelengths.items():
            ch_config = EngineConfig(
                height=base.height,
                width=base.width,
                num_planes=base.num_planes,
                color_space=base.color_space,
                norm_mean=base.norm_mean,
                norm_std=base.norm_std,
                provider=base.provider,
                device_id=base.device_id,
                enable_memory_pool=base.enable_memory_pool,
                intra_op_threads=base.intra_op_threads,
                inter_op_threads=base.inter_op_threads,
                phase_format=base.phase_format,
                wavelength=wl,
                pixel_size=base.pixel_size,
                focal_length=base.focal_length,
                plane_distance=base.plane_distance,
                quantization_bits=base.quantization_bits,
                int_factor=base.int_factor,
                use_cupy=base.use_cupy,
            )
            self._configs[channel] = ch_config

            # Create engine instance
            if config.use_cpp_engine:
                if not _CPP_ENGINE_AVAILABLE:
                    self._last_error = "C++ engine not available"
                    return Status.PROVIDER_UNAVAIL
                engine = CppDeepCGHEngine()
            else:
                engine = EngineAPI()

            status = engine.init(model_path, ch_config)
            if status != Status.OK:
                self._last_error = f"Failed to init {channel} channel engine: {getattr(engine, 'last_error', '')}"
                return status

            self._engines[channel] = engine

        # Pre-compute spatial multiplexing masks
        H, W = base.height, base.width
        if config.combine_mode == CombineMode.SpatialMultiplex:
            if config.spatial_pattern == SpatialPattern.Checkerboard:
                self._spatial_masks = _checkerboard_mask(H, W)
            else:
                self._spatial_masks = _stripe_mask(H, W)

        self._initialized = True
        self._last_error = ""
        self._perf_stats = {
            'channel_r': [], 'channel_g': [], 'channel_b': [],
            'combine': [], 'total': []
        }
        return Status.OK

    def is_ready(self) -> bool:
        return self._initialized and all(
            e.is_ready() if hasattr(e, 'is_ready') else True
            for e in self._engines.values()
        )

    def shutdown(self):
        """Release all sub-engine resources."""
        for engine in self._engines.values():
            engine.shutdown()
        self._engines = {}
        self._configs = {}
        self._initialized = False
        self._spatial_masks = None

    def generate_rgb_hologram(self, rgb: np.ndarray, depth: np.ndarray,
                              benchmark: bool = False) -> Tuple[Status, Optional[Dict[str, np.ndarray]]]:
        """Generate RGB hologram from RGB-D input.

        Args:
            rgb:   uint8 array [H, W, 3]
            depth: float32 array [H, W]
            benchmark: If True, record per-stage timing.

        Returns:
            (Status, dict) with keys:
              'phase_r': float32 [H, W] — Red channel phase map
              'phase_g': float32 [H, W] — Green channel phase map
              'phase_b': float32 [H, W] — Blue channel phase map
              'phase_combined': float32 [H, W] — Combined phase map
            or (Status, None) on failure.
        """
        if not self.is_ready():
            self._last_error = "Engine not initialized"
            return Status.NOT_INITIALIZED, None

        H, W = self._config.base_config.height, self._config.base_config.width

        if rgb.shape != (H, W, 3):
            self._last_error = f"RGB shape {rgb.shape} != expected ({H}, {W}, 3)"
            return Status.INVALID_INPUT, None

        t_total_start = time.perf_counter()

        # Split RGB into single-channel inputs
        r_channel = rgb[:, :, 0:1]  # [H, W, 1]
        g_channel = rgb[:, :, 1:2]
        b_channel = rgb[:, :, 2:3]

        # For each channel, create a grayscale RGB-D input and generate phase
        phases = {}
        channel_data = {
            'r': r_channel,
            'g': g_channel,
            'b': b_channel,
        }

        for ch, ch_img in channel_data.items():
            t0 = time.perf_counter()

            # Construct single-channel RGB input: replicate the channel to 3 channels
            # so the preprocessor can handle it normally
            ch_rgb = np.repeat(ch_img, 3, axis=2)  # [H, W, 3]

            engine = self._engines[ch]
            status, phase = engine.generate_hologram(ch_rgb, depth)

            if status != Status.OK:
                self._last_error = f"{ch} channel generation failed: {getattr(engine, 'last_error', '')}"
                return status, None

            phases[ch] = phase

            t1 = time.perf_counter()
            if benchmark:
                self._perf_stats[f'channel_{ch}'].append((t1 - t0) * 1000)

        # Combine phase maps
        t_combine_start = time.perf_counter()
        phase_combined = self._combine_phases(
            phases['r'], phases['g'], phases['b'])
        t_combine_end = time.perf_counter()

        if benchmark:
            self._perf_stats['combine'].append(
                (t_combine_end - t_combine_start) * 1000)
            self._perf_stats['total'].append(
                (t_combine_end - t_total_start) * 1000)

        return Status.OK, {
            'phase_r': phases['r'],
            'phase_g': phases['g'],
            'phase_b': phases['b'],
            'phase_combined': phase_combined,
        }

    def _combine_phases(self, phase_r: np.ndarray, phase_g: np.ndarray,
                        phase_b: np.ndarray) -> np.ndarray:
        """Combine three channel phase maps according to the configured mode.

        Args:
            phase_r, phase_g, phase_b: float32 [H, W] phase maps in [-pi, pi].

        Returns:
            Combined float32 [H, W] phase map.
        """
        if self._config.combine_mode == CombineMode.TimeDivision:
            # Simple average — for display, the three maps are shown sequentially
            # at high speed; the combined map is the average for preview purposes.
            return (phase_r + phase_g + phase_b) / 3.0

        elif self._config.combine_mode == CombineMode.SpatialMultiplex:
            # Spatial interleaving using pre-computed masks
            masks = self._spatial_masks  # [3, H, W]
            stacked = np.stack([phase_r, phase_g, phase_b], axis=0)  # [3, H, W]
            combined = np.sum(stacked * masks, axis=0)  # [H, W]
            return combined

        else:
            raise ValueError(f"Unknown combine mode: {self._config.combine_mode}")

    def generate_rgb_hologram_quantized(self, rgb: np.ndarray, depth: np.ndarray,
                                        benchmark: bool = False) -> Tuple[Status, Optional[Dict[str, np.ndarray]]]:
        """Generate quantized RGB hologram phase maps for SLM display.

        Returns:
            (Status, dict) with keys 'phase_r', 'phase_g', 'phase_b',
            'phase_combined' — all quantized according to base_config.phase_format.
        """
        status, result = self.generate_rgb_hologram(rgb, depth, benchmark=benchmark)
        if status != Status.OK or result is None:
            return status, None

        phase_format = self._config.base_config.phase_format

        def quantize(phase: np.ndarray) -> np.ndarray:
            if phase_format == PhaseFormat.Uint8:
                normalized = (phase + np.pi) / (2 * np.pi)
                return np.clip(normalized * 255, 0, 255).astype(np.uint8)
            elif phase_format == PhaseFormat.Uint16:
                normalized = (phase + np.pi) / (2 * np.pi)
                return np.clip(normalized * 65535, 0, 65535).astype(np.uint16)
            else:
                return phase

        return Status.OK, {
            'phase_r': quantize(result['phase_r']),
            'phase_g': quantize(result['phase_g']),
            'phase_b': quantize(result['phase_b']),
            'phase_combined': quantize(result['phase_combined']),
        }

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
    def config(self) -> Optional[RGBEngineConfig]:
        return self._config

    @property
    def engines(self) -> Dict[str, Any]:
        """Access the per-channel engine instances."""
        return self._engines
