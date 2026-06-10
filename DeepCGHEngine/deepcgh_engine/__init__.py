"""
DeepCGHEngine — Deep-learning Computer-Generated Holography Engine

A lightweight holographic rendering engine that accepts RGB-D data streams,
performs neural network inference, and outputs phase maps for SLM display.

Usage (Python engine — no C++ compilation required):
    from deepcgh_engine import EngineAPI, EngineConfig, ColorSpace

    engine = EngineAPI()
    engine.init(
        model_path="models/deepcgh_unet.onnx",
        config=EngineConfig(height=256, width=256, num_planes=5),
    )

    import numpy as np
    rgb = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    depth = np.random.rand(256, 256).astype(np.float32)

    status, phase = engine.generate_hologram(rgb, depth)
    status, phase_u8 = engine.generate_hologram_quantized(rgb, depth)

    engine.shutdown()
"""

from .engine import (
    EngineAPI,
    EngineConfig,
    PhaseFormat,
    ColorSpace,
    ExecutionProvider,
    Status,
    PreProcessor,
    InferenceCore,
    IFFTPostProcessor,
)

from .multi_wavelength import (
    RGBHologramEngine,
    RGBEngineConfig,
    CombineMode,
    SpatialPattern,
    WAVELENGTH_R,
    WAVELENGTH_G,
    WAVELENGTH_B,
)

from .realtime import (
    RealtimeHologramDisplay,
    RealtimeConfig,
    CameraType,
    SLABackend,
    RealSenseCamera,
    KinectCamera,
    TestPatternSource,
)

from .slm_driver import (
    SLMDriver,
    DirectDisplayBackend,
    SDKBackend,
    FileBackend,
    create_slm_driver,
)

from .slm_holoeye import (
    HoloeyeSLM,
    HOLOEYE_MODELS,
)

from .slm_meadowlark import (
    MeadowlarkSLM,
    MEADOWLARK_MODELS,
)

from .slm_hamamatsu import (
    HamamatsuSLM,
    HAMAMATSU_MODELS,
)

from .slm_manager import (
    SLMManager,
    SLMDeviceInfo,
)

from .gpu_engine import (
    GPUEngineAPI,
    GPUConfig,
    CuPyIFFTPostProcessor,
    GPUInferenceCore,
    detect_available_providers,
)

# Try to import C++ engine (optional — falls back to pure Python)
try:
    from ._deepcgh_engine import DeepCGHEngine as CppDeepCGHEngine
    _CPP_ENGINE_AVAILABLE = True
except ImportError:
    CppDeepCGHEngine = None
    _CPP_ENGINE_AVAILABLE = False

__version__ = "1.0.0"
__all__ = [
    "EngineAPI",
    "EngineConfig",
    "PhaseFormat",
    "ColorSpace",
    "ExecutionProvider",
    "Status",
    "PreProcessor",
    "InferenceCore",
    "IFFTPostProcessor",
    "CppDeepCGHEngine",
    "RGBHologramEngine",
    "RGBEngineConfig",
    "CombineMode",
    "SpatialPattern",
    "WAVELENGTH_R",
    "WAVELENGTH_G",
    "WAVELENGTH_B",
    "SLMDriver",
    "DirectDisplayBackend",
    "SDKBackend",
    "FileBackend",
    "create_slm_driver",
    "HoloeyeSLM",
    "HOLOEYE_MODELS",
    "MeadowlarkSLM",
    "MEADOWLARK_MODELS",
    "HamamatsuSLM",
    "HAMAMATSU_MODELS",
    "SLMManager",
    "SLMDeviceInfo",
    "RealtimeHologramDisplay",
    "RealtimeConfig",
    "CameraType",
    "SLABackend",
    "RealSenseCamera",
    "KinectCamera",
    "TestPatternSource",
    "GPUEngineAPI",
    "GPUConfig",
    "CuPyIFFTPostProcessor",
    "GPUInferenceCore",
    "detect_available_providers",
]
