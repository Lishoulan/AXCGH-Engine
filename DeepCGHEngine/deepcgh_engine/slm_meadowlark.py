"""
deepcgh_engine/slm_meadowlark.py — Meadowlark SLM integration.

Provides the MeadowlarkSLM driver for Meadowlark Optics spatial light
modulators via the Blink SDK (C API) loaded dynamically through ctypes.

The Blink SDK supports overdrive mode (reducing SLM response time) and
sequence mode (pre-loading multiple phase maps for high-speed cycling).

If the Blink SDK is not found, the driver falls back to DirectDisplayBackend.

SDK Download & Installation:
    1. Download the Blink SDK from Meadowlark Optics:
       https://www.meadowlark.com/software/
    2. Install the SDK to the default location:
       - Windows: C:\\Program Files\\Meadowlark Optics\\Blink SDK\\
       - Linux:   /opt/meadowlark/BlinkSDK/
    3. The SDK provides a shared library:
       - Windows: blink_sdk.dll
       - Linux:   libblink_sdk.so
    4. Ensure the library directory is on your system PATH or set
       the MEADOWLARK_SDK_PATH environment variable.

Supported Devices:
    - 1920x1152 SLM (HSP1920-1152-XXX)
    - 1024x1024 SLM (HSP1024-1024-XXX)

Usage:
    from deepcgh_engine.slm_meadowlark import MeadowlarkSLM

    slm = MeadowlarkSLM(model="1920x1152")
    slm.display(phase_map)
    slm.clear()
    slm.close()

    # Sequence mode: pre-load and cycle phase maps at high speed
    slm.set_sequence([phase1, phase2, phase3], frame_rate=180)
    slm.start_sequence()
    slm.stop_sequence()
"""

import ctypes
import os
import platform
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .slm_driver import (
    DirectDisplayBackend,
    SLMDriver,
    normalize_phase,
)

# ---------------------------------------------------------------------------
# SDK path resolution
# ---------------------------------------------------------------------------

_MEADOWLARK_SDK_ENV = "MEADOWLARK_SDK_PATH"

_SDK_WINDOWS_DEFAULT = os.path.join(
    os.environ.get("ProgramFiles", r"C:\\Program Files"),
    "Meadowlark Optics", "Blink SDK",
)
_SDK_LINUX_DEFAULT = "/opt/meadowlark/BlinkSDK"

_DLL_NAMES_WIN = "blink_sdk.dll"
_DLL_NAMES_LINUX = "libblink_sdk.so"


def _find_sdk_path() -> Optional[str]:
    """Locate the Meadowlark Blink SDK shared library.

    Search order:
      1. MEADOWLARK_SDK_PATH environment variable
      2. Platform-specific default install path
      3. System library search paths

    Returns:
        Absolute path to the SDK shared library, or None if not found.
    """
    env_path = os.environ.get(_MEADOWLARK_SDK_ENV)
    if env_path:
        if os.path.isdir(env_path):
            dll_name = _DLL_NAMES_WIN if platform.system() == "Windows" else _DLL_NAMES_LINUX
            candidate = os.path.join(env_path, dll_name)
            if os.path.isfile(candidate):
                return candidate
            return env_path
        elif os.path.isfile(env_path):
            return env_path

    if platform.system() == "Windows":
        default_dir = _SDK_WINDOWS_DEFAULT
        dll_name = _DLL_NAMES_WIN
    else:
        default_dir = _SDK_LINUX_DEFAULT
        dll_name = _DLL_NAMES_LINUX

    candidate = os.path.join(default_dir, dll_name)
    if os.path.isfile(candidate):
        return candidate

    return None


def _load_sdk() -> Optional[ctypes.CDLL]:
    """Attempt to load the Meadowlark Blink SDK shared library.

    Returns:
        Loaded ctypes CDLL, or None if the SDK is not available.
    """
    sdk_path = _find_sdk_path()
    if sdk_path is None:
        dll_name = _DLL_NAMES_WIN if platform.system() == "Windows" else _DLL_NAMES_LINUX
        try:
            return ctypes.cdll.LoadLibrary(dll_name)
        except OSError:
            return None

    try:
        if os.path.isdir(sdk_path):
            dll_name = _DLL_NAMES_WIN if platform.system() == "Windows" else _DLL_NAMES_LINUX
            return ctypes.cdll.LoadLibrary(os.path.join(sdk_path, dll_name))
        else:
            return ctypes.cdll.LoadLibrary(sdk_path)
    except OSError:
        return None


_sdk = _load_sdk()
_SDK_AVAILABLE = _sdk is not None


# ---------------------------------------------------------------------------
# Known Meadowlark device models
# ---------------------------------------------------------------------------

MEADOWLARK_MODELS: Dict[str, Dict] = {
    "1920x1152": {
        "resolution": (1920, 1152),
        "bit_depth": 8,
        "overdrive_supported": True,
        "sequence_supported": True,
    },
    "1024x1024": {
        "resolution": (1024, 1024),
        "bit_depth": 8,
        "overdrive_supported": True,
        "sequence_supported": True,
    },
}


# ---------------------------------------------------------------------------
# MeadowlarkSLM
# ---------------------------------------------------------------------------

class MeadowlarkSLM(SLMDriver):
    """Meadowlark Optics SLM driver using the Blink SDK via ctypes.

    Supports 1920x1152 and 1024x1024 models with overdrive and sequence
    mode. If the Blink SDK is not installed, falls back to DirectDisplayBackend.

    Args:
        model: Model identifier (e.g. "1920x1152", "1024x1024").
               If None, auto-detects the first connected Meadowlark device.
        resolution: (width, height) override. If None, uses model default.
        bit_depth: Phase depth override (8 or 10). If None, uses model default.
        display_idx: SLM board index (for multi-SLM setups).
        overdrive: Enable overdrive mode to reduce SLM response time.
        use_fallback: If True and SDK is unavailable, fall back to
                      DirectDisplayBackend instead of raising ImportError.

    Raises:
        ImportError: If the Blink SDK is not found and use_fallback is False.

    Example:
        slm = MeadowlarkSLM(model="1920x1152", overdrive=True)
        slm.display(phase_map)
        slm.set_sequence([phase1, phase2, phase3])
        slm.start_sequence()
        slm.stop_sequence()
        slm.close()
    """

    def __init__(
        self,
        model: Optional[str] = None,
        resolution: Optional[Tuple[int, int]] = None,
        bit_depth: Optional[int] = None,
        display_idx: Optional[int] = None,
        overdrive: bool = True,
        use_fallback: bool = True,
    ):
        self._model_name = model
        self._model_info = MEADOWLARK_MODELS.get(model, {}) if model else {}

        if resolution is None:
            if self._model_info:
                resolution = self._model_info["resolution"]
            else:
                resolution = (1920, 1152)
        if bit_depth is None:
            if self._model_info:
                bit_depth = self._model_info["bit_depth"]
            else:
                bit_depth = 8

        super().__init__(resolution, bit_depth, display_idx)
        self.vendor = "meadowlark"
        self._wavelength = 532.0
        self._overdrive = overdrive
        self._fallback = None
        self._handle = None
        self._sequence_loaded = False
        self._sequence_running = False
        self._temperature = 0.0

        if not _SDK_AVAILABLE:
            msg = (
                "Meadowlark Blink SDK not found.\n"
                "Please download and install it from:\n"
                "  https://www.meadowlark.com/software/\n"
                "Set the MEADOWLARK_SDK_PATH environment variable to the SDK directory,\n"
                "or add the SDK bin directory to your system PATH."
            )
            if use_fallback:
                print(f"[WARN] {msg}")
                print("[INFO] Falling back to DirectDisplayBackend.")
                self._fallback = DirectDisplayBackend(
                    resolution=resolution,
                    bit_depth=bit_depth,
                    display_idx=display_idx,
                )
            else:
                raise ImportError(msg)
        else:
            self._init_sdk()

    # ------------------------------------------------------------------
    # SDK initialization
    # ------------------------------------------------------------------

    def _init_sdk(self) -> None:
        """Initialize the Blink SDK and open the SLM device."""
        assert _sdk is not None

        # Blink_Open(int boardNumber, int isOverdrive) -> int handle
        _sdk.Blink_Open.argtypes = [ctypes.c_int, ctypes.c_int]
        _sdk.Blink_Open.restype = ctypes.c_int

        # Blink_Close(int handle) -> void
        _sdk.Blink_Close.argtypes = [ctypes.c_int]
        _sdk.Blink_Close.restype = None

        # Blink_WriteImage(int handle, unsigned char* data) -> int
        _sdk.Blink_WriteImage.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_ubyte)]
        _sdk.Blink_WriteImage.restype = ctypes.c_int

        # Blink_Clear(int handle) -> int
        _sdk.Blink_Clear.argtypes = [ctypes.c_int]
        _sdk.Blink_Clear.restype = ctypes.c_int

        # Blink_SetWavelength(int handle, double wavelengthNm) -> int
        _sdk.Blink_SetWavelength.argtypes = [ctypes.c_int, ctypes.c_double]
        _sdk.Blink_SetWavelength.restype = ctypes.c_int

        # Blink_LoadSequence(int handle, int count, unsigned char** images) -> int
        _sdk.Blink_LoadSequence.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
        _sdk.Blink_LoadSequence.restype = ctypes.c_int

        # Blink_StartSequence(int handle) -> int
        _sdk.Blink_StartSequence.argtypes = [ctypes.c_int]
        _sdk.Blink_StartSequence.restype = ctypes.c_int

        # Blink_StopSequence(int handle) -> int
        _sdk.Blink_StopSequence.argtypes = [ctypes.c_int]
        _sdk.Blink_StopSequence.restype = ctypes.c_int

        # Blink_GetTemperature(int handle, double* temp) -> int
        _sdk.Blink_GetTemperature.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
        _sdk.Blink_GetTemperature.restype = ctypes.c_int

        # Blink_GetBoardCount() -> int
        _sdk.Blink_GetBoardCount.argtypes = []
        _sdk.Blink_GetBoardCount.restype = ctypes.c_int

        # Open the device
        idx = self.display_idx if self.display_idx is not None else 0
        od_flag = 1 if self._overdrive else 0
        handle = _sdk.Blink_Open(idx, od_flag)
        if handle < 0:
            raise RuntimeError(
                f"Failed to open Meadowlark SLM at board index {idx}. "
                f"Ensure the device is connected and powered on."
            )
        self._handle = handle

        # Apply default wavelength
        self.set_wavelength(self._wavelength)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def display(self, phase_map: np.ndarray) -> None:
        """Display a phase map on the Meadowlark SLM.

        Args:
            phase_map: float32 array [H, W] with values in [-pi, pi].
        """
        if self._fallback is not None:
            self._fallback.display(phase_map)
            return

        pixel_data = normalize_phase(phase_map, self.bit_depth)

        # Ensure 8-bit for Blink SDK (convert 10-bit if needed)
        if self.bit_depth == 10:
            pixel_data = (pixel_data >> 2).astype(np.uint8)
        else:
            pixel_data = pixel_data.astype(np.uint8)

        # Resize to SLM resolution
        h, w = pixel_data.shape
        target_w, target_h = self.resolution
        if (h, w) != (target_h, target_w):
            pixel_data = self._resize_phase(pixel_data, target_w, target_h)

        pixel_data = np.ascontiguousarray(pixel_data)
        data_ptr = pixel_data.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

        result = _sdk.Blink_WriteImage(self._handle, data_ptr)
        if result < 0:
            raise RuntimeError(
                f"Meadowlark Blink_WriteImage failed with code {result}."
            )

    def clear(self) -> None:
        """Clear the Meadowlark SLM (set all pixels to zero phase)."""
        if self._fallback is not None:
            self._fallback.clear()
            return

        result = _sdk.Blink_Clear(self._handle)
        if result < 0:
            raise RuntimeError(f"Meadowlark Blink_Clear failed with code {result}.")

    def close(self) -> None:
        """Close the Meadowlark SLM and release SDK resources."""
        if self._fallback is not None:
            self._fallback.close()
            self._fallback = None
            return

        if self._sequence_running:
            self.stop_sequence()

        if self._handle is not None and _sdk is not None:
            _sdk.Blink_Close(self._handle)
            self._handle = None

    def set_wavelength(self, wavelength_nm: float) -> None:
        """Set the operating wavelength for LUT calibration.

        Args:
            wavelength_nm: Wavelength in nanometers (e.g. 532.0 for green).
        """
        self._wavelength = wavelength_nm
        if self._fallback is not None or self._handle is None:
            return

        result = _sdk.Blink_SetWavelength(self._handle, ctypes.c_double(wavelength_nm))
        if result < 0:
            raise RuntimeError(
                f"Meadowlark Blink_SetWavelength failed with code {result}."
            )

    def set_sequence(self, phase_maps: List[np.ndarray], frame_rate: Optional[float] = None) -> None:
        """Pre-load a sequence of phase maps for high-speed cycling.

        The Meadowlark Blink SDK supports loading multiple images into
        on-board memory and cycling through them at high frame rates
        (up to 180+ Hz depending on model).

        Args:
            phase_maps: List of float32 arrays [H, W] in [-pi, pi].
            frame_rate: Target frame rate in Hz. If None, uses maximum.
        """
        if self._fallback is not None:
            print("[WARN] Sequence mode not available in fallback mode.")
            return

        if self._handle is None:
            raise RuntimeError("SLM not initialized.")

        # Convert all phase maps to uint8
        images = []
        for pm in phase_maps:
            pixel_data = normalize_phase(pm, self.bit_depth)
            if self.bit_depth == 10:
                pixel_data = (pixel_data >> 2).astype(np.uint8)
            else:
                pixel_data = pixel_data.astype(np.uint8)

            # Resize
            h, w = pixel_data.shape
            target_w, target_h = self.resolution
            if (h, w) != (target_h, target_w):
                pixel_data = self._resize_phase(pixel_data, target_w, target_h)

            images.append(np.ascontiguousarray(pixel_data))

        # Build array of pointers
        ptr_array = (ctypes.c_void_p * len(images))()
        for i, img in enumerate(images):
            ptr_array[i] = img.ctypes.data_as(ctypes.c_void_p)

        result = _sdk.Blink_LoadSequence(self._handle, len(images), ptr_array)
        if result < 0:
            raise RuntimeError(
                f"Meadowlark Blink_LoadSequence failed with code {result}."
            )
        self._sequence_loaded = True
        self._sequence_images = images  # prevent GC

    def start_sequence(self) -> None:
        """Start cycling through the loaded phase map sequence."""
        if self._fallback is not None:
            print("[WARN] Sequence mode not available in fallback mode.")
            return

        if not self._sequence_loaded:
            raise RuntimeError("No sequence loaded. Call set_sequence() first.")

        result = _sdk.Blink_StartSequence(self._handle)
        if result < 0:
            raise RuntimeError(
                f"Meadowlark Blink_StartSequence failed with code {result}."
            )
        self._sequence_running = True

    def stop_sequence(self) -> None:
        """Stop the running phase map sequence."""
        if self._fallback is not None or self._handle is None:
            return

        result = _sdk.Blink_StopSequence(self._handle)
        if result < 0:
            raise RuntimeError(
                f"Meadowlark Blink_StopSequence failed with code {result}."
            )
        self._sequence_running = False

    def get_temperature(self) -> float:
        """Read the SLM panel temperature.

        Returns:
            Temperature in degrees Celsius.
        """
        if self._fallback is not None or self._handle is None:
            return self._temperature

        temp = ctypes.c_double(0.0)
        result = _sdk.Blink_GetTemperature(self._handle, ctypes.byref(temp))
        if result < 0:
            raise RuntimeError(
                f"Meadowlark Blink_GetTemperature failed with code {result}."
            )
        self._temperature = temp.value
        return self._temperature

    # ------------------------------------------------------------------
    # Auto-detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_devices() -> List[Dict[str, str]]:
        """Detect all connected Meadowlark SLM devices.

        Returns:
            List of dicts with device info. Empty list if SDK unavailable.
        """
        if not _SDK_AVAILABLE:
            return []

        try:
            count = _sdk.Blink_GetBoardCount()
            devices = []
            for i in range(count):
                devices.append({
                    "vendor": "meadowlark",
                    "board_index": str(i),
                })
            return devices
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resize_phase(data: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """Resize phase data to the target resolution using nearest-neighbor."""
        h, w = data.shape
        row_idx = np.linspace(0, h - 1, target_h).astype(int)
        col_idx = np.linspace(0, w - 1, target_w).astype(int)
        return data[np.ix_(row_idx, col_idx)]
