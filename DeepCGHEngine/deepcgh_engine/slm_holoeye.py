"""
deepcgh_engine/slm_holoeye.py — Holoeye SLM SDK integration.

Provides the HoloeyeSLM driver for Holoeye spatial light modulators
(PLUTO, GAEA, LUNA device families) via the Holoeye SLM Display SDK.

The SDK is loaded dynamically via ctypes, so there is no compile-time
dependency. If the SDK DLL/shared library is not found, the module
falls back to DirectDisplayBackend.

SDK Download & Installation:
    1. Download the Holoeye SLM Display SDK from:
       https://holoeye.com/software-downloads/
    2. Install the SDK to the default location:
       - Windows: C:\\Program Files\\HOLOEYE\\SLM Display SDK\\
       - Linux:   /opt/holoeye/SLM_Display_SDK/
    3. The SDK provides a shared library:
       - Windows: slm_display_sdk.dll
       - Linux:   libslm_display_sdk.so
    4. Ensure the library directory is on your system PATH or set
       the HOLOEYE_SDK_PATH environment variable to the install dir.

Supported Devices:
    - PLUTO-2  (1920x1080, 8-bit phase)
    - PLUTO-2.1 (1920x1080, 10-bit phase)
    - GAEA-2   (3840x2160, 8/10-bit phase)
    - LUNA     (1024x768, 8-bit phase)

Usage:
    from deepcgh_engine.slm_holoeye import HoloeyeSLM

    slm = HoloeyeSLM(model="PLUTO-2")
    slm.display(phase_map)   # phase_map: float32 [H, W] in [-pi, pi]
    slm.clear()
    slm.close()
"""

import ctypes
import os
import platform
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from .slm_driver import (
    DirectDisplayBackend,
    SLMDriver,
    normalize_phase,
    normalize_phase_to_uint8,
    normalize_phase_to_uint16_10bit,
)

# ---------------------------------------------------------------------------
# SDK path resolution
# ---------------------------------------------------------------------------

_HOLOEYE_SDK_ENV = "HOLOEYE_SDK_PATH"

_SDK_WINDOWS_DEFAULT = os.path.join(
    os.environ.get("ProgramFiles", r"C:\\Program Files"),
    "HOLOEYE", "SLM Display SDK",
)
_SDK_LINUX_DEFAULT = "/opt/holoeye/SLM_Display_SDK"

_DLL_NAMES_WIN = "slm_display_sdk.dll"
_DLL_NAMES_LINUX = "libslm_display_sdk.so"


def _find_sdk_path() -> Optional[str]:
    """Locate the Holoeye SDK shared library on disk.

    Search order:
      1. HOLOEYE_SDK_PATH environment variable
      2. Platform-specific default install path
      3. System library search paths (PATH / LD_LIBRARY_PATH)

    Returns:
        Absolute path to the SDK shared library, or None if not found.
    """
    env_path = os.environ.get(_HOLOEYE_SDK_ENV)
    if env_path:
        if os.path.isdir(env_path):
            dll_name = _DLL_NAMES_WIN if platform.system() == "Windows" else _DLL_NAMES_LINUX
            candidate = os.path.join(env_path, dll_name)
            if os.path.isfile(candidate):
                return candidate
            # Try loading from directory directly (ctypes will search PATH)
            return env_path
        elif os.path.isfile(env_path):
            return env_path

    # Platform defaults
    if platform.system() == "Windows":
        default_dir = _SDK_WINDOWS_DEFAULT
        dll_name = _DLL_NAMES_WIN
    else:
        default_dir = _SDK_LINUX_DEFAULT
        dll_name = _DLL_NAMES_LINUX

    candidate = os.path.join(default_dir, dll_name)
    if os.path.isfile(candidate):
        return candidate

    # Let ctypes try the system search paths
    return None


def _load_sdk() -> Optional[ctypes.CDLL]:
    """Attempt to load the Holoeye SDK shared library.

    Returns:
        Loaded ctypes CDLL, or None if the SDK is not available.
    """
    sdk_path = _find_sdk_path()
    if sdk_path is None:
        # Try loading by name alone (relies on PATH / LD_LIBRARY_PATH)
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


# Try loading the SDK at import time (non-fatal)
_sdk = _load_sdk()
_SDK_AVAILABLE = _sdk is not None


# ---------------------------------------------------------------------------
# Known Holoeye device models
# ---------------------------------------------------------------------------

HOLOEYE_MODELS: Dict[str, Dict] = {
    "PLUTO-2": {
        "resolution": (1920, 1080),
        "bit_depth": 8,
        "family": "PLUTO",
    },
    "PLUTO-2.1": {
        "resolution": (1920, 1080),
        "bit_depth": 10,
        "family": "PLUTO",
    },
    "GAEA-2": {
        "resolution": (3840, 2160),
        "bit_depth": 10,
        "family": "GAEA",
    },
    "LUNA": {
        "resolution": (1024, 768),
        "bit_depth": 8,
        "family": "LUNA",
    },
}


# ---------------------------------------------------------------------------
# HoloeyeSLM
# ---------------------------------------------------------------------------

class HoloeyeSLM(SLMDriver):
    """Holoeye SLM driver using the SLM Display SDK via ctypes.

    Supports PLUTO, GAEA, and LUNA device families. If the Holoeye SDK
    is not installed, the driver falls back to DirectDisplayBackend.

    Args:
        model: Holoeye model name (e.g. "PLUTO-2", "GAEA-2", "LUNA").
               If None, auto-detects the first connected Holoeye device.
        resolution: (width, height) override. If None, uses model default.
        bit_depth: Phase depth override (8 or 10). If None, uses model default.
        display_idx: SLM display index (for multi-SLM setups).
        use_fallback: If True and SDK is unavailable, fall back to
                      DirectDisplayBackend instead of raising ImportError.

    Raises:
        ImportError: If the Holoeye SDK is not found and use_fallback is False.

    Example:
        slm = HoloeyeSLM(model="PLUTO-2")
        slm.display(phase_map)
        slm.clear()
        slm.close()
    """

    def __init__(
        self,
        model: Optional[str] = None,
        resolution: Optional[Tuple[int, int]] = None,
        bit_depth: Optional[int] = None,
        display_idx: Optional[int] = None,
        use_fallback: bool = True,
    ):
        # Resolve model defaults
        self._model_name = model
        self._model_info = HOLOEYE_MODELS.get(model, {}) if model else {}

        if resolution is None:
            if self._model_info:
                resolution = self._model_info["resolution"]
            else:
                resolution = (1920, 1080)
        if bit_depth is None:
            if self._model_info:
                bit_depth = self._model_info["bit_depth"]
            else:
                bit_depth = 8

        super().__init__(resolution, bit_depth, display_idx)
        self.vendor = "holoeye"
        self._wavelength = 532.0  # default nm
        self._lut_id = 0
        self._fallback = None
        self._handle = None

        if not _SDK_AVAILABLE:
            msg = (
                "Holoeye SLM Display SDK not found.\n"
                "Please download and install it from:\n"
                "  https://holoeye.com/software-downloads/\n"
                "Set the HOLOEYE_SDK_PATH environment variable to the SDK directory,\n"
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
        """Initialize the Holoeye SDK and open the SLM device."""
        assert _sdk is not None

        # Define SDK function prototypes
        # SLM_Open(int displayIndex) -> int handle
        _sdk.SLM_Open.argtypes = [ctypes.c_int]
        _sdk.SLM_Open.restype = ctypes.c_int

        # SLM_Close(int handle) -> void
        _sdk.SLM_Close.argtypes = [ctypes.c_int]
        _sdk.SLM_Close.restype = None

        # SLM_DisplayPhaseArray(int handle, void* data, int width, int height, int bitDepth) -> int
        _sdk.SLM_DisplayPhaseArray.argtypes = [
            ctypes.c_int, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        _sdk.SLM_DisplayPhaseArray.restype = ctypes.c_int

        # SLM_Clear(int handle) -> int
        _sdk.SLM_Clear.argtypes = [ctypes.c_int]
        _sdk.SLM_Clear.restype = ctypes.c_int

        # SLM_SetWavelength(int handle, double wavelengthNm) -> int
        _sdk.SLM_SetWavelength.argtypes = [ctypes.c_int, ctypes.c_double]
        _sdk.SLM_SetWavelength.restype = ctypes.c_int

        # SLM_SetLUT(int handle, int lutId) -> int
        _sdk.SLM_SetLUT.argtypes = [ctypes.c_int, ctypes.c_int]
        _sdk.SLM_SetLUT.restype = ctypes.c_int

        # SLM_GetDeviceInfo(int handle, char* buffer, int bufferSize) -> int
        _sdk.SLM_GetDeviceInfo.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        _sdk.SLM_GetDeviceInfo.restype = ctypes.c_int

        # SLM_GetDeviceCount() -> int
        _sdk.SLM_GetDeviceCount.argtypes = []
        _sdk.SLM_GetDeviceCount.restype = ctypes.c_int

        # Open the device
        idx = self.display_idx if self.display_idx is not None else 0
        handle = _sdk.SLM_Open(idx)
        if handle < 0:
            raise RuntimeError(
                f"Failed to open Holoeye SLM at display index {idx}. "
                f"Ensure the device is connected and powered on."
            )
        self._handle = handle

        # Apply default wavelength
        self.set_wavelength(self._wavelength)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def display(self, phase_map: np.ndarray) -> None:
        """Display a phase map on the Holoeye SLM.

        Args:
            phase_map: float32 array [H, W] with values in [-pi, pi].
        """
        if self._fallback is not None:
            self._fallback.display(phase_map)
            return

        pixel_data = normalize_phase(phase_map, self.bit_depth)

        # Ensure correct resolution
        h, w = pixel_data.shape
        target_w, target_h = self.resolution
        if (h, w) != (target_h, target_w):
            pixel_data = self._resize_phase(pixel_data, target_w, target_h)

        # Ensure contiguous C-order array
        pixel_data = np.ascontiguousarray(pixel_data)

        # Determine data pointer and bit depth flag
        if self.bit_depth == 8:
            data_ptr = pixel_data.ctypes.data_as(ctypes.c_void_p)
            bd_flag = 8
        else:
            data_ptr = pixel_data.ctypes.data_as(ctypes.c_void_p)
            bd_flag = 10

        result = _sdk.SLM_DisplayPhaseArray(
            self._handle, data_ptr,
            target_w, target_h, bd_flag,
        )
        if result < 0:
            raise RuntimeError(
                f"Holoeye SLM_DisplayPhaseArray failed with code {result}."
            )

    def clear(self) -> None:
        """Clear the Holoeye SLM (set all pixels to zero phase)."""
        if self._fallback is not None:
            self._fallback.clear()
            return

        result = _sdk.SLM_Clear(self._handle)
        if result < 0:
            raise RuntimeError(f"Holoeye SLM_Clear failed with code {result}.")

    def close(self) -> None:
        """Close the Holoeye SLM and release SDK resources."""
        if self._fallback is not None:
            self._fallback.close()
            self._fallback = None
            return

        if self._handle is not None and _sdk is not None:
            _sdk.SLM_Close(self._handle)
            self._handle = None

    def set_wavelength(self, wavelength_nm: float) -> None:
        """Set the operating wavelength for LUT calibration.

        Args:
            wavelength_nm: Wavelength in nanometers (e.g. 532.0 for green).
        """
        self._wavelength = wavelength_nm
        if self._fallback is not None or self._handle is None:
            return

        result = _sdk.SLM_SetWavelength(self._handle, ctypes.c_double(wavelength_nm))
        if result < 0:
            raise RuntimeError(
                f"Holoeye SLM_SetWavelength failed with code {result}."
            )

    def set_lut(self, lut_id: int) -> None:
        """Select a pre-calibrated Look-Up Table on the SLM.

        Holoeye SLMs store multiple LUTs for different wavelengths.
        LUT 0 is typically the default.

        Args:
            lut_id: LUT index (0-based).
        """
        self._lut_id = lut_id
        if self._fallback is not None or self._handle is None:
            return

        result = _sdk.SLM_SetLUT(self._handle, lut_id)
        if result < 0:
            raise RuntimeError(f"Holoeye SLM_SetLUT failed with code {result}.")

    def get_device_info(self) -> Dict[str, str]:
        """Query device information from the Holoeye SLM.

        Returns:
            Dictionary with keys like 'model', 'serial', 'firmware', etc.
        """
        if self._fallback is not None or self._handle is None:
            return {
                "model": self._model_name or "unknown",
                "vendor": "holoeye",
                "status": "fallback_mode",
            }

        buf_size = 512
        buf = ctypes.create_string_buffer(buf_size)
        result = _sdk.SLM_GetDeviceInfo(self._handle, buf, buf_size)
        if result < 0:
            return {"model": "unknown", "vendor": "holoeye", "error": f"code {result}"}

        info_str = buf.value.decode("utf-8", errors="replace")
        info = {"vendor": "holoeye"}
        for part in info_str.split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                info[k.strip()] = v.strip()
        return info

    # ------------------------------------------------------------------
    # Auto-detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_devices() -> List[Dict[str, str]]:
        """Detect all connected Holoeye SLM devices.

        Returns:
            List of dicts with device info. Empty list if SDK unavailable.
        """
        if not _SDK_AVAILABLE:
            return []

        try:
            count = _sdk.SLM_GetDeviceCount()
            devices = []
            for i in range(count):
                buf_size = 512
                buf = ctypes.create_string_buffer(buf_size)
                _sdk.SLM_GetDeviceInfo(i, buf, buf_size)
                info_str = buf.value.decode("utf-8", errors="replace")
                info = {"vendor": "holoeye", "display_index": str(i)}
                for part in info_str.split(";"):
                    if "=" in part:
                        k, v = part.split("=", 1)
                        info[k.strip()] = v.strip()
                devices.append(info)
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
