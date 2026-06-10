"""
deepcgh_engine/slm_hamamatsu.py — Hamamatsu LCOS SLM integration.

Provides the HamamatsuSLM driver for Hamamatsu LCOS spatial light
modulators (X13138 series) via the Hamamatsu LCOS SDK loaded
dynamically through ctypes.

If the Hamamatsu LCOS SDK is not found, the driver falls back to
DirectDisplayBackend.

SDK Download & Installation:
    1. Download the LCOS SLM SDK from Hamamatsu:
       https://hamamatsu.com/eu/en/product/optical-components/spatial-light-modulator/index.html
    2. Install the SDK to the default location:
       - Windows: C:\\Program Files\\Hamamatsu\\LCOS_SLM\\
       - Linux:   /opt/hamamatsu/LCOS_SLM/
    3. The SDK provides a shared library:
       - Windows: lcos_sdk.dll
       - Linux:   liblcos_sdk.so
    4. Ensure the library directory is on your system PATH or set
       the HAMAMATSU_SDK_PATH environment variable.

Supported Devices:
    - X13138-01 (1272x1024, 8-bit phase)
    - X13138-02 (1272x1024, 8-bit phase, high-reflectivity)
    - X13138-03 (1272x1024, 8-bit phase, wide-band)

Usage:
    from deepcgh_engine.slm_hamamatsu import HamamatsuSLM

    slm = HamamatsuSLM(model="X13138-01")
    slm.display(phase_map)
    slm.clear()
    slm.close()
"""

import ctypes
import os
import platform
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

_HAMAMATSU_SDK_ENV = "HAMAMATSU_SDK_PATH"

_SDK_WINDOWS_DEFAULT = os.path.join(
    os.environ.get("ProgramFiles", r"C:\\Program Files"),
    "Hamamatsu", "LCOS_SLM",
)
_SDK_LINUX_DEFAULT = "/opt/hamamatsu/LCOS_SLM"

_DLL_NAMES_WIN = "lcos_sdk.dll"
_DLL_NAMES_LINUX = "liblcos_sdk.so"


def _find_sdk_path() -> Optional[str]:
    """Locate the Hamamatsu LCOS SDK shared library.

    Search order:
      1. HAMAMATSU_SDK_PATH environment variable
      2. Platform-specific default install path
      3. System library search paths

    Returns:
        Absolute path to the SDK shared library, or None if not found.
    """
    env_path = os.environ.get(_HAMAMATSU_SDK_ENV)
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
    """Attempt to load the Hamamatsu LCOS SDK shared library.

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
# Known Hamamatsu device models
# ---------------------------------------------------------------------------

HAMAMATSU_MODELS: Dict[str, Dict] = {
    "X13138-01": {
        "resolution": (1272, 1024),
        "bit_depth": 8,
        "description": "Standard reflectivity",
    },
    "X13138-02": {
        "resolution": (1272, 1024),
        "bit_depth": 8,
        "description": "High reflectivity",
    },
    "X13138-03": {
        "resolution": (1272, 1024),
        "bit_depth": 8,
        "description": "Wide-band",
    },
}


# ---------------------------------------------------------------------------
# HamamatsuSLM
# ---------------------------------------------------------------------------

class HamamatsuSLM(SLMDriver):
    """Hamamatsu LCOS SLM driver using the LCOS SDK via ctypes.

    Supports X13138 series devices. If the Hamamatsu LCOS SDK is not
    installed, falls back to DirectDisplayBackend.

    Args:
        model: Model identifier (e.g. "X13138-01").
               If None, auto-detects the first connected Hamamatsu device.
        resolution: (width, height) override. If None, uses model default.
        bit_depth: Phase depth override (8 or 10). If None, uses model default.
        display_idx: SLM board index (for multi-SLM setups).
        use_fallback: If True and SDK is unavailable, fall back to
                      DirectDisplayBackend instead of raising ImportError.

    Raises:
        ImportError: If the Hamamatsu LCOS SDK is not found and
                     use_fallback is False.

    Example:
        slm = HamamatsuSLM(model="X13138-01")
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
        self._model_name = model
        self._model_info = HAMAMATSU_MODELS.get(model, {}) if model else {}

        if resolution is None:
            if self._model_info:
                resolution = self._model_info["resolution"]
            else:
                resolution = (1272, 1024)
        if bit_depth is None:
            if self._model_info:
                bit_depth = self._model_info["bit_depth"]
            else:
                bit_depth = 8

        super().__init__(resolution, bit_depth, display_idx)
        self.vendor = "hamamatsu"
        self._wavelength = 532.0
        self._fallback = None
        self._handle = None

        if not _SDK_AVAILABLE:
            msg = (
                "Hamamatsu LCOS SDK not found.\n"
                "Please download and install it from:\n"
                "  https://hamamatsu.com/eu/en/product/optical-components/spatial-light-modulator/index.html\n"
                "Set the HAMAMATSU_SDK_PATH environment variable to the SDK directory,\n"
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
        """Initialize the Hamamatsu LCOS SDK and open the SLM device."""
        assert _sdk is not None

        # LCOS_Open(int boardNumber) -> int handle
        _sdk.LCOS_Open.argtypes = [ctypes.c_int]
        _sdk.LCOS_Open.restype = ctypes.c_int

        # LCOS_Close(int handle) -> void
        _sdk.LCOS_Close.argtypes = [ctypes.c_int]
        _sdk.LCOS_Close.restype = None

        # LCOS_WriteImage(int handle, unsigned char* data, int width, int height) -> int
        _sdk.LCOS_WriteImage.argtypes = [
            ctypes.c_int, ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int, ctypes.c_int,
        ]
        _sdk.LCOS_WriteImage.restype = ctypes.c_int

        # LCOS_Clear(int handle) -> int
        _sdk.LCOS_Clear.argtypes = [ctypes.c_int]
        _sdk.LCOS_Clear.restype = ctypes.c_int

        # LCOS_SetWavelength(int handle, double wavelengthNm) -> int
        _sdk.LCOS_SetWavelength.argtypes = [ctypes.c_int, ctypes.c_double]
        _sdk.LCOS_SetWavelength.restype = ctypes.c_int

        # LCOS_GetDeviceInfo(int handle, char* buffer, int bufferSize) -> int
        _sdk.LCOS_GetDeviceInfo.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        _sdk.LCOS_GetDeviceInfo.restype = ctypes.c_int

        # LCOS_GetDeviceCount() -> int
        _sdk.LCOS_GetDeviceCount.argtypes = []
        _sdk.LCOS_GetDeviceCount.restype = ctypes.c_int

        # Open the device
        idx = self.display_idx if self.display_idx is not None else 0
        handle = _sdk.LCOS_Open(idx)
        if handle < 0:
            raise RuntimeError(
                f"Failed to open Hamamatsu LCOS SLM at board index {idx}. "
                f"Ensure the device is connected and powered on."
            )
        self._handle = handle

        # Apply default wavelength
        self.set_wavelength(self._wavelength)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def display(self, phase_map: np.ndarray) -> None:
        """Display a phase map on the Hamamatsu LCOS SLM.

        Args:
            phase_map: float32 array [H, W] with values in [-pi, pi].
        """
        if self._fallback is not None:
            self._fallback.display(phase_map)
            return

        pixel_data = normalize_phase(phase_map, self.bit_depth)

        # Convert to uint8 for LCOS SDK
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

        result = _sdk.LCOS_WriteImage(self._handle, data_ptr, target_w, target_h)
        if result < 0:
            raise RuntimeError(
                f"Hamamatsu LCOS_WriteImage failed with code {result}."
            )

    def clear(self) -> None:
        """Clear the Hamamatsu LCOS SLM (set all pixels to zero phase)."""
        if self._fallback is not None:
            self._fallback.clear()
            return

        result = _sdk.LCOS_Clear(self._handle)
        if result < 0:
            raise RuntimeError(f"Hamamatsu LCOS_Clear failed with code {result}.")

    def close(self) -> None:
        """Close the Hamamatsu LCOS SLM and release SDK resources."""
        if self._fallback is not None:
            self._fallback.close()
            self._fallback = None
            return

        if self._handle is not None and _sdk is not None:
            _sdk.LCOS_Close(self._handle)
            self._handle = None

    def set_wavelength(self, wavelength_nm: float) -> None:
        """Set the operating wavelength for LUT calibration.

        Args:
            wavelength_nm: Wavelength in nanometers (e.g. 532.0 for green).
        """
        self._wavelength = wavelength_nm
        if self._fallback is not None or self._handle is None:
            return

        result = _sdk.LCOS_SetWavelength(self._handle, ctypes.c_double(wavelength_nm))
        if result < 0:
            raise RuntimeError(
                f"Hamamatsu LCOS_SetWavelength failed with code {result}."
            )

    def get_device_info(self) -> Dict[str, str]:
        """Query device information from the Hamamatsu LCOS SLM.

        Returns:
            Dictionary with keys like 'model', 'serial', 'firmware', etc.
        """
        if self._fallback is not None or self._handle is None:
            return {
                "model": self._model_name or "unknown",
                "vendor": "hamamatsu",
                "status": "fallback_mode",
            }

        buf_size = 512
        buf = ctypes.create_string_buffer(buf_size)
        result = _sdk.LCOS_GetDeviceInfo(self._handle, buf, buf_size)
        if result < 0:
            return {"model": "unknown", "vendor": "hamamatsu", "error": f"code {result}"}

        info_str = buf.value.decode("utf-8", errors="replace")
        info = {"vendor": "hamamatsu"}
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
        """Detect all connected Hamamatsu LCOS SLM devices.

        Returns:
            List of dicts with device info. Empty list if SDK unavailable.
        """
        if not _SDK_AVAILABLE:
            return []

        try:
            count = _sdk.LCOS_GetDeviceCount()
            devices = []
            for i in range(count):
                buf_size = 512
                buf = ctypes.create_string_buffer(buf_size)
                _sdk.LCOS_GetDeviceInfo(i, buf, buf_size)
                info_str = buf.value.decode("utf-8", errors="replace")
                info = {"vendor": "hamamatsu", "board_index": str(i)}
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
