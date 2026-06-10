"""
deepcgh_engine/slm_manager.py — SLM device manager.

Provides the SLMManager class that auto-detects and manages connected
SLMs from all supported vendors (Holoeye, Meadowlark, Hamamatsu).

The manager scans for devices across all vendor SDKs, maintains a
registry of discovered devices, and provides a unified interface for
creating SLMDriver instances.

Usage:
    from deepcgh_engine.slm_manager import SLMManager

    mgr = SLMManager()

    # Scan for connected SLMs
    devices = mgr.scan_devices()

    # Get a specific device
    slm = mgr.get_device(vendor="holoeye", model="PLUTO-2")

    # List all supported models
    supported = mgr.list_supported()

    # Use the SLM
    slm.display(phase_map)
    slm.clear()
    slm.close()
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .slm_driver import SLMDriver

# Vendor driver imports (non-fatal if SDK not available)
from .slm_holoeye import HoloeyeSLM, HOLOEYE_MODELS
from .slm_meadowlark import MeadowlarkSLM, MEADOWLARK_MODELS
from .slm_hamamatsu import HamamatsuSLM, HAMAMATSU_MODELS


# ---------------------------------------------------------------------------
# Unified device info structure
# ---------------------------------------------------------------------------

class SLMDeviceInfo:
    """Describes a discovered or supported SLM device.

    Attributes:
        vendor: Vendor name (e.g. "holoeye", "meadowlark", "hamamatsu").
        model: Model identifier (e.g. "PLUTO-2", "1920x1152", "X13138-01").
        resolution: (width, height) in pixels.
        bit_depth: Phase quantization depth (8 or 10 bits).
        connected: Whether the device is currently connected.
        display_index: Device/board index if connected.
        extra: Additional vendor-specific metadata.
    """

    def __init__(
        self,
        vendor: str,
        model: str,
        resolution: Tuple[int, int],
        bit_depth: int = 8,
        connected: bool = False,
        display_index: Optional[int] = None,
        extra: Optional[Dict] = None,
    ):
        self.vendor = vendor
        self.model = model
        self.resolution = resolution
        self.bit_depth = bit_depth
        self.connected = connected
        self.display_index = display_index
        self.extra = extra or {}

    def __repr__(self) -> str:
        status = "connected" if self.connected else "supported"
        return (
            f"SLMDeviceInfo(vendor='{self.vendor}', model='{self.model}', "
            f"resolution={self.resolution}, bit_depth={self.bit_depth}, "
            f"status='{status}')"
        )


# ---------------------------------------------------------------------------
# Vendor driver registry
# ---------------------------------------------------------------------------

_VENDOR_REGISTRY: Dict[str, Dict] = {
    "holoeye": {
        "driver_class": HoloeyeSLM,
        "models": HOLOEYE_MODELS,
        "detect_fn": HoloeyeSLM.detect_devices,
    },
    "meadowlark": {
        "driver_class": MeadowlarkSLM,
        "models": MEADOWLARK_MODELS,
        "detect_fn": MeadowlarkSLM.detect_devices,
    },
    "hamamatsu": {
        "driver_class": HamamatsuSLM,
        "models": HAMAMATSU_MODELS,
        "detect_fn": HamamatsuSLM.detect_devices,
    },
}


# ---------------------------------------------------------------------------
# SLMManager
# ---------------------------------------------------------------------------

class SLMManager:
    """Auto-detect and manage connected SLMs from all supported vendors.

    Provides a unified interface for scanning devices, creating driver
    instances, and listing supported models.

    Example:
        mgr = SLMManager()
        devices = mgr.scan_devices()
        slm = mgr.get_device(vendor="holoeye", model="PLUTO-2")
        slm.display(phase_map)
        slm.close()
    """

    def __init__(self):
        self._devices: List[SLMDeviceInfo] = []
        self._drivers: Dict[str, SLMDriver] = {}  # key: "vendor/model"

    def scan_devices(self) -> List[SLMDeviceInfo]:
        """Scan for all connected SLMs across all supported vendors.

        Queries each vendor SDK for connected devices. Devices that are
        physically connected but whose SDK is not installed will not
        appear in the results.

        Returns:
            List of SLMDeviceInfo for all discovered devices.
        """
        self._devices = []

        for vendor_name, vendor_info in _VENDOR_REGISTRY.items():
            detect_fn = vendor_info["detect_fn"]
            models = vendor_info["models"]

            try:
                detected = detect_fn()
            except Exception:
                detected = []

            for dev_info in detected:
                # Try to match detected device to a known model
                model_name = dev_info.get("model", "unknown")
                display_idx = int(dev_info.get("display_index",
                                               dev_info.get("board_index", 0)))

                # Look up model specs
                model_specs = models.get(model_name, {})

                self._devices.append(SLMDeviceInfo(
                    vendor=vendor_name,
                    model=model_name,
                    resolution=model_specs.get("resolution", (0, 0)),
                    bit_depth=model_specs.get("bit_depth", 8),
                    connected=True,
                    display_index=display_idx,
                    extra=dev_info,
                ))

        return self._devices

    def get_device(
        self,
        vendor: str,
        model: Optional[str] = None,
        resolution: Optional[Tuple[int, int]] = None,
        bit_depth: Optional[int] = None,
        display_idx: Optional[int] = None,
        **kwargs,
    ) -> SLMDriver:
        """Get an SLM driver instance for a specific device.

        If the device is already instantiated (same vendor/model), returns
        the existing driver. Otherwise creates a new one.

        Args:
            vendor: Vendor name ("holoeye", "meadowlark", "hamamatsu").
            model: Model identifier. If None, uses the first available.
            resolution: (width, height) override.
            bit_depth: Phase depth override (8 or 10).
            display_idx: Device/board index override.
            **kwargs: Additional vendor-specific arguments (e.g. overdrive
                      for Meadowlark).

        Returns:
            An SLMDriver instance for the requested device.

        Raises:
            ValueError: If the vendor is not supported.
            RuntimeError: If no device is found for the given criteria.
        """
        vendor_lower = vendor.lower()
        if vendor_lower not in _VENDOR_REGISTRY:
            raise ValueError(
                f"Unsupported vendor: '{vendor}'. "
                f"Supported vendors: {list(_VENDOR_REGISTRY.keys())}"
            )

        # Check for existing driver
        cache_key = f"{vendor_lower}/{model or 'default'}"
        if cache_key in self._drivers:
            return self._drivers[cache_key]

        # Create new driver
        vendor_info = _VENDOR_REGISTRY[vendor_lower]
        driver_class = vendor_info["driver_class"]

        driver_kwargs = {}
        if model is not None:
            driver_kwargs["model"] = model
        if resolution is not None:
            driver_kwargs["resolution"] = resolution
        if bit_depth is not None:
            driver_kwargs["bit_depth"] = bit_depth
        if display_idx is not None:
            driver_kwargs["display_idx"] = display_idx
        driver_kwargs.update(kwargs)

        driver = driver_class(**driver_kwargs)
        self._drivers[cache_key] = driver
        return driver

    def list_supported(self) -> List[SLMDeviceInfo]:
        """List all supported SLM models across all vendors.

        Returns:
            List of SLMDeviceInfo for all known models (not necessarily
            connected).
        """
        supported = []
        for vendor_name, vendor_info in _VENDOR_REGISTRY.items():
            for model_name, model_specs in vendor_info["models"].items():
                supported.append(SLMDeviceInfo(
                    vendor=vendor_name,
                    model=model_name,
                    resolution=model_specs.get("resolution", (0, 0)),
                    bit_depth=model_specs.get("bit_depth", 8),
                    connected=False,
                    extra=model_specs,
                ))
        return supported

    def close_all(self) -> None:
        """Close all active SLM drivers and release resources."""
        for key, driver in self._drivers.items():
            try:
                driver.close()
            except Exception:
                pass
        self._drivers.clear()

    @property
    def devices(self) -> List[SLMDeviceInfo]:
        """Currently discovered devices (from last scan)."""
        return self._devices

    @property
    def active_drivers(self) -> Dict[str, SLMDriver]:
        """Currently active driver instances, keyed by 'vendor/model'."""
        return dict(self._drivers)
