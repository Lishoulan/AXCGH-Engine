"""
deepcgh_engine/slm_driver.py — Spatial Light Modulator (SLM) driver abstraction.

Provides a unified interface for driving SLMs through multiple backends:
  - DirectDisplayBackend: Renders phase maps on a secondary display via pygame/OpenCV
  - SDKBackend: Placeholder for vendor-specific SLM SDKs (Holoeye, Meadowlark, etc.)
  - FileBackend: Saves phase maps to image files for offline testing

Usage:
    from deepcgh_engine import create_slm_driver

    driver = create_slm_driver(backend='direct', resolution=(1920, 1080), bit_depth=8)
    driver.display(phase_map)   # phase_map: float32 [H, W] in [-pi, pi]
    driver.clear()
    driver.close()
"""

import abc
import os
from typing import Optional, Tuple, Union

import numpy as np

# Optional: screeninfo for display detection
try:
    import screeninfo
    _SCREENINFO_AVAILABLE = True
except ImportError:
    _SCREENINFO_AVAILABLE = False

# Optional: pygame for rendering
try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False

# Optional: OpenCV as fallback renderer
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


# ===========================================================================
# Phase map normalization utilities
# ===========================================================================

def normalize_phase_to_uint8(phase_map: np.ndarray) -> np.ndarray:
    """Normalize phase map from [-pi, pi] to [0, 255] uint8.

    Args:
        phase_map: float32 array with values in [-pi, pi].

    Returns:
        uint8 array with values in [0, 255].
    """
    normalized = (phase_map + np.pi) / (2 * np.pi)
    return np.clip(normalized * 255, 0, 255).astype(np.uint8)


def normalize_phase_to_uint16_10bit(phase_map: np.ndarray) -> np.ndarray:
    """Normalize phase map from [-pi, pi] to [0, 1023] stored as uint16.

    Args:
        phase_map: float32 array with values in [-pi, pi].

    Returns:
        uint16 array with values in [0, 1023].
    """
    normalized = (phase_map + np.pi) / (2 * np.pi)
    return np.clip(normalized * 1023, 0, 1023).astype(np.uint16)


def normalize_phase(phase_map: np.ndarray, bit_depth: int) -> np.ndarray:
    """Normalize phase map from [-pi, pi] to the appropriate integer range.

    Args:
        phase_map: float32 array with values in [-pi, pi].
        bit_depth: 8 or 10.

    Returns:
        Normalized integer array.
    """
    if bit_depth == 8:
        return normalize_phase_to_uint8(phase_map)
    elif bit_depth == 10:
        return normalize_phase_to_uint16_10bit(phase_map)
    else:
        raise ValueError(f"Unsupported bit_depth: {bit_depth}. Must be 8 or 10.")


# ===========================================================================
# SLMDriver base class
# ===========================================================================

class SLMDriver(abc.ABC):
    """Abstract base class for SLM drivers.

    All phase maps are expected as float32 arrays in [-pi, pi].
    Subclasses handle the conversion to device-specific format.

    Args:
        resolution: (width, height) of the SLM in pixels.
        bit_depth: Phase quantization depth (8 or 10 bits).
        display_idx: Index of the display to use (backend-specific).
    """

    def __init__(self, resolution: Tuple[int, int], bit_depth: int = 8,
                 display_idx: Optional[int] = None):
        if bit_depth not in (8, 10):
            raise ValueError(f"bit_depth must be 8 or 10, got {bit_depth}")
        self.resolution = resolution
        self.bit_depth = bit_depth
        self.display_idx = display_idx

    @abc.abstractmethod
    def display(self, phase_map: np.ndarray) -> None:
        """Display a phase map on the SLM.

        Args:
            phase_map: float32 array [H, W] with values in [-pi, pi].
        """

    @abc.abstractmethod
    def clear(self) -> None:
        """Clear the SLM display (set to zero phase)."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release resources and close the SLM connection."""


# ===========================================================================
# DirectDisplayBackend
# ===========================================================================

class DirectDisplayBackend(SLMDriver):
    """Display phase maps on a secondary monitor (SLM) via pygame or OpenCV.

    The SLM is treated as a secondary display connected via HDMI/DVI.
    A borderless fullscreen window is created on the target display.

    Args:
        resolution: (width, height) of the SLM in pixels.
        bit_depth: Phase quantization depth (8 or 10 bits).
        display_idx: Monitor index for the SLM. If None, auto-detects
                     the first non-primary display using screeninfo.
    """

    def __init__(self, resolution: Tuple[int, int], bit_depth: int = 8,
                 display_idx: Optional[int] = None):
        super().__init__(resolution, bit_depth, display_idx)

        self._renderer = None  # 'pygame' or 'cv2'
        self._screen = None
        self._cv2_window_name = "DeepCGH_SLM"
        self._display_pos = (0, 0)
        self._display_size = resolution
        self._initialized = False

        # Detect SLM display
        self._detect_display()

        # Initialize renderer
        self._init_renderer()

    def _detect_display(self):
        """Auto-detect secondary display using screeninfo."""
        if self.display_idx is not None:
            # User specified a display index — use it
            if _SCREENINFO_AVAILABLE:
                monitors = screeninfo.get_monitors()
                if self.display_idx < len(monitors):
                    mon = monitors[self.display_idx]
                    self._display_pos = (mon.x, mon.y)
                    self._display_size = (mon.width, mon.height)
                else:
                    print(f"[WARN] display_idx {self.display_idx} out of range "
                          f"({len(monitors)} monitors detected). Using default.")
            else:
                # Without screeninfo, assume the SLM is to the right of primary
                self._display_pos = (1920, 0)
                self._display_size = self.resolution
            return

        # Auto-detect: find first non-primary display
        if _SCREENINFO_AVAILABLE:
            monitors = screeninfo.get_monitors()
            if len(monitors) > 1:
                # Use the first non-primary monitor
                for i, mon in enumerate(monitors):
                    if not mon.is_primary:
                        self.display_idx = i
                        self._display_pos = (mon.x, mon.y)
                        self._display_size = (mon.width, mon.height)
                        return
                # All monitors are primary? Use the second one
                self.display_idx = 1
                mon = monitors[1]
                self._display_pos = (mon.x, mon.y)
                self._display_size = (mon.width, mon.height)
            else:
                print("[WARN] Only one display detected. SLM window will appear "
                      "on the primary display.")
                self.display_idx = 0
                mon = monitors[0]
                self._display_pos = (mon.x, mon.y)
                self._display_size = (mon.width, mon.height)
        else:
            print("[WARN] screeninfo not available. Assuming SLM is at (1920, 0). "
                  "Install with: pip install screeninfo")
            self.display_idx = 1
            self._display_pos = (1920, 0)
            self._display_size = self.resolution

    def _init_renderer(self):
        """Initialize the rendering backend (pygame preferred, OpenCV fallback)."""
        if _PYGAME_AVAILABLE:
            self._init_pygame()
            self._renderer = 'pygame'
        elif _CV2_AVAILABLE:
            self._init_cv2()
            self._renderer = 'cv2'
        else:
            raise ImportError(
                "No rendering backend available. Install pygame or opencv-python:\n"
                "  pip install pygame\n"
                "  pip install opencv-python"
            )

    def _init_pygame(self):
        """Initialize pygame for fullscreen borderless rendering on SLM display."""
        pygame.init()
        pygame.display.init()

        # Set the window position to the SLM display before creating it
        os.environ['SDL_VIDEO_WINDOW_POS'] = (
            f"{self._display_pos[0]},{self._display_pos[1]}"
        )

        # Create a borderless fullscreen window
        self._screen = pygame.display.set_mode(
            self._display_size,
            pygame.NOFRAME
        )
        pygame.display.set_caption("DeepCGH SLM")
        self._initialized = True

    def _init_cv2(self):
        """Initialize OpenCV as fallback renderer."""
        cv2.namedWindow(self._cv2_window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(
            self._cv2_window_name,
            self._display_pos[0],
            self._display_pos[1]
        )
        cv2.setWindowProperty(
            self._cv2_window_name,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN
        )
        self._initialized = True

    def display(self, phase_map: np.ndarray) -> None:
        """Display a phase map on the SLM.

        Args:
            phase_map: float32 array [H, W] with values in [-pi, pi].
        """
        if not self._initialized:
            raise RuntimeError("SLM driver not initialized")

        # Normalize phase to pixel values
        pixel_data = normalize_phase(phase_map, self.bit_depth)

        # Resize to SLM resolution if needed
        h, w = pixel_data.shape
        target_w, target_h = self.resolution
        if (h, w) != (target_h, target_w):
            if self.bit_depth == 8:
                pixel_data = cv2.resize(pixel_data, (target_w, target_h),
                                        interpolation=cv2.INTER_NEAREST)
            else:
                # OpenCV resize doesn't handle uint16 well via cv2.resize
                # Normalize to float, resize, then re-quantize
                norm_float = pixel_data.astype(np.float32) / (1023.0 if self.bit_depth == 10 else 255.0)
                norm_float = cv2.resize(norm_float, (target_w, target_h),
                                        interpolation=cv2.INTER_NEAREST)
                if self.bit_depth == 10:
                    pixel_data = np.clip(norm_float * 1023, 0, 1023).astype(np.uint16)
                else:
                    pixel_data = np.clip(norm_float * 255, 0, 255).astype(np.uint8)

        if self._renderer == 'pygame':
            self._display_pygame(pixel_data)
        elif self._renderer == 'cv2':
            self._display_cv2(pixel_data)

    def _display_pygame(self, pixel_data: np.ndarray) -> None:
        """Render pixel data via pygame."""
        if self.bit_depth == 8:
            # 8-bit: grayscale -> RGB surface
            rgb_data = np.stack([pixel_data] * 3, axis=-1)
            surface = pygame.surfarray.make_surface(
                np.transpose(rgb_data, (1, 0, 2))  # pygame expects (W, H, 3)
            )
        else:
            # 10-bit: pygame doesn't natively support 10-bit,
            # downscale to 8-bit for display (data is still 10-bit in the array)
            scaled = (pixel_data >> 2).astype(np.uint8)
            rgb_data = np.stack([scaled] * 3, axis=-1)
            surface = pygame.surfarray.make_surface(
                np.transpose(rgb_data, (1, 0, 2))
            )

        # Scale surface to fill the display
        surface = pygame.transform.scale(surface, self._display_size)
        self._screen.blit(surface, (0, 0))
        pygame.display.flip()

    def _display_cv2(self, pixel_data: np.ndarray) -> None:
        """Render pixel data via OpenCV."""
        if self.bit_depth == 8:
            display_data = pixel_data
        else:
            # Downscale 10-bit to 8-bit for OpenCV display
            display_data = (pixel_data >> 2).astype(np.uint8)

        cv2.imshow(self._cv2_window_name, display_data)
        cv2.waitKey(1)

    def clear(self) -> None:
        """Clear the SLM display (zero phase = mid-gray for 8-bit)."""
        if self._renderer == 'pygame' and self._screen is not None:
            self._screen.fill((0, 0, 0))
            pygame.display.flip()
        elif self._renderer == 'cv2':
            blank = np.zeros(self.resolution[::-1], dtype=np.uint8)
            cv2.imshow(self._cv2_window_name, blank)
            cv2.waitKey(1)

    def close(self) -> None:
        """Release renderer resources."""
        if self._renderer == 'pygame' and self._screen is not None:
            pygame.display.quit()
            pygame.quit()
            self._screen = None
        elif self._renderer == 'cv2':
            cv2.destroyWindow(self._cv2_window_name)
        self._initialized = False


# ===========================================================================
# SDKBackend (placeholder for vendor-specific SLM SDKs)
# ===========================================================================

class SDKBackend(SLMDriver):
    """Placeholder backend for vendor-specific SLM SDKs.

    This defines the interface for integrating with vendor SDKs such as
    Holoeye, Meadowlark, or similar. Subclass and override the methods
    to implement vendor-specific communication.

    Example subclass skeleton:
        class HoloeyeBackend(SDKBackend):
            def __init__(self, resolution, bit_depth=8, display_idx=None):
                super().__init__(resolution, bit_depth, display_idx)
                # Initialize Holoeye SDK here

            def display(self, phase_map):
                pixel_data = normalize_phase(phase_map, self.bit_depth)
                # Send pixel_data to Holoeye SLM via SDK
                ...

    Args:
        resolution: (width, height) of the SLM in pixels.
        bit_depth: Phase quantization depth (8 or 10 bits).
        display_idx: Device index (vendor-specific).
        vendor: Vendor name string (for logging/identification).
    """

    def __init__(self, resolution: Tuple[int, int], bit_depth: int = 8,
                 display_idx: Optional[int] = None, vendor: str = "unknown"):
        super().__init__(resolution, bit_depth, display_idx)
        self.vendor = vendor
        self._connected = False

    def display(self, phase_map: np.ndarray) -> None:
        """Send phase map to the SLM via vendor SDK.

        Args:
            phase_map: float32 array [H, W] with values in [-pi, pi].

        Raises:
            NotImplementedError: Must be implemented by vendor-specific subclass.
        """
        raise NotImplementedError(
            f"SDKBackend.display() not implemented for vendor '{self.vendor}'. "
            "Subclass SDKBackend and override display() with vendor-specific logic."
        )

    def clear(self) -> None:
        """Clear the SLM display via vendor SDK.

        Raises:
            NotImplementedError: Must be implemented by vendor-specific subclass.
        """
        raise NotImplementedError(
            f"SDKBackend.clear() not implemented for vendor '{self.vendor}'. "
            "Subclass SDKBackend and override clear() with vendor-specific logic."
        )

    def close(self) -> None:
        """Disconnect from the SLM via vendor SDK.

        Raises:
            NotImplementedError: Must be implemented by vendor-specific subclass.
        """
        raise NotImplementedError(
            f"SDKBackend.close() not implemented for vendor '{self.vendor}'. "
            "Subclass SDKBackend and override close() with vendor-specific logic."
        )


# ===========================================================================
# FileBackend
# ===========================================================================

class FileBackend(SLMDriver):
    """Save phase maps to image files for offline testing without an SLM.

    Supports PNG (lossless) and BMP formats. Each call to display() writes
    a new file with an incrementing index.

    Args:
        resolution: (width, height) of the SLM in pixels.
        bit_depth: Phase quantization depth (8 or 10 bits).
        display_idx: Unused (kept for interface consistency).
        output_dir: Directory to save phase map images.
        format: Image format — 'png' or 'bmp'.
    """

    def __init__(self, resolution: Tuple[int, int], bit_depth: int = 8,
                 display_idx: Optional[int] = None,
                 output_dir: str = "./slm_output",
                 format: str = "png"):
        super().__init__(resolution, bit_depth, display_idx)
        self.output_dir = output_dir
        self.format = format.lower()
        self._frame_index = 0

        if self.format not in ('png', 'bmp'):
            raise ValueError(f"Unsupported format: {self.format}. Use 'png' or 'bmp'.")

        os.makedirs(self.output_dir, exist_ok=True)

    def display(self, phase_map: np.ndarray) -> None:
        """Save phase map to an image file.

        Args:
            phase_map: float32 array [H, W] with values in [-pi, pi].
        """
        pixel_data = normalize_phase(phase_map, self.bit_depth)

        # Resize to SLM resolution if needed
        h, w = pixel_data.shape
        target_w, target_h = self.resolution
        if (h, w) != (target_h, target_w):
            if _CV2_AVAILABLE:
                if self.bit_depth == 8:
                    pixel_data = cv2.resize(pixel_data, (target_w, target_h),
                                            interpolation=cv2.INTER_NEAREST)
                else:
                    norm_float = pixel_data.astype(np.float32) / 1023.0
                    norm_float = cv2.resize(norm_float, (target_w, target_h),
                                            interpolation=cv2.INTER_NEAREST)
                    pixel_data = np.clip(norm_float * 1023, 0, 1023).astype(np.uint16)
            else:
                # Fallback: simple nearest-neighbor resize with numpy
                row_idx = np.linspace(0, h - 1, target_h).astype(int)
                col_idx = np.linspace(0, w - 1, target_w).astype(int)
                pixel_data = pixel_data[np.ix_(row_idx, col_idx)]

        filename = os.path.join(
            self.output_dir,
            f"slm_frame_{self._frame_index:06d}.{self.format}"
        )
        self._frame_index += 1

        if _CV2_AVAILABLE:
            cv2.imwrite(filename, pixel_data)
        else:
            # Fallback: save as raw numpy file
            np_path = filename.rsplit('.', 1)[0] + '.npy'
            np.save(np_path, pixel_data)
            print(f"[INFO] OpenCV not available. Saved as NumPy file: {np_path}")

    def clear(self) -> None:
        """Save a blank (zero-phase) image."""
        if self.bit_depth == 8:
            blank = np.zeros(self.resolution[::-1], dtype=np.uint8)
        else:
            blank = np.zeros(self.resolution[::-1], dtype=np.uint16)
        filename = os.path.join(
            self.output_dir,
            f"slm_frame_{self._frame_index:06d}.{self.format}"
        )
        self._frame_index += 1

        if _CV2_AVAILABLE:
            cv2.imwrite(filename, blank)
        else:
            np_path = filename.rsplit('.', 1)[0] + '.npy'
            np.save(np_path, blank)

    def close(self) -> None:
        """No resources to release for file backend."""
        pass

    @property
    def frame_count(self) -> int:
        """Number of frames saved so far."""
        return self._frame_index


# ===========================================================================
# Factory function
# ===========================================================================

def create_slm_driver(backend: str = 'direct', **kwargs) -> SLMDriver:
    """Create an SLM driver with the specified backend.

    Args:
        backend: Backend type — 'direct', 'sdk', or 'file'.
        **kwargs: Additional arguments passed to the backend constructor.
            Common kwargs:
                resolution (tuple): (width, height) of the SLM.
                bit_depth (int): 8 or 10 bits.
                display_idx (int, optional): Display/device index.

            DirectDisplayBackend-specific:
                (none beyond common kwargs)

            SDKBackend-specific:
                vendor (str): Vendor name.

            FileBackend-specific:
                output_dir (str): Output directory path.
                format (str): 'png' or 'bmp'.

    Returns:
        An SLMDriver instance.

    Raises:
        ValueError: If backend type is unknown.

    Examples:
        # Direct display on secondary monitor
        driver = create_slm_driver('direct', resolution=(1920, 1080))

        # Save to files for offline testing
        driver = create_slm_driver('file', resolution=(1920, 1080),
                                   output_dir='./test_output')

        # Vendor SDK (must be subclassed)
        driver = create_slm_driver('sdk', resolution=(1920, 1080), vendor='holoeye')
    """
    if backend == 'direct':
        return DirectDisplayBackend(**kwargs)
    elif backend == 'sdk':
        return SDKBackend(**kwargs)
    elif backend == 'file':
        return FileBackend(**kwargs)
    else:
        raise ValueError(
            f"Unknown backend: '{backend}'. Must be 'direct', 'sdk', or 'file'."
        )
