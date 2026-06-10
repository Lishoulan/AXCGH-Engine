"""
deepcgh_engine/realtime.py — Real-time hologram preview and SLM display system.

Provides a threaded pipeline:
  Camera Capture (thread) -> Hologram Generation (thread) -> SLM Display + Preview (main)

Camera sources:
  - RealSenseCamera  (pyrealsense2)
  - KinectCamera     (pyk4a)
  - TestPatternSource (synthetic data for development)

Usage:
    from deepcgh_engine import RealtimeHologramDisplay, RealtimeConfig

    config = RealtimeConfig(
        camera_type="test",
        model_path="models/deepcgh_unet.onnx",
        target_fps=30,
    )
    display = RealtimeHologramDisplay(config)
    display.run()
"""

import enum
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

import numpy as np

from .engine import EngineAPI, EngineConfig, Status, PhaseFormat, ExecutionProvider

# Optional camera SDKs — gracefully degrade if not installed
try:
    import pyrealsense2 as rs
    _REALSENSE_AVAILABLE = True
except ImportError:
    _REALSENSE_AVAILABLE = False

try:
    import pyk4a
    from pyk4a import Config as K4AConfig, PyK4A
    _K4A_AVAILABLE = True
except ImportError:
    _K4A_AVAILABLE = False

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


# ===========================================================================
# Configuration
# ===========================================================================

class CameraType(enum.Enum):
    """Supported camera input sources."""
    REALSENSE = "realsense"
    KINECT = "kinect"
    TEST = "test"


class SLABackend(enum.Enum):
    """SLM display backend."""
    NONE = "none"           # No physical SLM — preview only
    SDL = "sdl"             # SDL-based direct display
    OPENCV = "opencv"       # OpenCV window (development)
    SDK = "sdk"             # Vendor SDK (e.g. Meadowlark, Holoeye)


@dataclass
class RealtimeConfig:
    """Configuration for RealtimeHologramDisplay."""
    # Camera
    camera_type: CameraType = CameraType.TEST
    device_id: int = 0

    # Performance
    target_fps: float = 30.0
    max_queue_size: int = 2           # Frame queue depth between threads

    # Engine
    model_path: str = "models/deepcgh_unet.onnx"
    engine_config: EngineConfig = field(default_factory=EngineConfig)

    # SLM
    slm_backend: SLABackend = SLABackend.OPENCV
    slm_device_id: int = 0

    # Preview
    preview_width: int = 640
    preview_height: int = 480
    show_preview: bool = True

    # Auto-adjust
    auto_adjust_resolution: bool = True
    min_resolution: int = 64          # Minimum model input dimension
    resolution_step: int = 64         # Step size when down-scaling

    # Screenshot
    screenshot_dir: str = "screenshots"


# ===========================================================================
# Camera Sources
# ===========================================================================

class RealSenseCamera:
    """Intel RealSense RGB-D camera source using pyrealsense2."""

    def __init__(self, device_id: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        if not _REALSENSE_AVAILABLE:
            raise RuntimeError("pyrealsense2 is not installed: pip install pyrealsense2")

        self._pipeline = None
        self._align = None
        self._device_id = device_id
        self._width = width
        self._height = height
        self._fps = fps

    def open(self) -> bool:
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            if self._device_id >= len(devices):
                print(f"[RealSense] Device {self._device_id} not found "
                      f"({len(devices)} device(s) available)")
                return False

            self._pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(devices[self._device_id].get_info(rs.camera_info.serial_number))
            config.enable_stream(rs.stream.color, self._width, self._height, rs.format.bgr8, self._fps)
            config.enable_stream(rs.stream.depth, self._width, self._height, rs.format.z16, self._fps)

            self._pipeline.start(config)
            align_to = rs.stream.color
            self._align = rs.align(align_to)
            return True
        except Exception as e:
            print(f"[RealSense] Failed to open: {e}")
            return False

    def capture(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Capture a frame. Returns (rgb [H,W,3] uint8, depth [H,W] float32) or None."""
        if self._pipeline is None:
            return None
        try:
            frames = self._pipeline.wait_for_frames()
            aligned = self._align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            rgb = np.asanyarray(color_frame.get_data())  # BGR
            rgb = rgb[:, :, ::-1].copy()  # BGR -> RGB

            depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)
            depth = depth_raw * depth_frame.get_units()  # Convert to meters

            return rgb, depth
        except Exception as e:
            print(f"[RealSense] Capture error: {e}")
            return None

    def close(self):
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None


class KinectCamera:
    """Azure Kinect RGB-D camera source using pyk4a."""

    def __init__(self, device_id: int = 0):
        if not _K4A_AVAILABLE:
            raise RuntimeError("pyk4a is not installed: pip install pyk4a")

        self._device = None
        self._device_id = device_id

    def open(self) -> bool:
        try:
            config = K4AConfig(
                color_resolution=pyk4a.ColorResolution.RES_720P,
                depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            )
            self._device = PyK4A(config, device_id=self._device_id)
            self._device.start()
            return True
        except Exception as e:
            print(f"[Kinect] Failed to open: {e}")
            return False

    def capture(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Capture a frame. Returns (rgb [H,W,3] uint8, depth [H,W] float32) or None."""
        if self._device is None:
            return None
        try:
            capture = self._device.get_capture()
            if capture.color is None or capture.depth is None:
                return None

            rgb = capture.color[:, :, :3].copy()  # Drop alpha if BGRA
            if capture.color.shape[2] == 4:
                rgb = capture.color[:, :, 2::-1].copy()  # BGRA -> RGB
            else:
                rgb = capture.color[:, :, ::-1].copy()

            # Transform depth to color camera space
            depth = capture.transformed_depth.astype(np.float32) * 0.001  # mm -> m

            return rgb, depth
        except Exception as e:
            print(f"[Kinect] Capture error: {e}")
            return None

    def close(self):
        if self._device is not None:
            try:
                self._device.stop()
            except Exception:
                pass
            self._device = None


class TestPatternSource:
    """Synthetic RGB-D test pattern generator for development."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self._width = width
        self._height = height
        self._fps = fps
        self._frame_count = 0
        self._last_time = time.perf_counter()

    def open(self) -> bool:
        self._frame_count = 0
        self._last_time = time.perf_counter()
        return True

    def capture(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Generate a synthetic RGB-D frame."""
        # Throttle to target FPS
        now = time.perf_counter()
        elapsed = now - self._last_time
        frame_interval = 1.0 / self._fps if self._fps > 0 else 0
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)
        self._last_time = time.perf_counter()

        H, W = self._height, self._width
        t = self._frame_count * 0.05

        # Animated color bars
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        bar_width = max(W // 8, 1)
        for i in range(8):
            x0 = i * bar_width
            x1 = min(x0 + bar_width, W)
            phase = (i + self._frame_count * 0.02) % 1.0
            rgb[:, x0:x1, 0] = int(128 + 127 * np.sin(2 * np.pi * phase))
            rgb[:, x0:x1, 1] = int(128 + 127 * np.sin(2 * np.pi * phase + 2.094))
            rgb[:, x0:x1, 2] = int(128 + 127 * np.sin(2 * np.pi * phase + 4.189))

        # Moving depth plane
        yy, xx = np.mgrid[0:H, 0:W]
        cx = W / 2 + W / 4 * np.sin(t)
        cy = H / 2 + H / 4 * np.cos(t * 0.7)
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        depth = (0.5 + 0.5 * np.sin(dist * 0.05 - t)).astype(np.float32)

        self._frame_count += 1
        return rgb, depth

    def close(self):
        pass


# ===========================================================================
# Performance Monitor
# ===========================================================================

class _PerfMonitor:
    """Track per-stage timing and compute FPS breakdown."""

    def __init__(self, window: int = 60):
        self._window = window
        self._capture_times: list = []
        self._inference_times: list = []
        self._display_times: list = []
        self._total_times: list = []
        self._frame_timestamps: list = []

    def record_capture(self, ms: float):
        self._capture_times.append(ms)
        if len(self._capture_times) > self._window:
            self._capture_times.pop(0)

    def record_inference(self, ms: float):
        self._inference_times.append(ms)
        if len(self._inference_times) > self._window:
            self._inference_times.pop(0)

    def record_display(self, ms: float):
        self._display_times.append(ms)
        if len(self._display_times) > self._window:
            self._display_times.pop(0)

    def record_total(self, ms: float):
        self._total_times.append(ms)
        if len(self._total_times) > self._window:
            self._total_times.pop(0)

    def record_frame(self):
        self._frame_timestamps.append(time.perf_counter())
        if len(self._frame_timestamps) > self._window:
            self._frame_timestamps.pop(0)

    def get_fps(self) -> float:
        if len(self._frame_timestamps) < 2:
            return 0.0
        dt = self._frame_timestamps[-1] - self._frame_timestamps[0]
        if dt <= 0:
            return 0.0
        return (len(self._frame_timestamps) - 1) / dt

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        result = {}
        for name, times in [
            ("capture", self._capture_times),
            ("inference", self._inference_times),
            ("display", self._display_times),
            ("total", self._total_times),
        ]:
            if times:
                arr = np.array(times)
                result[name] = {
                    "mean_ms": float(np.mean(arr)),
                    "max_ms": float(np.max(arr)),
                    "min_ms": float(np.min(arr)),
                }
        result["fps"] = self.get_fps()
        return result


# ===========================================================================
# RealtimeHologramDisplay
# ===========================================================================

class RealtimeHologramDisplay:
    """
    Real-time hologram preview and SLM display system.

    Threading model:
      - Capture thread: grabs frames from the camera source
      - Process thread: runs hologram generation via EngineAPI
      - Main thread: displays preview with FPS overlay and sends to SLM

    Keyboard controls (when preview window is active):
      q — Quit
      s — Save screenshot
      p — Pause / resume
    """

    def __init__(self, config: Optional[RealtimeConfig] = None):
        if config is None:
            config = RealtimeConfig()
        self._config = config

        # Camera source
        self._camera = self._create_camera(config)

        # Engine
        self._engine: Optional[EngineAPI] = None

        # Threading
        self._capture_queue: queue.Queue = queue.Queue(maxsize=config.max_queue_size)
        self._result_queue: queue.Queue = queue.Queue(maxsize=config.max_queue_size)
        self._stop_event = threading.Event()
        self._paused = threading.Event()
        self._paused.set()  # Not paused initially

        # State
        self._running = False
        self._current_phase: Optional[np.ndarray] = None
        self._current_rgb: Optional[np.ndarray] = None
        self._current_depth: Optional[np.ndarray] = None

        # Performance
        self._perf = _PerfMonitor()

        # Auto-adjust
        self._effective_resolution = (
            config.engine_config.height, config.engine_config.width
        )

    @staticmethod
    def _create_camera(config: RealtimeConfig):
        if config.camera_type == CameraType.REALSENSE:
            return RealSenseCamera(device_id=config.device_id)
        elif config.camera_type == CameraType.KINECT:
            return KinectCamera(device_id=config.device_id)
        elif config.camera_type == CameraType.TEST:
            return TestPatternSource(fps=int(config.target_fps))
        else:
            raise ValueError(f"Unknown camera type: {config.camera_type}")

    def _init_engine(self) -> Status:
        """Initialize the DeepCGH engine."""
        self._engine = EngineAPI()
        return self._engine.init(self._config.model_path, self._config.engine_config)

    # -----------------------------------------------------------------------
    # Resize helper
    # -----------------------------------------------------------------------

    @staticmethod
    def _resize_frame(rgb: np.ndarray, depth: np.ndarray,
                      target_h: int, target_w: int) -> Tuple[np.ndarray, np.ndarray]:
        """Resize RGB-D frame to model input resolution."""
        if rgb.shape[0] == target_h and rgb.shape[1] == target_w:
            return rgb, depth

        if _CV2_AVAILABLE:
            rgb_resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            depth_resized = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            # Fallback to NumPy (nearest-neighbor via simple indexing)
            h_in, w_in = rgb.shape[:2]
            row_idx = (np.arange(target_h) * h_in // target_h).astype(int)
            col_idx = (np.arange(target_w) * w_in // target_w).astype(int)
            rgb_resized = rgb[np.ix_(row_idx, col_idx)]
            depth_resized = depth[np.ix_(row_idx, col_idx)]

        return rgb_resized, depth_resized

    # -----------------------------------------------------------------------
    # Capture thread
    # -----------------------------------------------------------------------

    def _capture_loop(self):
        """Capture frames from the camera and push to queue."""
        while not self._stop_event.is_set():
            if not self._paused.is_set():
                t0 = time.perf_counter()
                frame = self._camera.capture()
                t1 = time.perf_counter()
                if frame is not None:
                    self._perf.record_capture((t1 - t0) * 1000)
                    try:
                        self._capture_queue.put_nowait(frame)
                    except queue.Full:
                        # Drop oldest frame
                        try:
                            self._capture_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self._capture_queue.put_nowait(frame)
                        except queue.Full:
                            pass
            else:
                time.sleep(0.001)

    # -----------------------------------------------------------------------
    # Process thread
    # -----------------------------------------------------------------------

    def _process_loop(self):
        """Pull frames from capture queue, generate holograms, push results."""
        while not self._stop_event.is_set():
            try:
                rgb, depth = self._capture_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            t0 = time.perf_counter()

            # Resize to current effective resolution
            h, w = self._effective_resolution
            rgb_resized, depth_resized = self._resize_frame(rgb, depth, h, w)

            # Generate hologram
            if self._engine is not None and self._engine.is_ready():
                status, phase = self._engine.generate_hologram(rgb_resized, depth_resized)
                if status != Status.OK:
                    continue
            else:
                # Fallback: generate a dummy phase map
                phase = np.random.rand(h, w).astype(np.float32) * 2 * np.pi - np.pi

            t1 = time.perf_counter()
            self._perf.record_inference((t1 - t0) * 1000)

            result = {
                "phase": phase,
                "rgb": rgb,
                "depth": depth,
                "rgb_resized": rgb_resized,
            }

            try:
                self._result_queue.put_nowait(result)
            except queue.Full:
                try:
                    self._result_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._result_queue.put_nowait(result)
                except queue.Full:
                    pass

    # -----------------------------------------------------------------------
    # SLM display
    # -----------------------------------------------------------------------

    def _display_slm(self, phase: np.ndarray):
        """Send phase map to the SLM backend."""
        if self._config.slm_backend == SLABackend.NONE:
            return

        # Quantize phase to 8-bit for display
        normalized = (phase + np.pi) / (2 * np.pi)
        phase_u8 = np.clip(normalized * 255, 0, 255).astype(np.uint8)

        if self._config.slm_backend == SLABackend.OPENCV:
            if _CV2_AVAILABLE:
                cv2.imshow("SLM Output", phase_u8)
            # else silently skip

        elif self._config.slm_backend == SLABackend.SDL:
            # SDL direct display — placeholder for vendor-specific integration
            pass

        elif self._config.slm_backend == SLABackend.SDK:
            # Vendor SDK integration — placeholder
            pass

    # -----------------------------------------------------------------------
    # Preview rendering
    # -----------------------------------------------------------------------

    def _render_preview(self, result: Dict[str, Any]) -> Optional[np.ndarray]:
        """Compose a preview image with FPS overlay."""
        if not _CV2_AVAILABLE:
            return None

        pw, ph = self._config.preview_width, self._config.preview_height

        # Phase map visualization (colormap)
        phase = result["phase"]
        normalized = (phase + np.pi) / (2 * np.pi)
        phase_u8 = np.clip(normalized * 255, 0, 255).astype(np.uint8)
        phase_color = cv2.applyColorMap(phase_u8, cv2.COLORMAP_JET)

        # Resize to preview size
        phase_color = cv2.resize(phase_color, (pw, ph))

        # FPS overlay
        fps = self._perf.get_fps()
        summary = self._perf.get_summary()
        capture_ms = summary.get("capture", {}).get("mean_ms", 0)
        infer_ms = summary.get("inference", {}).get("mean_ms", 0)
        display_ms = summary.get("display", {}).get("mean_ms", 0)

        lines = [
            f"FPS: {fps:.1f}",
            f"Capture: {capture_ms:.1f}ms",
            f"Inference: {infer_ms:.1f}ms",
            f"Display: {display_ms:.1f}ms",
            f"Resolution: {self._effective_resolution[0]}x{self._effective_resolution[1]}",
        ]

        y_offset = 25
        for line in lines:
            cv2.putText(phase_color, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            # Shadow for readability
            cv2.putText(phase_color, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            y_offset += 22

        return phase_color

    # -----------------------------------------------------------------------
    # Screenshot
    # -----------------------------------------------------------------------

    def _save_screenshot(self, result: Dict[str, Any]):
        """Save current frame and phase map to disk."""
        if not _CV2_AVAILABLE:
            print("[Screenshot] OpenCV not available, cannot save screenshot")
            return

        os.makedirs(self._config.screenshot_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save RGB
        rgb_path = os.path.join(self._config.screenshot_dir, f"rgb_{timestamp}.png")
        cv2.imwrite(rgb_path, cv2.cvtColor(result["rgb"], cv2.COLOR_RGB2BGR))

        # Save phase
        phase = result["phase"]
        normalized = (phase + np.pi) / (2 * np.pi)
        phase_u8 = np.clip(normalized * 255, 0, 255).astype(np.uint8)
        phase_path = os.path.join(self._config.screenshot_dir, f"phase_{timestamp}.png")
        cv2.imwrite(phase_path, phase_u8)

        # Save depth
        depth = result["depth"]
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth_u8 = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            depth_u8 = np.zeros_like(depth, dtype=np.uint8)
        depth_path = os.path.join(self._config.screenshot_dir, f"depth_{timestamp}.png")
        cv2.imwrite(depth_path, depth_u8)

        print(f"[Screenshot] Saved to {self._config.screenshot_dir}/")

    # -----------------------------------------------------------------------
    # Auto-resolution adjustment
    # -----------------------------------------------------------------------

    def _check_and_adjust_resolution(self):
        """If actual FPS is below target, reduce resolution; if well above, increase."""
        if not self._config.auto_adjust_resolution:
            return

        fps = self._perf.get_fps()
        if fps <= 0:
            return

        h, w = self._effective_resolution
        step = self._config.resolution_step
        min_res = self._config.min_resolution
        max_h = self._config.engine_config.height
        max_w = self._config.engine_config.width

        if fps < self._config.target_fps * 0.8:
            # Below 80% of target — reduce resolution
            new_h = max(h - step, min_res)
            new_w = max(w - step, min_res)
            if new_h != h or new_w != w:
                self._effective_resolution = (new_h, new_w)
                print(f"[AutoAdjust] FPS={fps:.1f} < target={self._config.target_fps:.1f}, "
                      f"reducing resolution to {new_h}x{new_w}")
        elif fps > self._config.target_fps * 1.2:
            # Above 120% of target — increase resolution
            new_h = min(h + step, max_h)
            new_w = min(w + step, max_w)
            if new_h != h or new_w != w:
                self._effective_resolution = (new_h, new_w)
                print(f"[AutoAdjust] FPS={fps:.1f} > target={self._config.target_fps:.1f}, "
                      f"increasing resolution to {new_h}x{new_w}")

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def run(self):
        """Start the real-time display loop. Blocks until quit."""
        # Open camera
        if not self._camera.open():
            print("[ERROR] Failed to open camera source")
            return

        # Init engine
        status = self._init_engine()
        if status != Status.OK:
            print(f"[ERROR] Engine initialization failed (status={status})")
            self._camera.close()
            return

        self._running = True
        self._stop_event.clear()

        # Start threads
        capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        process_thread = threading.Thread(target=self._process_loop, daemon=True)
        capture_thread.start()
        process_thread.start()

        # Auto-adjust check interval
        last_adjust_time = time.perf_counter()
        adjust_interval = 2.0  # Check every 2 seconds

        print("[RealtimeHologramDisplay] Running. Controls: q=quit, s=screenshot, p=pause")

        try:
            while not self._stop_event.is_set():
                t0 = time.perf_counter()

                # Get latest result (non-blocking)
                result = None
                try:
                    while True:
                        result = self._result_queue.get_nowait()
                except queue.Empty:
                    pass

                if result is not None:
                    self._current_phase = result["phase"]
                    self._current_rgb = result["rgb"]
                    self._current_depth = result["depth"]

                    # Display on SLM
                    self._display_slm(result["phase"])

                    # Show preview
                    if self._config.show_preview and _CV2_AVAILABLE:
                        preview = self._render_preview(result)
                        if preview is not None:
                            cv2.imshow("DeepCGH Preview", preview)

                # Handle keyboard input
                if _CV2_AVAILABLE and self._config.show_preview:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s') and result is not None:
                        self._save_screenshot(result)
                    elif key == ord('p'):
                        if self._paused.is_set():
                            self._paused.clear()
                            print("[Paused]")
                        else:
                            self._paused.set()
                            print("[Resumed]")
                else:
                    # No OpenCV preview — check stop event with a small sleep
                    time.sleep(0.001)

                t1 = time.perf_counter()
                self._perf.record_display((t1 - t0) * 1000)
                self._perf.record_frame()

                # Auto-adjust resolution periodically
                now = time.perf_counter()
                if now - last_adjust_time > adjust_interval:
                    self._check_and_adjust_resolution()
                    last_adjust_time = now

        except KeyboardInterrupt:
            print("\n[Interrupted]")
        finally:
            self._stop_event.set()
            self._running = False

            # Wait for threads
            capture_thread.join(timeout=2.0)
            process_thread.join(timeout=2.0)

            # Cleanup
            self._camera.close()
            if self._engine is not None:
                self._engine.shutdown()
            if _CV2_AVAILABLE:
                cv2.destroyAllWindows()

            # Print final stats
            summary = self._perf.get_summary()
            print("\n[Performance Summary]")
            for stage, vals in summary.items():
                if isinstance(vals, dict):
                    print(f"  {stage}: mean={vals.get('mean_ms', 0):.1f}ms, "
                          f"max={vals.get('max_ms', 0):.1f}ms")
                else:
                    print(f"  {stage}: {vals:.1f}")

    def stop(self):
        """Signal the display loop to stop."""
        self._stop_event.set()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def perf_summary(self) -> Dict[str, Any]:
        return self._perf.get_summary()

    @property
    def current_phase(self) -> Optional[np.ndarray]:
        return self._current_phase
