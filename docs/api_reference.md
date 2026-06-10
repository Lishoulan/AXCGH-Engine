# API Reference

Complete reference for all AXCGH-Engine public APIs.

---

## Table of Contents

- [EngineAPI (Python Engine)](#engineapi-python-engine)
- [CppDeepCGHEngine (C++ Engine)](#cppdeepcghengine-c-engine)
- [RGBHologramEngine](#rgbhologramengine)
- [SLMDriver](#slmdriver)
- [RealtimeHologramDisplay](#realtimehologramdisplay)
- [EngineConfig](#engineconfig)
- [Types and Enums](#types-and-enums)

---

## EngineAPI (Python Engine)

The primary Python engine. Uses ONNX Runtime for inference and NumPy/CuPy for IFFT post-processing.

**Module**: `deepcgh_engine.engine`

### Constructor

```python
EngineAPI()
```

Creates a new engine instance. No resources are allocated until `init()` is called.

### Methods

#### `init(model_path, config=None)`

Initialize the engine with an ONNX model and configuration.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str` | required | Path to the ONNX model file |
| `config` | `EngineConfig` | `None` (uses defaults) | Engine configuration |

**Returns**: `Status` — `Status.OK` on success.

**Raises**: No exceptions; check return value and `last_error` property.

```python
engine = EngineAPI()
status = engine.init("model.onnx", EngineConfig(height=256, width=256))
if status != Status.OK:
    print(engine.last_error)
```

#### `is_ready()`

Check if the engine is initialized and ready for inference.

**Returns**: `bool`

#### `generate_hologram(rgb, depth, benchmark=False)`

Generate a hologram phase map from RGB-D input.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rgb` | `np.ndarray` uint8 | required | RGB image, shape `[H, W, 3]` |
| `depth` | `np.ndarray` float32 | required | Depth map, shape `[H, W]` |
| `benchmark` | `bool` | `False` | Record per-stage timing |

**Returns**: `Tuple[Status, Optional[np.ndarray]]`
- On success: `(Status.OK, float32 [H, W])` — phase in [-π, π]
- On failure: `(error_status, None)`

```python
status, phase = engine.generate_hologram(rgb, depth)
# phase.shape == (256, 256), phase.dtype == float32
```

#### `generate_hologram_quantized(rgb, depth, benchmark=False)`

Generate a quantized phase map suitable for direct SLM display.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rgb` | `np.ndarray` uint8 | required | RGB image, shape `[H, W, 3]` |
| `depth` | `np.ndarray` float32 | required | Depth map, shape `[H, W]` |
| `benchmark` | `bool` | `False` | Record per-stage timing |

**Returns**: `Tuple[Status, Optional[np.ndarray]]`
- `PhaseFormat.Uint8`: uint8 `[H, W]` in [0, 255]
- `PhaseFormat.Uint16`: uint16 `[H, W]` in [0, 65535]
- `PhaseFormat.Float`: float32 `[H, W]` in [-π, π]

#### `get_perf_stats()`

Get performance statistics from benchmarked runs.

**Returns**: `Dict[str, Dict[str, float]]`

```python
stats = engine.get_perf_stats()
# {'preprocess': {'mean_ms': 2.1, 'std_ms': 0.3, 'min_ms': 1.8, 'max_ms': 3.5, 'n_runs': 10},
#  'inference': {'mean_ms': 12.5, ...},
#  'postprocess': {'mean_ms': 1.2, ...},
#  'total': {'mean_ms': 15.8, ...}}
```

#### `shutdown()`

Release all resources (model session, preprocessor, post-processor).

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `last_error` | `str` | Last error message |
| `config` | `Optional[EngineConfig]` | Current engine configuration |
| `gpu_enabled` | `bool` | Whether CuPy GPU acceleration is active |

---

## CppDeepCGHEngine (C++ Engine)

The C++ engine exposed via PyBind11. Uses FFTW3f for IFFT post-processing. Only available if the C++ extension was compiled.

**Module**: `deepcgh_engine._deepcgh_engine`

**Import**: `from deepcgh_engine import CppDeepCGHEngine`

> If the C++ extension is not compiled, `CppDeepCGHEngine` will be `None`.

### Constructor

```python
CppDeepCGHEngine()
```

### Methods

#### `init(model_path, height=256, width=256, num_planes=5, ...)`

Initialize the C++ engine with keyword arguments (not an `EngineConfig` object).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str` | required | Path to ONNX model |
| `height` | `int` | 256 | Input height |
| `width` | `int` | 256 | Input width |
| `num_planes` | `int` | 5 | Number of depth planes |
| `color_space` | `str` | `"ycbcr"` | Color space: `"ycbcr"`, `"rgb"`, `"gray"` |
| `provider` | `str` | `"cpu"` | Execution provider: `"cpu"`, `"cuda"`, `"dml"` |
| `device_id` | `int` | 0 | GPU device index |
| `enable_memory_pool` | `bool` | `True` | Reuse I/O buffers |
| `phase_format` | `str` | `"uint8"` | Output format: `"uint8"`, `"uint16"`, `"float"` |
| `wavelength` | `float` | 532e-6 | Laser wavelength in mm |
| `pixel_size` | `float` | 8e-3 | SLM pixel pitch in mm |
| `focal_length` | `float` | 200.0 | Focal length in mm |
| `plane_distance` | `float` | 10.0 | Inter-plane distance in mm |
| `quantization_bits` | `int` | 8 | SLM quantization bits |
| `int_factor` | `int` | 2 | Interleave factor |
| `norm_mean` | `float` | 0.0 | Normalization mean |
| `norm_std` | `float` | 1.0 | Normalization std |
| `intra_op_threads` | `int` | 4 | ORT intra-op threads |
| `inter_op_threads` | `int` | 1 | ORT inter-op threads |

**Raises**: `RuntimeError` on failure.

```python
engine = CppDeepCGHEngine()
engine.init("model.onnx", height=256, width=256, num_planes=5)
```

#### `generate_hologram(rgb, depth)`

Generate a hologram phase map. Uses NumPy FFT for IFFT (not FFTW3) in the Python binding layer.

| Parameter | Type | Description |
|-----------|------|-------------|
| `rgb` | `np.ndarray` uint8 | RGB image `[H, W, 3]` |
| `depth` | `np.ndarray` float32 | Depth map `[H, W]` |

**Returns**: `np.ndarray` float32 `[H, W]` — phase in [-π, π]

**Raises**: `RuntimeError` if engine not initialized or inference fails.

#### `generate_hologram_quantized(rgb, depth)`

Generate a quantized phase map for SLM display.

| Parameter | Type | Description |
|-----------|------|-------------|
| `rgb` | `np.ndarray` uint8 | RGB image `[H, W, 3]` |
| `depth` | `np.ndarray` float32 | Depth map `[H, W]` |

**Returns**: `np.ndarray` — type depends on `phase_format`:
- `"uint8"` → uint8 `[H, W]` in [0, 255]
- `"uint16"` → uint16 `[H, W]` in [0, 65535]
- `"float"` → float32 `[H, W]` in [-π, π]

#### `infer_raw(rgb, depth)`

Run preprocessing + inference only, returning raw model outputs. Bypasses IFFT post-processing.

| Parameter | Type | Description |
|-----------|------|-------------|
| `rgb` | `np.ndarray` uint8 | RGB image `[H, W, 3]` |
| `depth` | `np.ndarray` float32 | Depth map `[H, W]` |

**Returns**: `Tuple[np.ndarray, np.ndarray]` — `(amp_0, phi_0)` each of shape `[1, H, W, 1]`

#### `is_ready()`

**Returns**: `bool`

#### `shutdown()`

Release all C++ resources.

#### `last_error()`

**Returns**: `str` — last error message.

---

## RGBHologramEngine

Multi-wavelength RGB hologram engine. Wraps three `EngineAPI` instances (one per wavelength) and combines the phase maps.

**Module**: `deepcgh_engine.multi_wavelength`

### Constructor

```python
RGBHologramEngine()
```

### Methods

#### `init(model_path, config=None)`

Initialize three sub-engines with respective wavelengths.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str` | required | Path to ONNX model (shared) |
| `config` | `RGBEngineConfig` | `None` | RGB engine configuration |

**Returns**: `Status`

#### `generate_rgb_hologram(rgb, depth, benchmark=False)`

Generate RGB hologram from RGB-D input.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rgb` | `np.ndarray` uint8 | required | RGB image `[H, W, 3]` |
| `depth` | `np.ndarray` float32 | required | Depth map `[H, W]` |
| `benchmark` | `bool` | `False` | Record timing |

**Returns**: `Tuple[Status, Optional[Dict[str, np.ndarray]]]`

The result dictionary contains:

| Key | Type | Description |
|-----|------|-------------|
| `phase_r` | float32 `[H, W]` | Red channel phase (633 nm) |
| `phase_g` | float32 `[H, W]` | Green channel phase (532 nm) |
| `phase_b` | float32 `[H, W]` | Blue channel phase (450 nm) |
| `phase_combined` | float32 `[H, W]` | Combined phase map |

#### `generate_rgb_hologram_quantized(rgb, depth, benchmark=False)`

Same as `generate_rgb_hologram` but returns quantized phase maps.

**Returns**: `Tuple[Status, Optional[Dict[str, np.ndarray]]]` — all values quantized per `phase_format`.

#### `is_ready()`

**Returns**: `bool`

#### `shutdown()`

Release all sub-engine resources.

#### `get_perf_stats()`

**Returns**: `Dict[str, Dict[str, float]]` — per-channel and combine timing.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `last_error` | `str` | Last error message |
| `config` | `Optional[RGBEngineConfig]` | Current configuration |
| `engines` | `Dict[str, Any]` | Per-channel engine instances (`'r'`, `'g'`, `'b'`) |

### RGBEngineConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `base_config` | `EngineConfig` | `EngineConfig()` | Base engine config (wavelength overridden per channel) |
| `wavelength_r` | `float` | 633e-6 | Red wavelength in mm (633 nm) |
| `wavelength_g` | `float` | 532e-6 | Green wavelength in mm (532 nm) |
| `wavelength_b` | `float` | 450e-6 | Blue wavelength in mm (450 nm) |
| `combine_mode` | `CombineMode` | `TimeDivision` | Phase combination mode |
| `spatial_pattern` | `SpatialPattern` | `Checkerboard` | Spatial multiplex pattern |
| `use_cpp_engine` | `bool` | `False` | Use C++ engine for sub-engines |

### CombineMode

| Value | Description |
|-------|-------------|
| `TimeDivision` | Average of R/G/B phases (for sequential display at high speed) |
| `SpatialMultiplex` | Interleave R/G/B pixels spatially (checkerboard or stripe) |

### SpatialPattern

| Value | Description |
|-------|-------------|
| `Checkerboard` | 3-color checkerboard pattern |
| `Stripe` | Row-based stripe pattern |

### Standard Wavelengths

| Constant | Value | Description |
|----------|-------|-------------|
| `WAVELENGTH_R` | 633e-6 mm | HeNe red laser |
| `WAVELENGTH_G` | 532e-6 mm | Frequency-doubled Nd:YAG green |
| `WAVELENGTH_B` | 450e-6 mm | Blue laser diode |

---

## SLMDriver

Abstract base class and concrete implementations for driving Spatial Light Modulators.

**Module**: `deepcgh_engine.slm_driver`

### SLMDriver (Abstract Base)

| Method | Description |
|--------|-------------|
| `display(phase_map)` | Display a phase map on the SLM. Input: float32 `[H, W]` in [-π, π] |
| `clear()` | Clear the SLM display (zero phase) |
| `close()` | Release resources and close connection |

**Constructor parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `resolution` | `Tuple[int, int]` | required | (width, height) of SLM in pixels |
| `bit_depth` | `int` | 8 | Phase quantization depth (8 or 10) |
| `display_idx` | `Optional[int]` | `None` | Display/device index |

### DirectDisplayBackend

Renders phase maps on a secondary display (SLM connected via HDMI/DVI) using pygame or OpenCV.

**Additional constructor parameters**: None beyond `SLMDriver` base.

```python
slm = DirectDisplayBackend(resolution=(1920, 1080), bit_depth=8, display_idx=1)
slm.display(phase_map)
slm.clear()
slm.close()
```

### SDKBackend

Placeholder for vendor-specific SLM SDKs (Holoeye, Meadowlark, etc.). Must be subclassed.

**Additional constructor parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vendor` | `str` | `"unknown"` | Vendor name for identification |

All methods raise `NotImplementedError`. Subclass and override:

```python
class HoloeyeBackend(SDKBackend):
    def __init__(self, resolution, bit_depth=8):
        super().__init__(resolution, bit_depth, vendor="holoeye")
        # Initialize Holoeye SDK

    def display(self, phase_map):
        pixel_data = normalize_phase(phase_map, self.bit_depth)
        # Send to Holoeye SLM via SDK
```

### FileBackend

Saves phase maps to image files for offline testing.

**Additional constructor parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | `str` | `"./slm_output"` | Output directory |
| `format` | `str` | `"png"` | Image format: `"png"` or `"bmp"` |

```python
slm = FileBackend(resolution=(1920, 1080), output_dir="./frames", format="png")
slm.display(phase_map)
print(f"Saved {slm.frame_count} frames")
slm.close()
```

**Additional property**: `frame_count` → `int` — number of frames saved.

### `create_slm_driver(backend, **kwargs)`

Factory function to create an SLM driver.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `str` | `"direct"` | Backend type: `"direct"`, `"sdk"`, `"file"` |
| `**kwargs` | — | — | Passed to the backend constructor |

```python
# Direct display
slm = create_slm_driver('direct', resolution=(1920, 1080))

# File output
slm = create_slm_driver('file', resolution=(256, 256), output_dir='./out')

# Vendor SDK
slm = create_slm_driver('sdk', resolution=(1920, 1080), vendor='holoeye')
```

### Utility Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `normalize_phase_to_uint8` | `(phase_map) → ndarray` | [-π, π] → [0, 255] uint8 |
| `normalize_phase_to_uint16_10bit` | `(phase_map) → ndarray` | [-π, π] → [0, 1023] uint16 |
| `normalize_phase` | `(phase_map, bit_depth) → ndarray` | Generic normalization for bit_depth 8 or 10 |

---

## RealtimeHologramDisplay

Real-time hologram preview and SLM display system with threaded pipeline.

**Module**: `deepcgh_engine.realtime`

### Constructor

```python
RealtimeHologramDisplay(config=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `RealtimeConfig` | `None` (uses defaults) | Display configuration |

### Methods

#### `run()`

Start the real-time display loop. **Blocks until quit** (press `q`).

**Keyboard controls**:
- `q` — Quit
- `s` — Save screenshot
- `p` — Pause / resume

#### `stop()`

Signal the display loop to stop (non-blocking).

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `is_running` | `bool` | Whether the display loop is active |
| `perf_summary` | `Dict[str, Any]` | Performance summary (FPS, timing) |
| `current_phase` | `Optional[np.ndarray]` | Most recently generated phase map |

### RealtimeConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `camera_type` | `CameraType` | `CameraType.TEST` | Camera source |
| `device_id` | `int` | 0 | Camera device index |
| `target_fps` | `float` | 30.0 | Target frame rate |
| `max_queue_size` | `int` | 2 | Frame queue depth |
| `model_path` | `str` | `"models/deepcgh_unet.onnx"` | ONNX model path |
| `engine_config` | `EngineConfig` | `EngineConfig()` | Engine configuration |
| `slm_backend` | `SLABackend` | `SLABackend.OPENCV` | SLM display backend |
| `slm_device_id` | `int` | 0 | SLM device index |
| `preview_width` | `int` | 640 | Preview window width |
| `preview_height` | `int` | 480 | Preview window height |
| `show_preview` | `bool` | `True` | Show OpenCV preview window |
| `auto_adjust_resolution` | `bool` | `True` | Auto-adjust resolution to maintain FPS |
| `min_resolution` | `int` | 64 | Minimum resolution when auto-adjusting |
| `resolution_step` | `int` | 64 | Step size for resolution changes |
| `screenshot_dir` | `str` | `"screenshots"` | Directory for saved screenshots |

### CameraType

| Value | Description |
|-------|-------------|
| `REALSENSE` | Intel RealSense (requires `pyrealsense2`) |
| `KINECT` | Azure Kinect (requires `pyk4a`) |
| `TEST` | Synthetic test pattern (no hardware needed) |

### SLABackend

| Value | Description |
|-------|-------------|
| `NONE` | No physical SLM — preview only |
| `SDL` | SDL-based direct display |
| `OPENCV` | OpenCV window (development) |
| `SDK` | Vendor SDK |

### Camera Sources

#### RealSenseCamera

```python
camera = RealSenseCamera(device_id=0, width=640, height=480, fps=30)
camera.open()
rgb, depth = camera.capture()  # rgb: [H,W,3] uint8, depth: [H,W] float32
camera.close()
```

#### KinectCamera

```python
camera = KinectCamera(device_id=0)
camera.open()
rgb, depth = camera.capture()
camera.close()
```

#### TestPatternSource

```python
source = TestPatternSource(width=640, height=480, fps=30)
source.open()
rgb, depth = source.capture()  # Animated color bars + moving depth plane
source.close()
```

---

## EngineConfig

**Module**: `deepcgh_engine.engine`

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `height` | `int` | 256 | Input frame height in pixels |
| `width` | `int` | 256 | Input frame width in pixels |
| `num_planes` | `int` | 5 | Number of depth planes (channels) |
| `color_space` | `ColorSpace` | `YCbCr` | Color-space conversion mode |
| `norm_mean` | `float` | 0.0 | Per-channel normalization mean |
| `norm_std` | `float` | 1.0 | Per-channel normalization std |
| `provider` | `ExecutionProvider` | `CPU` | ONNX execution provider |
| `device_id` | `int` | 0 | GPU device index |
| `enable_memory_pool` | `bool` | `True` | Reuse I/O buffers across frames |
| `intra_op_threads` | `int` | 4 | ORT intra-op thread count |
| `inter_op_threads` | `int` | 1 | ORT inter-op thread count |
| `phase_format` | `PhaseFormat` | `Uint8` | SLM output pixel format |
| `wavelength` | `float` | 532e-6 | Laser wavelength in mm (532 nm default) |
| `pixel_size` | `float` | 8e-3 | SLM pixel pitch in mm |
| `focal_length` | `float` | 200.0 | Focal length in mm |
| `plane_distance` | `float` | 10.0 | Inter-plane distance in mm |
| `quantization_bits` | `int` | 8 | SLM quantization levels (2^bits) |
| `int_factor` | `int` | 2 | Space-to-depth block size |
| `use_cupy` | `bool` | `True` | Auto-detect CuPy for GPU IFFT |

### Methods

#### `validate()`

Validate configuration values. Raises `AssertionError` on failure.

**Constraints**:
- `height > 0` and `width > 0`
- `num_planes > 0`
- `height % int_factor == 0`
- `width % int_factor == 0`

---

## Types and Enums

### PhaseFormat

**Module**: `deepcgh_engine.engine`

| Value | String | Description |
|-------|--------|-------------|
| `Uint8` | `"uint8"` | 8-bit grayscale [0, 255] — common for nematic SLMs |
| `Uint16` | `"uint16"` | 16-bit grayscale [0, 65535] — high-precision SLMs |
| `Float` | `"float"` | 32-bit float [-π, π] — raw phase (debug/export) |

### ColorSpace

**Module**: `deepcgh_engine.engine`

| Value | String | Description |
|-------|--------|-------------|
| `RGB` | `"rgb"` | Keep raw RGB channels (3-channel input) |
| `YCbCr` | `"ycbcr"` | Convert to YCbCr, use luminance (1-channel) |
| `Gray` | `"gray"` | Convert to grayscale (1-channel, simplest) |

### ExecutionProvider

**Module**: `deepcgh_engine.engine`

| Value | String | Description |
|-------|--------|-------------|
| `CPU` | `"cpu"` | CPU-only execution (portable) |
| `CUDA` | `"cuda"` | NVIDIA CUDA GPU acceleration |
| `DML` | `"dml"` | DirectML (Windows GPU, vendor-agnostic) |

### Status

**Module**: `deepcgh_engine.engine`

| Value | Code | Description |
|-------|------|-------------|
| `OK` | 0 | Success |
| `NOT_INITIALIZED` | 1 | Engine not initialized |
| `MODEL_LOAD_FAILED` | 2 | Model loading failed |
| `INVALID_INPUT` | 3 | Invalid input data |
| `INFERENCE_FAILED` | 4 | Inference execution failed |
| `SIZE_MISMATCH` | 5 | Input size mismatch |
| `PROVIDER_UNAVAIL` | 6 | Execution provider unavailable |
| `ALLOCATION_FAILED` | 7 | Memory allocation failed |

### CombineMode

**Module**: `deepcgh_engine.multi_wavelength`

| Value | String | Description |
|-------|--------|-------------|
| `TimeDivision` | `"time_division"` | Sequential display (average for preview) |
| `SpatialMultiplex` | `"spatial_multiplex"` | Spatial interleaving (checkerboard/stripe) |

### SpatialPattern

**Module**: `deepcgh_engine.multi_wavelength`

| Value | String | Description |
|-------|--------|-------------|
| `Checkerboard` | `"checkerboard"` | 3-color checkerboard |
| `Stripe` | `"stripe"` | Row-based stripe pattern |

### CameraType

**Module**: `deepcgh_engine.realtime`

| Value | String | Description |
|-------|--------|-------------|
| `REALSENSE` | `"realsense"` | Intel RealSense camera |
| `KINECT` | `"kinect"` | Azure Kinect camera |
| `TEST` | `"test"` | Synthetic test pattern |

### SLABackend

**Module**: `deepcgh_engine.realtime`

| Value | String | Description |
|-------|--------|-------------|
| `NONE` | `"none"` | No physical SLM |
| `SDL` | `"sdl"` | SDL-based direct display |
| `OPENCV` | `"opencv"` | OpenCV window |
| `SDK` | `"sdk"` | Vendor SDK |
