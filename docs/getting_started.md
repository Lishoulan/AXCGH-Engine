---
layout: default
title: Getting Started
---

# Getting Started with AXCGH-Engine

This guide walks you through installing AXCGH-Engine, generating your first hologram, understanding the pipeline, and troubleshooting common issues.

---

## Installation

### Prerequisites

- Python 3.8 or later
- pip package manager

### Option 1: Install via pip (Pure Python Engine)

The Python engine requires only `onnxruntime` and `numpy`. No C++ compilation is needed.

```bash
pip install onnxruntime numpy pillow
```

Then add the `DeepCGHEngine` package to your Python path:

```bash
cd F:\deepcgh\DeepCGHEngine
pip install .
```

This installs the `deepcgh-engine` package with the pure Python engine. The C++ extension is **not** built by default.

### Option 2: Install with C++ Engine

If you want the C++ engine (FFTW3-based IFFT), you need additional dependencies:

1. **ONNX Runtime C++ SDK** — download from [onnxruntime.ai](https://onnxruntime.ai/) and extract to `deps/onnxruntime/`
2. **FFTW3** — download the Windows DLL from [fftw.org](http://www.fftw.org/) and place in `deps/fftw/`
3. **pybind11** — `pip install pybind11`
4. **CMake** ≥ 3.15 and a C++17 compiler (MinGW-w64 or MSVC)

```bash
# Set environment variable for ONNX Runtime
set ONNXRUNTIME_ROOT=deps/onnxruntime

# Install with C++ extension
pip install . --install-option="--cpp"
```

Or build manually with CMake:

```bash
cmake -B build -G "MinGW Makefiles" ^
  -DONNXRUNTIME_ROOT=deps/onnxruntime ^
  -DFFTW3_ROOT=deps/fftw ^
  -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")

cmake --build build -j4

# Copy the built module
copy build\_deepcgh_engine*.pyd deepcgh_engine\
copy deps\fftw\libfftw3f-3.dll deepcgh_engine\
copy deps\onnxruntime\lib\onnxruntime.dll deepcgh_engine\
```

### Option 3: GPU Acceleration

For CUDA GPU acceleration, install the appropriate packages:

```bash
# ONNX Runtime with CUDA
pip install onnxruntime-gpu

# CuPy for GPU-accelerated IFFT (optional)
pip install cupy-cuda12x
```

Then configure the engine to use the CUDA execution provider:

```python
from deepcgh_engine import EngineAPI, EngineConfig, ExecutionProvider

config = EngineConfig(provider=ExecutionProvider.CUDA, use_cupy=True)
```

### Verifying Installation

```python
from deepcgh_engine import EngineAPI, EngineConfig, Status

print("AXCGH-Engine imported successfully!")

# Check if C++ engine is available
try:
    from deepcgh_engine import CppDeepCGHEngine
    print("C++ engine: available")
except ImportError:
    print("C++ engine: not available (using Python engine)")
```

---

## First Hologram Generation

### Step 1: Prepare the ONNX Model

You need a trained DeepCGH U-Net model in ONNX format. If you have a TensorFlow checkpoint, export it:

```bash
python export_onnx_v2.py
# Output: models/deepcgh_unet.onnx
```

The model must have **dual outputs**: `amp_0` [1, H, W, 1] and `phi_0` [1, H, W, 1].

### Step 2: Create Input Data

The engine accepts RGB-D data:
- **RGB**: `uint8` array of shape `[H, W, 3]` — a color image
- **Depth**: `float32` array of shape `[H, W]` — a depth map in meters (or arbitrary units)

```python
import numpy as np

H, W = 256, 256

# Create a simple test image (color bars)
rgb = np.zeros((H, W, 3), dtype=np.uint8)
for i in range(8):
    rgb[:, i*W//8:(i+1)*W//8, :] = [i*32, 255-i*32, 128]

# Create a depth map (flat plane at center, closer at edges)
yy, xx = np.mgrid[0:H, 0:W]
depth = (0.5 + 0.3 * np.sin(xx * 0.05) * np.cos(yy * 0.05)).astype(np.float32)
```

### Step 3: Initialize the Engine

```python
from deepcgh_engine import EngineAPI, EngineConfig

engine = EngineAPI()
config = EngineConfig(height=256, width=256, num_planes=5)

status = engine.init("models/deepcgh_unet.onnx", config)
if status != Status.OK:
    print(f"Init failed: {engine.last_error}")
    exit(1)

print("Engine initialized successfully!")
```

### Step 4: Generate the Hologram

```python
# Generate raw phase map (float32, range [-pi, pi])
status, phase = engine.generate_hologram(rgb, depth)
if status == Status.OK:
    print(f"Phase map shape: {phase.shape}")   # (256, 256)
    print(f"Phase range: [{phase.min():.3f}, {phase.max():.3f}]")

# Generate quantized phase map for SLM display (uint8, range [0, 255])
status, phase_u8 = engine.generate_hologram_quantized(rgb, depth)
if status == Status.OK:
    print(f"Quantized shape: {phase_u8.shape}")  # (256, 256)
    print(f"Quantized dtype: {phase_u8.dtype}")   # uint8
```

### Step 5: Save the Result

```python
from PIL import Image

# Save the phase map as an image
phase_img = Image.fromarray(phase_u8)
phase_img.save("my_first_hologram.png")
print("Hologram saved to my_first_hologram.png")
```

### Step 6: Clean Up

```python
engine.shutdown()
```

### Complete Example

```python
"""Generate and save your first hologram."""
import numpy as np
from deepcgh_engine import EngineAPI, EngineConfig, Status
from PIL import Image

# 1. Create engine and configure
engine = EngineAPI()
config = EngineConfig(height=256, width=256, num_planes=5)

# 2. Initialize with model
status = engine.init("models/deepcgh_unet.onnx", config)
assert status == Status.OK, f"Init failed: {engine.last_error}"

# 3. Prepare input data
rgb = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
depth = np.random.rand(256, 256).astype(np.float32)

# 4. Generate hologram
status, phase_u8 = engine.generate_hologram_quantized(rgb, depth)
assert status == Status.OK, f"Generation failed: {engine.last_error}"

# 5. Save result
Image.fromarray(phase_u8).save("first_hologram.png")

# 6. Clean up
engine.shutdown()
print("Done! Hologram saved to first_hologram.png")
```

---

## Understanding the Pipeline

The AXCGH-Engine pipeline transforms RGB-D input into an SLM phase map through four stages:

```
RGB-D Input → PreProcessor → InferenceCore → IFFT PostProcessor → PhaseMap
```

### Stage 1: PreProcessor

Converts raw RGB-D data into a model-ready tensor:

1. **Color-space conversion**: RGB → YCbCr (default), Grayscale, or passthrough RGB
   - Uses ITU-R BT.601 coefficients: Y = 0.299R + 0.587G + 0.114B
2. **Depth normalization**: Min-max normalization to [0, 1]
3. **Multi-plane volume assembly**: Distributes color and depth across `num_planes` depth planes
   - Center plane: full-intensity color data
   - Off-center planes: blend of color and depth (farther from center = more depth weight)
4. **Mean/std normalization**: Optional per-channel normalization

**Output**: `float32` tensor of shape `[1, H, W, num_planes]` (NHWC layout)

### Stage 2: InferenceCore (ONNX Runtime)

Runs the U-Net neural network forward pass:

- **Input**: `[1, H, W, num_planes]` preprocessed volume
- **Output**: Two tensors (dual output):
  - `amp_0` [1, H, W, 1] — predicted amplitude
  - `phi_0` [1, H, W, 1] — predicted initial phase

The model is a non-iterative U-Net that directly predicts the complex field representation. This is much faster than iterative methods like Gerchberg-Saxton.

### Stage 3: IFFT Post-Processor

Converts the neural network output to a physical SLM phase map:

1. **Complex field construction**: `E = amp_0 * exp(j * phi_0)`
2. **IFFT shift**: Swap quadrants (equivalent to `np.fft.ifftshift`)
3. **2D Inverse FFT**: Transform from frequency domain to spatial domain
   - Python engine: `numpy.fft.ifft2` (or `cupy.fft.ifft2` with GPU)
   - C++ engine: FFTW3f `fftwf_plan_dft_2d`
4. **Phase extraction**: `angle(E)` → phase in [-π, π]
5. **SLM quantization**: Round phase to discrete levels (e.g., 256 levels for 8-bit SLM)

**Output**: `float32` phase map of shape `[H, W]` with values in [-π, π]

### Stage 4: Quantization (Optional)

Maps the continuous phase to SLM-compatible integer values:

| PhaseFormat | Range | Use Case |
|-------------|-------|----------|
| `Uint8` | [0, 255] | Common 8-bit SLMs |
| `Uint16` | [0, 65535] | High-precision 10/16-bit SLMs |
| `Float` | [-π, π] | Debug / export / further processing |

---

## Common Troubleshooting

### "Model load failed" / Status.MODEL_LOAD_FAILED

**Cause**: The ONNX model file is missing, corrupted, or incompatible.

**Solutions**:
- Verify the model path exists: `os.path.exists("models/deepcgh_unet.onnx")`
- Ensure the model has **two outputs** named `amp_0` and `phi_0`
- Check that the model was exported with `export_onnx_v2.py` (not the older `export_onnx.py`)
- Re-export the model if necessary

### "Size mismatch" / Status.SIZE_MISMATCH

**Cause**: Input dimensions don't match the model's expected input shape.

**Solutions**:
- Ensure `EngineConfig(height, width)` matches the model's training resolution
- The default model is trained at 256×256; use `EngineConfig(height=256, width=256)`
- For 512×512 models, use `EngineConfig(height=512, width=512)`
- `height` and `width` must be divisible by `int_factor` (default: 2)

### "Execution provider unavailable" / Status.PROVIDER_UNAVAIL

**Cause**: The requested GPU execution provider is not available.

**Solutions**:
- For CUDA: install `onnxruntime-gpu` and verify CUDA/cuDNN installation
- For DirectML (Windows): install `onnxruntime-directml`
- Fall back to CPU: `EngineConfig(provider=ExecutionProvider.CPU)`

### ImportError: No module named 'deepcgh_engine'

**Solutions**:
- Install the package: `pip install .` from the `DeepCGHEngine/` directory
- Or add to PYTHONPATH: `set PYTHONPATH=F:\deepcgh\DeepCGHEngine`

### ImportError: No module named 'onnxruntime'

**Solutions**:
```bash
pip install onnxruntime          # CPU only
pip install onnxruntime-gpu      # With CUDA support
```

### C++ Engine Not Available

The C++ engine (`CppDeepCGHEngine`) requires compilation. If you see:

```
C++ engine: not available (using Python engine)
```

This is normal if you haven't built the C++ extension. The Python engine is fully functional and uses NumPy FFT instead of FFTW3. The C++ engine offers slightly different performance characteristics but is not required.

### Slow Inference on CPU

**Solutions**:
- Increase thread count: `EngineConfig(intra_op_threads=8)`
- Use INT8 quantized model: see `examples/06_int8_quantization.py`
- Switch to GPU: `EngineConfig(provider=ExecutionProvider.CUDA)`
- Reduce resolution: `EngineConfig(height=128, width=128)` (requires compatible model)

### CuPy Not Accelerating

**Solutions**:
- Verify CuPy is installed: `python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"`
- Ensure `use_cupy=True` and `provider=ExecutionProvider.CUDA` are both set
- CuPy acceleration only affects IFFT post-processing, not ONNX inference

### Phase Map Appears Blank or Uniform

**Cause**: The depth map may be constant (all zeros or all same value).

**Solutions**:
- Ensure depth has variation: `depth.min() != depth.max()`
- Normalize depth to [0, 1] before passing to the engine
- Try a non-trivial test pattern instead of random noise
