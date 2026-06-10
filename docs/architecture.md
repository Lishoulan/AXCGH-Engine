---
layout: default
title: Architecture
---

# Architecture Deep Dive

This document provides a comprehensive look at the internal architecture of AXCGH-Engine, including system structure, module responsibilities, data flow, memory management, FFT implementation comparison, and ONNX model structure.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            AXCGH-Engine                                     │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        Python Layer                                  │   │
│  │                                                                      │   │
│  │  ┌──────────────┐  ┌──────────────────┐  ┌───────────────────────┐  │   │
│  │  │  EngineAPI   │  │ RGBHologramEngine │  │ RealtimeHologramDisp │  │   │
│  │  │  (Python)    │  │ (3x EngineAPI)    │  │ (Threaded Pipeline)   │  │   │
│  │  └──────┬───────┘  └────────┬─────────┘  └───────────┬───────────┘  │   │
│  │         │                   │                        │              │   │
│  │  ┌──────┴───────────────────┴────────────────────────┴──────────┐  │   │
│  │  │                    Core Pipeline                              │  │   │
│  │  │                                                               │  │   │
│  │  │  ┌──────────────┐  ┌───────────────┐  ┌──────────────────┐  │  │   │
│  │  │  │ PreProcessor │─▶│ InferenceCore │─▶│ IFFTPostProcess  │  │  │   │
│  │  │  │ (NumPy/CuPy) │  │ (ONNX Runtime)│  │ (NumPy/CuPy FFT) │  │  │   │
│  │  │  └──────────────┘  └───────────────┘  └──────────────────┘  │  │   │
│  │  └──────────────────────────────────────────────────────────────┘  │   │
│  │                              │                                      │   │
│  │  ┌───────────────────────────┴──────────────────────────────────┐  │   │
│  │  │                    SLM Driver Layer                           │  │   │
│  │  │  ┌────────────────┐  ┌──────────┐  ┌──────────────────────┐ │  │   │
│  │  │  │ DirectDisplay  │  │   SDK    │  │    FileBackend       │ │  │   │
│  │  │  │ (pygame/opencv)│  │ (vendor) │  │  (PNG/BMP output)   │ │  │   │
│  │  │  └────────────────┘  └──────────┘  └──────────────────────┘ │  │   │
│  │  └──────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     C++ Layer (Optional)                             │   │
│  │                                                                      │   │
│  │  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────────┐  │   │
│  │  │ PreProcessor │  │ InferenceCore │  │  IFFTPostProcessor       │  │   │
│  │  │ (C++ loops)  │  │ (ORT C++ API) │  │  (FFTW3f)               │  │   │
│  │  └──────────────┘  └───────────────┘  └──────────────────────────┘  │   │
│  │         │                  │                     │                   │   │
│  │  ┌──────┴──────────────────┴─────────────────────┴──────────────┐  │   │
│  │  │                    EngineAPI (C++)                            │  │   │
│  │  └──────────────────────────┬───────────────────────────────────┘  │   │
│  │                             │                                       │   │
│  │  ┌──────────────────────────┴───────────────────────────────────┐  │   │
│  │  │              PyBind11 Bindings (_deepcgh_engine.pyd)         │  │   │
│  │  │  • NumPy ↔ C++ buffer conversion                            │  │   │
│  │  │  • IFFT via NumPy FFT (not FFTW3) in Python binding layer   │  │   │
│  │  └──────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Responsibilities

### Python Package (`deepcgh_engine/`)

| Module | Responsibility |
|--------|---------------|
| `engine.py` | Core pipeline: `EngineAPI`, `PreProcessor`, `InferenceCore`, `IFFTPostProcessor`, `EngineConfig`, and all type enums |
| `multi_wavelength.py` | RGB three-wavelength holography: `RGBHologramEngine`, `CombineMode`, `SpatialPattern` |
| `slm_driver.py` | SLM display abstraction: `SLMDriver`, `DirectDisplayBackend`, `SDKBackend`, `FileBackend` |
| `realtime.py` | Real-time preview: `RealtimeHologramDisplay`, camera sources, performance monitor |
| `__init__.py` | Package entry point: re-exports all public APIs, attempts C++ engine import |

### C++ Source (`src/`, `include/`)

| File | Responsibility |
|------|---------------|
| `Types.h` | Core data structures: `EngineConfig`, `RGBDFrame`, `PhaseMap`, enums (`PhaseFormat`, `ColorSpace`, `ExecutionProvider`, `Status`) |
| `PreProcessor.h/.cpp` | RGB-D preprocessing: color-space conversion (BT.601), depth normalization, multi-plane volume assembly |
| `InferenceCore.h/.cpp` | ONNX Runtime inference: model loading, session management, memory pooling, dual-output forward pass |
| `EngineAPI.h/.cpp` | Top-level facade: orchestrates PreProcessor → InferenceCore → IFFTPostProcessor; FFTW3f-based IFFT |
| `pybind_module.cpp` | PyBind11 bindings: exposes C++ engine to Python with NumPy array interop |

### Tools

| File | Responsibility |
|------|---------------|
| `quantize_model.py` | INT8 quantization: static/dynamic quantization, calibration, accuracy comparison |

---

## Data Flow Through the Pipeline

```
                         Input
                           │
                    ┌──────┴──────┐
                    │  RGB [H,W,3] │  uint8
                    │ Depth [H,W]  │  float32
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │ PreProcessor │
                    │              │
                    │ 1. Color     │  RGB → YCbCr/Gray
                    │    convert   │  ITU-R BT.601
                    │              │
                    │ 2. Depth     │  Min-max → [0,1]
                    │    normalize │
                    │              │
                    │ 3. Volume    │  Color + Depth → [H,W,C]
                    │    assemble  │  Center plane = color
                    │              │  Off-center = blend
                    │              │
                    │ 4. Norm      │  (x - mean) / std
                    └──────┬──────┘
                           │
                  [1, H, W, C] float32
                  (NHWC layout)
                           │
                    ┌──────┴──────┐
                    │InferenceCore│
                    │             │
                    │ ONNX Runtime│  U-Net forward pass
                    │ session.Run │
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │                         │
        amp_0 [1,H,W,1]          phi_0 [1,H,W,1]
        (amplitude)               (initial phase)
              │                         │
              └────────────┬────────────┘
                           │
                    ┌──────┴──────┐
                    │   IFFT      │
                    │ PostProcess │
                    │             │
                    │ 1. Complex  │  amp * exp(j * phi)
                    │    field    │
                    │             │
                    │ 2. IFFT     │  ifftshift → ifft2
                    │    shift+2D │
                    │             │
                    │ 3. Angle    │  atan2(im, re)
                    │    extract  │
                    │             │
                    │ 4. Quantize │  [-π,π] → levels → [-π,π]
                    └──────┬──────┘
                           │
                  phase [H, W] float32
                  range: [-π, π]
                           │
                    ┌──────┴──────┐
                    │ Quantization │
                    │  (optional)  │
                    │              │
                    │ Uint8: [0,255]│
                    │ Uint16:[0,65535]│
                    │ Float: [-π,π]│
                    └──────┬──────┘
                           │
                       SLM Display
```

### Multi-Plane Volume Assembly Detail

The volume assembly step is critical for encoding 3D structure:

```
num_planes = 5, center_plane = 2

Plane 0:  color * (1 - 1.0) + depth * 1.0  =  depth only
Plane 1:  color * (1 - 0.5) + depth * 0.5  =  blend
Plane 2:  color * 1.0       + depth * 0.0  =  color only (center)
Plane 3:  color * (1 - 0.5) + depth * 0.5  =  blend
Plane 4:  color * (1 - 1.0) + depth * 1.0  =  depth only

depth_weight = |plane_index - center| / (num_planes // 2)
```

This distribution tells the U-Net where objects are in depth, allowing it to generate appropriate diffraction patterns for each depth plane.

---

## Memory Management (Pooling)

### Why Pooling?

In a real-time hologram generation loop, allocating and deallocating tensors every frame introduces significant overhead. AXCGH-Engine uses **memory pooling** to reuse buffers across frames.

### C++ Engine Pooling

The `InferenceCore` class maintains three pooled buffers:

| Buffer | Size | Purpose |
|--------|------|---------|
| `pooled_input_` | `input_size_` floats | Preprocessed input tensor |
| `pooled_amp_` | `amp_size_` floats | Amplitude output from U-Net |
| `pooled_phi_` | `phi_size_` floats | Phase output from U-Net |

**Lifecycle**:
1. `load_model()` → `allocate_pools()` allocates buffers based on model I/O shapes
2. `forward()` → copies input into `pooled_input_`, ORT writes to `pooled_amp_`/`pooled_phi_`
3. Results are `memcpy`'d to the caller's output buffers
4. Buffers persist across frames — no per-frame allocation

**Configuration**: `EngineConfig.enable_memory_pool = True` (default)

### IFFT Buffer Reuse

The `IFFTPostProcessor` reuses FFTW3 plans and working buffers:

- `fft_in_` / `fft_out_`: FFTW3 complex arrays, allocated once per dimension
- `fft_plan_`: FFTW3 plan, created with `FFTW_MEASURE` for optimal performance
- Plans are **reused** if dimensions haven't changed since the last call
- Plans are **recreated** only when input dimensions change

### Python Engine

The Python engine uses NumPy array reuse implicitly:
- Pre-allocated weight arrays in `PreProcessor` (`_color_weights`, `_depth_weights`)
- CuPy GPU arrays are transferred only when needed (`cp.asarray` / `cp.asnumpy`)
- ONNX Runtime session handles its own internal memory management

---

## FFTW3 vs NumPy FFT Comparison

Both FFT implementations produce identical results but differ in performance characteristics and integration:

| Aspect | FFTW3f (C++ Engine) | NumPy FFT (Python Engine) |
|--------|---------------------|--------------------------|
| **Library** | FFTW3 (C library) | NumPy (wraps FFTPACK/PocketFFT) |
| **Precision** | Single-precision (float) | Single-precision (complex64) |
| **Plan Strategy** | `FFTW_MEASURE` — benchmarks multiple algorithms at load time | No planning — uses fixed algorithm |
| **First Call** | Slower (plan measurement) | Fast (no planning overhead) |
| **Subsequent Calls** | Faster (cached plan) | Same speed every call |
| **Memory** | Manual allocation via `fftwf_alloc_complex` | NumPy managed arrays |
| **Normalization** | Manual: `1/N` factor applied after IFFT | Built-in: `np.fft.ifft2` includes `1/N` |
| **GPU Acceleration** | Not available | CuPy `cp.fft.ifft2` on GPU |
| **Thread Safety** | Plan reuse requires same dimensions | Stateless — inherently thread-safe |
| **Windows DLL** | Requires `libfftw3f-3.dll` | No external DLL needed |
| **Python Binding** | Bypassed — PyBind11 uses NumPy FFT instead | Direct use |

### Why PyBind11 Uses NumPy FFT

The Python binding layer (`pybind_module.cpp`) intentionally uses NumPy FFT instead of the C++ FFTW3 IFFT:

1. **Correctness**: NumPy's `ifft2` is well-tested and includes proper normalization
2. **Simplicity**: No need to ship FFTW3 DLL with the Python package
3. **Performance**: NumPy FFT is competitive for typical hologram sizes (256×256 to 1080p)
4. **GPU Option**: Users can swap to CuPy FFT for GPU acceleration

The C++ FFTW3 path is used in the standalone C++ application (`apps/main.cpp`) where Python is not available.

### Performance Characteristics

At 256×256 resolution:

| Implementation | IFFT Time | Notes |
|----------------|-----------|-------|
| NumPy FFT (CPU) | ~1-2 ms | Consistent, no warm-up |
| CuPy FFT (GPU) | ~0.3-0.5 ms | Requires CUDA + CuPy |
| FFTW3f (C++) | ~0.5-1 ms | Faster after plan measurement |

At 1920×1080 resolution:

| Implementation | IFFT Time | Notes |
|----------------|-----------|-------|
| NumPy FFT (CPU) | ~15-25 ms | Single-threaded |
| CuPy FFT (GPU) | ~1-2 ms | Significant speedup |
| FFTW3f (C++) | ~8-15 ms | Multi-threaded FFTW3 |

---

## ONNX Model Structure (Dual Output)

### Model Architecture

The DeepCGH ONNX model is a **U-Net** that takes a multi-plane volume as input and produces **two outputs**: amplitude and initial phase.

```
Input: target [1, H, W, num_planes]
         │
         ▼
    ┌────────────────────────────┐
    │         U-Net              │
    │                            │
    │  Encoder: Conv → BN → ReLU │
    │  Bottleneck                │
    │  Decoder: ConvT → BN → ReLU│
    │  Skip connections          │
    │                            │
    └──────┬──────────┬──────────┘
           │          │
     amp_0 [1,H,W,1]  phi_0 [1,H,W,1]
     (amplitude)       (initial phase)
```

### Why Dual Output?

The original DeepCGH TensorFlow model includes a Lambda layer (`_ifft_AmPh`) that converts `(amp_0, phi_0)` into the final SLM phase. This Lambda cannot be exported to ONNX because it uses `tf.ifft2d`.

**Solution**: The ONNX model exports only the U-Net subgraph (up to `amp_0` and `phi_0`), and the IFFT post-processing is implemented natively in the engine:

- **Python engine**: `numpy.fft.ifft2` or `cupy.fft.ifft2`
- **C++ engine**: FFTW3f `fftwf_plan_dft_2d`

### Export Process

The model is exported from TensorFlow using `export_onnx_v2.py`:

```python
# Simplified export flow:
# 1. Load DeepCGH TensorFlow model
# 2. Extract U-Net subgraph (before _ifft_AmPh Lambda)
# 3. Export to ONNX with two outputs:
#    - amp_0: shape [1, H, W, 1], dtype float32
#    - phi_0: shape [1, H, W, 1], dtype float32
# 4. Save as deepcgh_unet.onnx
```

### Model I/O Specification

| I/O | Name | Shape | Dtype | Description |
|-----|------|-------|-------|-------------|
| Input | `target` | `[1, H, W, num_planes]` | float32 | Preprocessed RGB-D volume |
| Output 0 | `amp_0` | `[1, H, W, 1]` | float32 | Predicted amplitude |
| Output 1 | `phi_0` | `[1, H, W, 1]` | float32 | Predicted initial phase |

### INT8 Quantized Model

The quantized model (`deepcgh_unet_int8.onnx`) uses QDQ (Quantize-Dequantize) format:

- **Weights**: INT8 with per-channel quantization
- **Activations**: UINT8 with static calibration
- **Size reduction**: ~2.79× (0.54 MB → 0.19 MB for 256×256 model)
- **Accuracy**: Max absolute difference < 0.3 compared to FP32

The quantization is performed using `tools/quantize_model.py` with synthetic calibration data matching the model's input distribution.

### Model Validation

When loading a model, `InferenceCore` validates:

1. Input shape is 4D (either NHWC or NCHW layout)
2. Spatial dimensions match `EngineConfig.height` and `EngineConfig.width`
3. At least 2 outputs exist (amp_0 and phi_0)

If validation fails, `load_model()` returns `Status.SIZE_MISMATCH`.
