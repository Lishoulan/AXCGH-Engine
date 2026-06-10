# AXCGH-Engine Examples

This directory contains self-contained example scripts demonstrating various features of the AXCGH-Engine holographic rendering engine.

## Quick Start

Make sure you have the engine installed and an ONNX model available:

```bash
pip install onnxruntime numpy pillow
cd DeepCGHEngine && pip install .
```

## Examples

| # | File | Description |
|---|------|-------------|
| 01 | [01_basic_hologram.py](01_basic_hologram.py) | **Basic hologram generation** — Generate and save a hologram phase map using the Python engine. The simplest starting point. |
| 02 | [02_cpp_engine.py](02_cpp_engine.py) | **C++ engine** — Use the C++ engine (via PyBind11) and compare performance with the Python engine. Falls back to Python if C++ extension is not available. |
| 03 | [03_rgb_wavelength.py](03_rgb_wavelength.py) | **Multi-wavelength RGB hologram** — Generate separate phase maps for red (633nm), green (532nm), and blue (450nm) wavelengths, combined using time-division or spatial multiplexing. |
| 04 | [04_slm_display.py](04_slm_display.py) | **SLM display** — Send phase maps to a Spatial Light Modulator using DirectDisplay (HDMI), FileBackend (save to disk), or SDKBackend (vendor-specific). |
| 05 | [05_realtime_preview.py](05_realtime_preview.py) | **Real-time preview** — Live hologram generation with camera input (RealSense/Kinect/test pattern), threaded pipeline, and FPS overlay. |
| 06 | [06_int8_quantization.py](06_int8_quantization.py) | **INT8 quantization** — Quantize an FP32 ONNX model to INT8 for smaller size and faster inference. Compare accuracy and performance. |
| 07 | [07_batch_processing.py](07_batch_processing.py) | **Batch processing** — Process an entire folder of images, generating holograms for each. Supports custom depth map generation methods. |
| 08 | [08_custom_depth.py](08_custom_depth.py) | **Custom depth maps** — Create various depth maps (text, shapes, gradients, multi-plane) and generate holograms. Demonstrates how depth affects hologram output. |

## Running Examples

All examples can be run directly:

```bash
# From the project root directory
python examples/01_basic_hologram.py
python examples/07_batch_processing.py --input_dir data/natural_images
```

## Prerequisites

| Example | Additional Dependencies |
|---------|------------------------|
| 01, 02, 03, 06, 07, 08 | `onnxruntime`, `numpy`, `pillow` |
| 02 | C++ extension (optional) |
| 04 | `pygame` or `opencv-python` (for DirectDisplay) |
| 05 | `opencv-python` (for preview), `pyrealsense2` or `pyk4a` (for cameras) |

## Model Files

All examples require an ONNX model file. The default path is:

```
DeepCGHEngine/models/deepcgh_unet.onnx
```

If you don't have a model yet, export one from a TensorFlow checkpoint:

```bash
python export_onnx_v2.py
```

## Output

Example outputs are saved to the `result/` directory by default.
