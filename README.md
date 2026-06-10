<div align="center">

# AXCGH-Engine

**Deep Learning Holographic Rendering Engine**

RGB-D → Neural Network Inference → SLM Phase Map

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-green.svg)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-orange.svg)](https://isocpp.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.26%2B-purple.svg)](https://onnxruntime.ai/)
[![WeChat Pay](https://img.shields.io/badge/WeChat%20Pay-%E5%BE%AE%E4%BF%A1-green.svg)](#-support--purchase)

[English](#overview) · [功能特性](#功能特性) · [快速开始](#快速开始) · [架构](#架构) · [API](#api) · [性能](#性能) · [构建](#构建)

</div>

---

## Overview

AXCGH-Engine is a lightweight, production-ready holographic rendering engine that leverages deep learning (U-Net) to generate computer-generated holograms (CGH) in real time. It accepts RGB-D data streams, performs neural network inference via ONNX Runtime, and outputs quantized phase maps for Spatial Light Modulator (SLM) display.

```
┌──────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────┐    ┌──────────┐
│  RGB-D   │───▶│ PreProcessor │───▶│ InferenceCore │───▶│   IFFT   │───▶│   SLM    │
│  Input   │    │ (YCbCr/Depth)│    │  (U-Net/ORT)  │    │ (FFTW3f) │    │ Display  │
└──────────┘    └──────────────┘    └───────────────┘    └──────────┘    └──────────┘
```

## 功能特性

- **双引擎架构** — Python (NumPy FFT) 和 C++ (FFTW3f) 两种引擎，可互换使用
- **多波长 RGB 全息图** — 支持 633nm/532nm/450nm 三波长，时分/空分复用
- **INT8 量化** — 模型压缩 2.79x，精度损失 < 0.02
- **SLM 驱动** — 支持 HDMI 直连显示、SDK 接口、文件输出三种后端
- **实时预览** — 摄像头采集 → 全息渲染 → SLM 显示，三线程流水线
- **C++ 独立部署** — 无 Python 依赖，纯 C++ + FFTW3 + ONNX Runtime
- **多分辨率** — 支持 256×256 / 512×512 / 1920×1080 训练脚本

## 快速开始

### 安装依赖

```bash
pip install onnxruntime numpy pillow
```

### 30 秒生成全息图

```python
from deepcgh_engine import EngineAPI, EngineConfig
import numpy as np

engine = EngineAPI()
engine.init("models/deepcgh_unet.onnx", EngineConfig(height=256, width=256, num_planes=5))

rgb = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
depth = np.random.rand(256, 256).astype(np.float32)

status, phase = engine.generate_hologram(rgb, depth)
# phase: float32 [256, 256], range [-π, π]
```

### C++ 引擎（更快）

```python
from deepcgh_engine import CppDeepCGHEngine

engine = CppDeepCGHEngine()
engine.init("models/deepcgh_unet.onnx", height=256, width=256, num_planes=5)

phase = engine.generate_hologram(rgb, depth)          # float32 [-π, π]
quantized = engine.generate_hologram_quantized(rgb, depth)  # uint8 [0, 255]
```

### RGB 三波长全息图

```python
from deepcgh_engine import RGBHologramEngine, RGBEngineConfig, CombineMode, EngineConfig

config = RGBEngineConfig(
    base_config=EngineConfig(height=256, width=256, num_planes=5),
    combine_mode=CombineMode.SpatialMultiplex
)
engine = RGBHologramEngine()
engine.init("models/deepcgh_unet.onnx", config)

status, result = engine.generate_rgb_hologram(rgb, depth)
# result['phase_r']  — Red (633nm)
# result['phase_g']  — Green (532nm)
# result['phase_b']  — Blue (450nm)
# result['phase_combined'] — Combined phase map
```

### SLM 显示

```python
from deepcgh_engine import create_slm_driver

# HDMI 直连 SLM（副屏全屏显示）
slm = create_slm_driver('direct', resolution=(1920, 1080), bit_depth=8)
slm.display(phase)

# 或保存到文件
slm = create_slm_driver('file', resolution=(256, 256), output_dir='frames')
slm.display(phase)
```

### 实时预览

```python
from deepcgh_engine import RealtimeHologramDisplay, RealtimeConfig, EngineConfig

config = RealtimeConfig(
    model_path="models/deepcgh_unet.onnx",
    engine_config=EngineConfig(height=256, width=256, num_planes=5),
    camera_type='test',   # 'realsense', 'kinect', 'test'
    target_fps=30
)
display = RealtimeHologramDisplay(config)
display.run()  # q=quit, s=screenshot, p=pause
```

## 架构

```
AXCGH-Engine/
├── include/deepcgh/              # C++ Headers
│   ├── Types.h                   # Data structures (RGBDFrame, PhaseMap, EngineConfig)
│   ├── PreProcessor.h            # RGB-D preprocessing (YCbCr, depth normalization)
│   ├── InferenceCore.h           # ONNX Runtime inference engine
│   └── EngineAPI.h               # High-level API (FFTW3 IFFT post-processor)
├── src/                          # C++ Implementation
│   ├── PreProcessor.cpp
│   ├── InferenceCore.cpp
│   └── EngineAPI.cpp             # FFTW3f-based IFFT2D
├── bindings/
│   └── pybind_module.cpp         # PyBind11 Python bindings
├── apps/
│   └── main.cpp                  # Standalone C++ demo (no Python)
├── deepcgh_engine/               # Python Package
│   ├── engine.py                 # Pure Python engine (NumPy FFT)
│   ├── multi_wavelength.py       # RGB three-wavelength holography
│   ├── slm_driver.py             # SLM display driver
│   ├── realtime.py               # Real-time preview system
│   └── __init__.py
├── tools/
│   └── quantize_model.py         # INT8 quantization tool
├── models/                       # ONNX models (git-ignored)
├── train_512.py                  # Train 512×512 model
├── train_1080p.py                # Train 1920×1080 model
├── export_onnx_v2.py             # Export TF → ONNX (dual output)
└── CMakeLists.txt                # Build system
```

### Pipeline

```
RGB-D Input
    │
    ▼
┌─────────────────────────────────┐
│  PreProcessor                   │
│  • RGB → YCbCr (ITU-R BT.601)  │
│  • Depth normalization          │
│  • Multi-plane volume assembly  │
│  • Output: [1, H, W, N_planes] │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  InferenceCore (ONNX Runtime)   │
│  • U-Net forward pass           │
│  • Input:  [1, H, W, N_planes] │
│  • Output: amp_0 [1,H,W,1]     │
│           phi_0 [1,H,W,1]      │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  IFFT Post-Processor (FFTW3f)   │
│  • Complex field: amp·exp(jφ)   │
│  • IFFT shift + IFFT2D          │
│  • Extract angle → phase        │
│  • SLM quantization (8/10-bit)  │
│  • Output: phase [H, W] [-π, π] │
└─────────────────────────────────┘
```

## API

### Python Engine

| Method | Description |
|--------|-------------|
| `EngineAPI()` | Create engine instance |
| `engine.init(model_path, config)` | Initialize with model and config |
| `engine.generate_hologram(rgb, depth)` | Generate phase map → `(Status, ndarray)` |
| `engine.shutdown()` | Release resources |

### C++ Engine (PyBind11)

| Method | Description |
|--------|-------------|
| `CppDeepCGHEngine()` | Create C++ engine instance |
| `engine.init(model_path, **kwargs)` | Initialize with keyword config |
| `engine.generate_hologram(rgb, depth)` | Phase map via FFTW3 IFFT → `ndarray` |
| `engine.generate_hologram_quantized(rgb, depth)` | Quantized phase → `ndarray` |
| `engine.infer_raw(rgb, depth)` | Raw amp_0, phi_0 → `(ndarray, ndarray)` |

### Multi-Wavelength

| Method | Description |
|--------|-------------|
| `RGBHologramEngine()` | Create RGB engine |
| `engine.init(model_path, config)` | Initialize three sub-engines |
| `engine.generate_rgb_hologram(rgb, depth)` | R/G/B + combined phases |
| `CombineMode.TimeDivision` | Sequential display mode |
| `CombineMode.SpatialMultiplex` | Checkerboard/stripe interleaving |

### SLM Driver

| Backend | Description |
|---------|-------------|
| `create_slm_driver('direct')` | Fullscreen on secondary display (pygame/opencv) |
| `create_slm_driver('sdk')` | Vendor SDK placeholder (Holoeye/Meadowlark) |
| `create_slm_driver('file')` | Save to PNG/BMP files |

## 性能

Benchmark at 256×256, 5 depth planes, CPU-only:

| Engine | Latency | FPS | IFFT Backend |
|--------|---------|-----|-------------|
| Python | 16.7 ms | 60.0 | NumPy FFT |
| C++ | 21.0 ms | 47.6 | FFTW3f |
| C++ (INT8) | ~12 ms* | ~83* | FFTW3f |

*\*INT8 估算值，实际取决于量化精度*

| Model | Size (FP32) | Size (INT8) | Compression | Max Error |
|-------|-------------|-------------|-------------|-----------|
| deepcgh_unet | 0.54 MB | 0.19 MB | 2.79× | 0.291 |

## 构建

### Python 包（无需编译）

```bash
pip install onnxruntime numpy pillow
# 直接使用 Python 引擎
```

### C++ 引擎 + Python 绑定（MinGW-w64）

```bash
# 1. 准备依赖
#    - ONNX Runtime C++ SDK → deps/onnxruntime/
#    - FFTW3 Windows DLL    → deps/fftw/
#    - pybind11             → pip install pybind11

# 2. 生成 MinGW 导入库
cd deps/onnxruntime/lib
gendef onnxruntime.dll
dlltool --dllname onnxruntime.dll --input-def onnxruntime.def --output-lib libonnxruntime.dll.a

# 3. CMake 配置 & 编译
cmake -B build -G "MinGW Makefiles" \
  -DONNXRUNTIME_ROOT=deps/onnxruntime \
  -DFFTW3_ROOT=deps/fftw \
  -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")

cmake --build build -j4

# 4. 复制产物
cp build/_deepcgh_engine*.pyd deepcgh_engine/
cp deps/fftw/libfftw3f-3.dll deepcgh_engine/
cp deps/onnxruntime/lib/onnxruntime.dll deepcgh_engine/
```

### 独立 C++ 程序（无 Python 依赖）

```bash
cmake -B build -G "MinGW Makefiles" \
  -DONNXRUNTIME_ROOT=deps/onnxruntime \
  -DFFTW3_ROOT=deps/fftw \
  -DDEEPCGH_BUILD_APPS=ON

cmake --build build --target deepcgh_demo

./build/deepcgh_demo --model models/deepcgh_unet.onnx --benchmark 30
```

### INT8 量化

```bash
python tools/quantize_model.py \
  --input models/deepcgh_unet.onnx \
  --output models/deepcgh_unet_int8.onnx \
  --calibration-samples 100

# 批量量化
python tools/quantize_model.py --quantize-all
```

### 训练高分辨率模型

```bash
# 512×512
python train_512.py

# 1920×1080
python train_1080p.py
```

## 模型导出

从 TensorFlow DeepCGH 导出 U-Net 子图到 ONNX：

```bash
python export_onnx_v2.py
# 输出: models/deepcgh_unet.onnx (dual output: amp_0 + phi_0)
```

ONNX 模型仅包含 U-Net 推理部分（amp_0, phi_0），IFFT 后处理在引擎中完成。

## 依赖

| Dependency | Version | Purpose |
|-----------|---------|---------|
| ONNX Runtime | ≥ 1.26 | Neural network inference |
| FFTW3 | 3.3.5+ | 2D IFFT (C++ engine) |
| NumPy | ≥ 1.20 | FFT + array ops (Python engine) |
| pybind11 | ≥ 2.12 | Python-C++ bindings |
| CMake | ≥ 3.15 | Build system |
| GCC/MSVC | C++17 | Compiler |

## 🚀 AXCGH-Engine Pro

Need GPU acceleration, TensorRT, multi-GPU, or cloud inference?

| Feature | Community (Free) | Pro ($499/yr) | Enterprise ($2,999/yr) |
|---------|-----------------|---------------|----------------------|
| CPU Inference | ✅ | ✅ | ✅ |
| CUDA GPU | — | ✅ | ✅ |
| TensorRT FP16/INT8 | — | ✅ | ✅ |
| Multi-GPU | — | ✅ | ✅ |
| High-Res Models (512/1080p) | — | ✅ | ✅ |
| SLM SDK (Holoeye/Meadowlark) | — | ✅ | ✅ |
| Cloud API | — | 10K frames/yr | Unlimited |
| Priority Support | — | ✅ | ✅ |
| Custom Model Training | — | — | ✅ |

```bash
pip install axcgh-engine-pro
export AXCGH_LICENSE_KEY=your-key
```

```python
from axcgh_pro import ProEngineAPI, CloudEngineClient

# Local GPU + TensorRT
engine = ProEngineAPI()
engine.init("model.onnx", tensorrt_fp16=True, multi_gpu=True)
phase = engine.generate_hologram(rgb, depth)  # 200+ FPS on RTX 4090

# Cloud API
client = CloudEngineClient()
client.connect(api_key="your-key")
phase = client.generate_hologram(rgb, depth)  # Remote inference
```

👉 [Contact for Pro license](https://github.com/Lishoulan/AXCGH-Engine/issues/new?template=feature_request.md)

### 💰 Support & Purchase

| Channel | Link | For |
|---------|------|-----|
| WeChat Pay | Scan QR code below | One-time donation |
| Pro License | [Open an issue](https://github.com/Lishoulan/AXCGH-Engine/issues/new?template=feature_request.md) | Commercial license purchase |
| Enterprise | Contact via GitHub | Custom integration & support |

<div align="center">
  <img src="docs/wechat_pay.jpg" width="200" alt="WeChat Pay QR Code" />
  <p><em>微信扫码支持开发者</em></p>
</div>

## 引用

If you use AXCGH-Engine in your research, please cite the original DeepCGH paper:

```bibtex
@article{horstmeyer2020deepcgh,
  title={DeepCGH: Deep learning computer-generated holography},
  author={Horstmeyer, Roarke and Chen, Richard and Kappes, Benjamin and Judkewitz, Beth},
  journal={Optics Express},
  volume={28},
  number={18},
  pages={26536--26551},
  year={2020},
  publisher={Optica Publishing Group}
}
```

## License

MIT License

---

<div align="center">
Built with ❤️ for the holography community
</div>
