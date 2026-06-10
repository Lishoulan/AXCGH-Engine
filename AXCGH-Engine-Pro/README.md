# AXCGH-Engine Pro

高性能全息渲染引擎专业版 — TensorRT 加速、多 GPU 并行、云 API 推理。

## 版本对比

| 特性 | Community | Pro | Enterprise |
|------|-----------|-----|------------|
| CPU 推理 (ONNX Runtime) | ✅ | ✅ | ✅ |
| CUDA GPU 加速 | ✅ | ✅ | ✅ |
| TensorRT FP16 推理 | ❌ | ✅ | ✅ |
| TensorRT INT8 推理 | ❌ | ✅ | ✅ |
| 多 GPU 并行 (帧级分发) | ❌ | ✅ | ✅ |
| 动态批处理推理 | ❌ | ✅ | ✅ |
| 流式推理 (Ring Buffer) | ❌ | ✅ | ✅ |
| 自动调优 (Batch Size / Precision) | ❌ | ✅ | ✅ |
| TRT 引擎磁盘缓存 | ❌ | ✅ | ✅ |
| 云 API 远程推理 | ❌ | ✅ | ✅ |
| SLM 驱动支持 | ✅ | ✅ | ✅ |
| 实时预览 | ✅ | ✅ | ✅ |
| 多波长 RGB | ✅ | ✅ | ✅ |
| 优先技术支持 | ❌ | 社区 | 专属 |
| SLA 保障 | ❌ | ❌ | ✅ |
| 私有化部署 | ❌ | ❌ | ✅ |
| 定制模型训练 | ❌ | ❌ | ✅ |

## 定价

| 版本 | 年费 | 说明 |
|------|------|------|
| Community | 免费 | 开源，MIT 许可 |
| Pro | ¥4,999/年 | 单机授权，含云 API 额度 |
| Enterprise | 联系销售 | 私有化部署，无限额度 |

> Pro 版包含每月 10,000 帧云 API 推理额度。超出部分按量计费。

## 安装

> **注意**: Pro 版本需要有效的许可证密钥才能使用。

### 1. 安装基础引擎

```bash
pip install axcgh-engine>=1.0.0
```

### 2. 安装 Pro 版本

```bash
# 从私有仓库安装
pip install axcgh-engine-pro-1.0.0-py3-none-any.whl

# 或从源码安装
git clone https://github.com/Lishoulan/AXCGH-Engine-Pro.git
cd AXCGH-Engine-Pro
pip install .
```

### 3. 激活许可证

```python
from axcgh_pro import LicenseManager

lm = LicenseManager()
lm.validate_key("AXCGH-PRO-XXXX-XXXX-XXXX-XXXX")
```

或设置环境变量:

```bash
export AXCGH_LICENSE_KEY="AXCGH-PRO-XXXX-XXXX-XXXX-XXXX"
```

### 4. 验证安装

```python
from axcgh_pro import ProEngineAPI

engine = ProEngineAPI()
print(f"Pro 版本已激活: {engine.is_licensed()}")
```

## GPU / TensorRT 性能基准

测试环境: NVIDIA RTX 4090, ONNX Model 256x256, 5 planes

| 配置 | 推理延迟 (ms) | 吞吐量 (fps) | 加速比 |
|------|---------------|-------------|--------|
| CPU (FP32) | 45.2 | 22 | 1.0x |
| CUDA (FP32) | 8.3 | 120 | 5.4x |
| CUDA (FP16) | 4.7 | 213 | 9.6x |
| TensorRT (FP16) | 2.1 | 476 | 21.5x |
| TensorRT (INT8) | 1.4 | 714 | 32.3x |
| TensorRT (FP16, Batch=4) | 0.8/frame | 1250 | 56.5x |
| TensorRT (INT8, Batch=4) | 0.5/frame | 2000 | 90.4x |

测试环境: NVIDIA A100 80GB, ONNX Model 512x512, 5 planes

| 配置 | 推理延迟 (ms) | 吞吐量 (fps) | 加速比 |
|------|---------------|-------------|--------|
| CPU (FP32) | 182.5 | 5 | 1.0x |
| CUDA (FP32) | 15.6 | 64 | 11.7x |
| CUDA (FP16) | 8.2 | 122 | 22.3x |
| TensorRT (FP16) | 3.8 | 263 | 48.0x |
| TensorRT (INT8) | 2.3 | 435 | 79.3x |
| TensorRT (FP16, Batch=8) | 1.1/frame | 909 | 165.9x |
| TensorRT (INT8, Batch=8) | 0.7/frame | 1429 | 260.7x |

> 以上数据为 50 次推理取中位数，不含 IFFT 后处理时间。

## 快速开始

### TensorRT 加速推理

```python
from axcgh_pro import ProEngineAPI
from deepcgh_engine import EngineConfig
import numpy as np

engine = ProEngineAPI()
engine.init(
    model_path="models/deepcgh_unet.onnx",
    config=EngineConfig(height=256, width=256),
    auto_tune=True,  # 自动寻找最优 batch size 和精度
)

rgb = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
depth = np.random.rand(256, 256).astype(np.float32)

status, phase = engine.generate_hologram(rgb, depth)
engine.shutdown()
```

### 多 GPU 并行

```python
from axcgh_pro import ProEngineAPI

engine = ProEngineAPI(multi_gpu=True, gpu_ids=[0, 1])
engine.init("models/deepcgh_unet.onnx", EngineConfig(height=256, width=256))

# 批量帧自动分发到多 GPU
frames = [(rgb1, depth1), (rgb2, depth2), (rgb3, depth3), (rgb4, depth4)]
status, phases = engine.generate_hologram_batch(
    [f[0] for f in frames],
    [f[1] for f in frames],
)
```

### 流式推理

```python
from axcgh_pro import ProEngineAPI

engine = ProEngineAPI()
engine.init("models/deepcgh_unet.onnx", EngineConfig(height=256, width=256))

# 启动流式模式 (Ring Buffer)
engine.start_stream(buffer_size=8)

# 持续推入帧
for rgb, depth in video_stream:
    engine.push_frame(rgb, depth)

# 获取结果
while engine.is_streaming():
    result = engine.pop_result()
    if result is not None:
        phase = result
```

### 云 API 推理

```python
from axcgh_pro import CloudEngineClient

client = CloudEngineClient()
client.connect(api_key="axcgh-api-xxxxx", endpoint="https://api.axcgh.com")

status, phase = client.generate_hologram(rgb, depth)
credits = client.get_credits()
```

## 联系我们

- **官网**: https://axcgh.com
- **邮箱**: pro@axcgh.com
- **GitHub**: https://github.com/Lishoulan/AXCGH-Engine
- **技术支持**: support@axcgh.com
- **企业咨询**: enterprise@axcgh.com
