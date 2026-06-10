# AXCGH Cloud API

全息图推理云服务 — 无需本地 GPU，按量付费的远程全息图生成 API。

## API 参考

### 认证

所有 API 请求需在 Header 中携带 API Key:

```
X-API-Key: axcgh-api-xxxxxxxxxxxx
```

### 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/generate` | 单帧全息图推理 |
| POST | `/v1/generate_batch` | 批量全息图推理 |
| GET | `/v1/credits` | 查询剩余额度 |
| GET | `/v1/info` | 服务信息 |
| GET | `/health` | 健康检查 |

### POST /v1/generate

单帧全息图推理。

**请求体:**

```json
{
  "rgb": "<base64+zlib compressed RGB data>",
  "depth": "<base64+zlib compressed depth data>",
  "shape": [256, 256, 3],
  "depth_shape": [256, 256],
  "dtype": "uint8",
  "depth_dtype": "float32",
  "gpu_tier": "auto"
}
```

**响应:**

```json
{
  "phase": "<base64+zlib compressed phase data>",
  "shape": [256, 256],
  "dtype": "float32",
  "credits_used": 1,
  "latency_ms": 3.5
}
```

### POST /v1/generate_batch

批量全息图推理。

**请求体:**

```json
{
  "frames": [
    {"rgb": "...", "depth": "...", "shape": [256, 256, 3], "depth_shape": [256, 256]},
    {"rgb": "...", "depth": "...", "shape": [256, 256, 3], "depth_shape": [256, 256]}
  ],
  "gpu_tier": "a100"
}
```

**响应:**

```json
{
  "results": [
    {"phase": "...", "shape": [256, 256], "dtype": "float32"},
    {"phase": "...", "shape": [256, 256], "dtype": "float32"}
  ],
  "credits_used": 6,
  "latency_ms": 8.2
}
```

### GET /v1/credits

查询剩余推理额度。

**响应:**

```json
{
  "credits": 9500,
  "tier": "standard",
  "used_this_month": 500,
  "message": "ok"
}
```

### GET /v1/info

获取服务信息。

**响应:**

```json
{
  "service": "axcgh-cloud-api",
  "version": "1.0.0",
  "gpu_tiers": ["t4", "a10", "a100"],
  "pricing": {"t4": 0.01, "a10": 0.02, "a100": 0.03},
  "status": "ok"
}
```

## 定价

| GPU 层级 | 每帧价格 | 典型延迟 (256x256) | 典型延迟 (512x512) |
|----------|----------|-------------------|-------------------|
| T4 | $0.01 | ~5ms | ~15ms |
| A10 | $0.02 | ~3ms | ~8ms |
| A100 | $0.03 | ~2ms | ~4ms |

> 延迟为 TensorRT FP16 推理时间，不含网络传输。

### 额度消耗

| GPU 层级 | 每帧消耗额度 |
|----------|-------------|
| T4 | 1 credit |
| A10 | 2 credits |
| A100 | 3 credits |

## 快速开始

### 1. 获取 API Key

注册 [axcgh.com](https://axcgh.com) 账户，在控制台创建 API Key。

### 2. 安装 SDK

```bash
pip install axcgh-engine-pro
```

### 3. 调用 API

```python
from axcgh_pro import CloudEngineClient
import numpy as np

# 连接云服务
client = CloudEngineClient()
result = client.connect(
    api_key="axcgh-api-xxxxxxxxxxxx",
    endpoint="https://api.axcgh.com",
)
print(f"连接状态: {result}")

# 单帧推理
rgb = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
depth = np.random.rand(256, 256).astype(np.float32)

status, phase = client.generate_hologram(rgb, depth, gpu_tier="a100")
if status == "ok":
    print(f"相位图形状: {phase.shape}")

# 批量推理
frames = [(rgb, depth)] * 4
status, phases = client.generate_hologram_batch(frames, gpu_tier="a100")
if status == "ok":
    print(f"批量结果: {len(phases)} 帧")

# 查询额度
credits = client.get_credits()
print(f"剩余额度: {credits['credits']}")
```

### 4. 直接 HTTP 调用

```bash
# 查询服务信息
curl -H "X-API-Key: axcgh-api-xxxxx" https://api.axcgh.com/v1/info

# 查询额度
curl -H "X-API-Key: axcgh-api-xxxxx" https://api.axcgh.com/v1/credits
```

## 自部署

### Docker Compose

```bash
# 设置环境变量
export AXCGH_API_KEYS="your-key:standard:10000:t4"

# 启动服务
docker-compose up -d

# 启动 A100 工作器 (可选)
docker-compose --profile a100 up -d gpu-worker-a100
```

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `AXCGH_MODEL_PATH` | ONNX 模型路径 | `models/deepcgh_unet.onnx` |
| `AXCGH_MODEL_HEIGHT` | 模型输入高度 | `256` |
| `AXCGH_MODEL_WIDTH` | 模型输入宽度 | `256` |
| `AXCGH_GPU_CONFIG` | GPU 配置 (格式: `gpu_id:tier,...`) | `0:t4` |
| `AXCGH_API_KEYS` | API Key 列表 (格式: `key:tier:credits:gpu_tier,...`) | - |
| `REDIS_URL` | Redis 连接 URL | - |
| `DATABASE_URL` | PostgreSQL 连接 URL | - |

### 架构

```
                    ┌─────────────┐
                    │   Client    │
                    └──────┬──────┘
                           │ HTTPS
                    ┌──────▼──────┐
                    │  API Server │ (FastAPI + Uvicorn)
                    │  Rate Limit │
                    │  Auth       │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──┐  ┌─────▼───┐  ┌────▼────┐
       │ GPU T4  │  │ GPU A10 │  │GPU A100 │
       │ Worker  │  │ Worker  │  │ Worker  │
       └─────────┘  └─────────┘  └─────────┘

       ┌─────────────────────────────────────┐
       │  Redis: Rate Limiting + Job Queue   │
       └─────────────────────────────────────┘
       ┌─────────────────────────────────────┐
       │  PostgreSQL: Usage & Billing        │
       └─────────────────────────────────────┘
```

## 限流

| GPU 层级 | 请求限制 |
|----------|---------|
| T4 | 60 req/min |
| A10 | 120 req/min |
| A100 | 300 req/min |

## 错误码

| HTTP 状态码 | 说明 |
|-------------|------|
| 401 | API Key 无效 |
| 402 | 额度不足 |
| 429 | 请求频率超限 |
| 400 | 输入数据格式错误 |
| 503 | GPU 工作器不可用 |
