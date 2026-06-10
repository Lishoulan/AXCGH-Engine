"""
AXCGH Cloud API Server — 全息图推理云服务。

FastAPI 服务端:
  - API Key 认证
  - 按 Key / 层级限流
  - POST /v1/generate — 单帧推理
  - POST /v1/generate_batch — 批量推理
  - GET /v1/credits — 额度查询
  - GET /v1/info — 服务信息
  - 用量追踪与计费
  - GPU 工作池 + 队列
  - 多 GPU 层级 (T4 / A10 / A100)
"""

import os
import time
import zlib
import base64
import hashlib
import asyncio
import logging
from typing import Optional, Dict, List, Any
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

try:
    import redis
except ImportError:
    redis = None

try:
    import psycopg2
except ImportError:
    psycopg2 = None

try:
    from deepcgh_engine import GPUEngineAPI, GPUConfig, EngineConfig, Status
except ImportError:
    GPUEngineAPI = None
    GPUConfig = None
    EngineConfig = None
    Status = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY_HEADER = "X-API-Key"
RATE_LIMIT_T4 = 60       # T4: 60 req/min
RATE_LIMIT_A10 = 120     # A10: 120 req/min
RATE_LIMIT_A100 = 300    # A100: 300 req/min
CREDITS_PER_FRAME_T4 = 1
CREDITS_PER_FRAME_A10 = 2
CREDITS_PER_FRAME_A100 = 3

logger = logging.getLogger("axcgh-cloud")


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    rgb: str = Field(..., description="Base64 + zlib compressed RGB data")
    depth: str = Field(..., description="Base64 + zlib compressed depth data")
    shape: List[int] = Field(..., description="RGB shape [H, W, 3]")
    depth_shape: List[int] = Field(..., description="Depth shape [H, W]")
    dtype: str = Field(default="uint8", description="RGB dtype")
    depth_dtype: str = Field(default="float32", description="Depth dtype")
    gpu_tier: str = Field(default="auto", description="GPU tier: t4/a10/a100/auto")


class BatchGenerateRequest(BaseModel):
    frames: List[Dict[str, Any]] = Field(..., description="List of frame data")
    gpu_tier: str = Field(default="auto", description="GPU tier: t4/a10/a100/auto")


class GenerateResponse(BaseModel):
    phase: str = Field(..., description="Base64 + zlib compressed phase data")
    shape: List[int] = Field(..., description="Phase shape [H, W]")
    dtype: str = Field(default="float32", description="Phase dtype")
    credits_used: int = Field(default=1, description="Credits consumed")
    latency_ms: float = Field(default=0, description="Inference latency in ms")


class BatchGenerateResponse(BaseModel):
    results: List[Dict[str, Any]] = Field(..., description="List of phase results")
    credits_used: int = Field(default=0, description="Total credits consumed")
    latency_ms: float = Field(default=0, description="Total latency in ms")


class CreditsResponse(BaseModel):
    credits: int
    tier: str
    used_this_month: int
    message: str = "ok"


class InfoResponse(BaseModel):
    service: str = "axcgh-cloud-api"
    version: str = "1.0.0"
    gpu_tiers: List[str] = ["t4", "a10", "a100"]
    pricing: Dict[str, float] = {"t4": 0.01, "a10": 0.02, "a100": 0.03}
    status: str = "ok"


# ---------------------------------------------------------------------------
# GPU Worker Pool
# ---------------------------------------------------------------------------

@dataclass
class GPUWorker:
    """单个 GPU 推理工作器。"""
    gpu_id: int
    tier: str  # t4, a10, a100
    engine: Optional[GPUEngineAPI] = None
    busy: bool = False


class GPUWorkerPool:
    """GPU 工作池，管理多个 GPU 推理引擎。"""

    def __init__(self, model_path: str, config: EngineConfig):
        self.model_path = model_path
        self.config = config
        self._workers: Dict[str, List[GPUWorker]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=8)

    def add_worker(self, gpu_id: int, tier: str) -> bool:
        """添加一个 GPU 工作器。"""
        if GPUEngineAPI is None:
            logger.error("deepcgh_engine not available")
            return False

        gpu_config = GPUConfig(
            provider_priority=['tensorrt', 'cuda', 'cpu'],
            enable_fp16=True,
            max_batch_size=8,
            device_id=gpu_id,
            trt_fp16=True,
            trt_int8=False,
            trt_max_workspace_size_mb=4096,
            preallocate_io_bindings=True,
        )

        engine = GPUEngineAPI(gpu_config)
        status = engine.init(self.model_path, self.config)

        if status != Status.OK:
            logger.error(f"GPU {gpu_id} ({tier}) init failed: {engine.last_error}")
            return False

        worker = GPUWorker(gpu_id=gpu_id, tier=tier, engine=engine)
        self._workers[tier].append(worker)
        logger.info(f"GPU worker added: gpu={gpu_id}, tier={tier}")
        return True

    async def acquire(self, tier: str = "auto") -> Optional[GPUWorker]:
        """获取一个空闲的 GPU 工作器。"""
        async with self._lock:
            # 按优先级选择 tier
            tier_order = [tier] if tier != "auto" else ["a100", "a10", "t4"]
            for t in tier_order:
                for worker in self._workers.get(t, []):
                    if not worker.busy:
                        worker.busy = True
                        return worker
        return None

    async def release(self, worker: GPUWorker):
        """释放 GPU 工作器。"""
        async with self._lock:
            worker.busy = False

    async def infer(self, rgb: np.ndarray, depth: np.ndarray,
                    tier: str = "auto") -> Optional[Dict[str, Any]]:
        """执行单帧推理。"""
        worker = await self.acquire(tier)
        if worker is None:
            return None

        try:
            t0 = time.perf_counter()
            loop = asyncio.get_event_loop()
            status, phase = await loop.run_in_executor(
                self._executor,
                worker.engine.generate_hologram,
                rgb, depth,
            )
            latency = (time.perf_counter() - t0) * 1000

            if status != Status.OK or phase is None:
                return None

            return {
                "phase": phase,
                "latency_ms": latency,
                "tier": worker.tier,
                "gpu_id": worker.gpu_id,
            }
        finally:
            await self.release(worker)

    async def infer_batch(self, rgb_list: List[np.ndarray],
                          depth_list: List[np.ndarray],
                          tier: str = "auto") -> Optional[Dict[str, Any]]:
        """执行批量推理。"""
        worker = await self.acquire(tier)
        if worker is None:
            return None

        try:
            t0 = time.perf_counter()
            loop = asyncio.get_event_loop()
            status, phases = await loop.run_in_executor(
                self._executor,
                worker.engine.generate_hologram_batch,
                rgb_list, depth_list,
            )
            latency = (time.perf_counter() - t0) * 1000

            if status != Status.OK or phases is None:
                return None

            return {
                "phases": phases,
                "latency_ms": latency,
                "tier": worker.tier,
                "gpu_id": worker.gpu_id,
            }
        finally:
            await self.release(worker)

    def get_status(self) -> Dict[str, Any]:
        """获取工作池状态。"""
        status = {}
        for tier, workers in self._workers.items():
            total = len(workers)
            busy = sum(1 for w in workers if w.busy)
            status[tier] = {"total": total, "busy": busy, "idle": total - busy}
        return status

    def shutdown(self):
        """关闭所有工作器。"""
        for tier, workers in self._workers.items():
            for worker in workers:
                if worker.engine:
                    worker.engine.shutdown()
        self._executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """基于 Redis 或内存的限流器。"""

    def __init__(self, redis_client=None):
        self._redis = redis_client
        self._memory: Dict[str, List[float]] = defaultdict(list)

    def check(self, api_key: str, tier: str) -> bool:
        """检查是否在限流范围内。返回 True 表示允许。"""
        limits = {"t4": RATE_LIMIT_T4, "a10": RATE_LIMIT_A10, "a100": RATE_LIMIT_A100}
        limit = limits.get(tier, RATE_LIMIT_T4)

        key = f"rate:{api_key}:{tier}"
        now = time.time()
        window = 60.0  # 1 分钟窗口

        if self._redis:
            try:
                count = self._redis.incr(key)
                if count == 1:
                    self._redis.expire(key, 60)
                return count <= limit
            except Exception:
                pass

        # 内存回退
        timestamps = self._memory[key]
        self._memory[key] = [t for t in timestamps if now - t < window]
        self._memory[key].append(now)
        return len(self._memory[key]) <= limit


# ---------------------------------------------------------------------------
# API Key Store (简化版，生产环境应使用数据库)
# ---------------------------------------------------------------------------

@dataclass
class APIKeyInfo:
    key_hash: str
    tier: str = "standard"
    credits: int = 10000
    used_this_month: int = 0
    gpu_tier: str = "t4"


# 预置 API Key (生产环境应从数据库加载)
_API_KEYS: Dict[str, APIKeyInfo] = {}


def _load_api_keys():
    """从环境变量或数据库加载 API Key。"""
    keys_str = os.environ.get("AXCGH_API_KEYS", "")
    if keys_str:
        for entry in keys_str.split(","):
            parts = entry.strip().split(":")
            if len(parts) >= 2:
                key = parts[0]
                tier = parts[1] if len(parts) > 1 else "standard"
                credits = int(parts[2]) if len(parts) > 2 else 10000
                gpu_tier = parts[3] if len(parts) > 3 else "t4"
                key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
                _API_KEYS[key] = APIKeyInfo(
                    key_hash=key_hash, tier=tier, credits=credits, gpu_tier=gpu_tier
                )


_load_api_keys()


def validate_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> APIKeyInfo:
    """验证 API Key 依赖。"""
    if x_api_key not in _API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return _API_KEYS[x_api_key]


# ---------------------------------------------------------------------------
# Data Coding Helpers
# ---------------------------------------------------------------------------

def decode_input(data: Dict[str, Any]) -> tuple:
    """解码客户端发送的输入数据。"""
    rgb_bytes = zlib.decompress(base64.b64decode(data["rgb"]))
    depth_bytes = zlib.decompress(base64.b64decode(data["depth"]))
    rgb = np.frombuffer(rgb_bytes, dtype=np.dtype(data.get("dtype", "uint8"))).reshape(
        tuple(data["shape"])
    )
    depth = np.frombuffer(depth_bytes, dtype=np.dtype(data.get("depth_dtype", "float32"))).reshape(
        tuple(data["depth_shape"])
    )
    return rgb, depth


def encode_output(phase: np.ndarray) -> Dict[str, Any]:
    """编码输出数据。"""
    phase_bytes = zlib.compress(phase.tobytes())
    return {
        "phase": base64.b64encode(phase_bytes).decode("ascii"),
        "shape": list(phase.shape),
        "dtype": str(phase.dtype),
    }


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

worker_pool: Optional[GPUWorkerPool] = None
rate_limiter: Optional[RateLimiter] = None
redis_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。"""
    global worker_pool, rate_limiter, redis_client

    # 初始化 Redis (可选)
    redis_url = os.environ.get("REDIS_URL", "")
    if redis_url and redis:
        try:
            redis_client = redis.from_url(redis_url)
            redis_client.ping()
            logger.info("Redis connected")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, using memory fallback")
            redis_client = None

    rate_limiter = RateLimiter(redis_client)

    # 初始化 GPU Worker Pool
    model_path = os.environ.get("AXCGH_MODEL_PATH", "models/deepcgh_unet.onnx")
    model_h = int(os.environ.get("AXCGH_MODEL_HEIGHT", "256"))
    model_w = int(os.environ.get("AXCGH_MODEL_WIDTH", "256"))
    config = EngineConfig(height=model_h, width=model_w)

    worker_pool = GPUWorkerPool(model_path, config)

    # 从环境变量加载 GPU 配置
    gpu_config_str = os.environ.get("AXCGH_GPU_CONFIG", "0:t4")
    for entry in gpu_config_str.split(","):
        parts = entry.strip().split(":")
        if len(parts) >= 2:
            gpu_id = int(parts[0])
            tier = parts[1]
            worker_pool.add_worker(gpu_id, tier)

    logger.info(f"GPU Worker Pool initialized: {worker_pool.get_status()}")

    yield

    # 清理
    if worker_pool:
        worker_pool.shutdown()
    if redis_client:
        redis_client.close()


app = FastAPI(
    title="AXCGH Cloud API",
    description="全息图推理云服务 API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/v1/info", response_model=InfoResponse)
async def get_info():
    """获取服务信息。"""
    return InfoResponse()


@app.get("/v1/credits", response_model=CreditsResponse)
async def get_credits(api_key: APIKeyInfo = Depends(validate_api_key)):
    """查询剩余推理额度。"""
    return CreditsResponse(
        credits=api_key.credits,
        tier=api_key.tier,
        used_this_month=api_key.used_this_month,
    )


@app.post("/v1/generate", response_model=GenerateResponse)
async def generate_hologram(
    request: GenerateRequest,
    api_key: APIKeyInfo = Depends(validate_api_key),
):
    """单帧全息图推理。"""
    # 限流检查
    gpu_tier = request.gpu_tier if request.gpu_tier != "auto" else api_key.gpu_tier
    if rate_limiter and not rate_limiter.check(api_key.key_hash, gpu_tier):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # 额度检查
    credits_cost = {"t4": CREDITS_PER_FRAME_T4, "a10": CREDITS_PER_FRAME_A10,
                    "a100": CREDITS_PER_FRAME_A100}.get(gpu_tier, 1)
    if api_key.credits < credits_cost:
        raise HTTPException(status_code=402, detail="Insufficient credits")

    # 解码输入
    try:
        rgb, depth = decode_input(request.dict())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {e}")

    # 推理
    if worker_pool is None:
        raise HTTPException(status_code=503, detail="GPU worker pool not initialized")

    result = await worker_pool.infer(rgb, depth, tier=gpu_tier)
    if result is None:
        raise HTTPException(status_code=503, detail="No available GPU worker")

    # 编码输出
    phase = result["phase"]
    encoded = encode_output(phase)

    # 更新额度
    api_key.credits -= credits_cost
    api_key.used_this_month += credits_cost

    return GenerateResponse(
        phase=encoded["phase"],
        shape=encoded["shape"],
        dtype=encoded["dtype"],
        credits_used=credits_cost,
        latency_ms=result["latency_ms"],
    )


@app.post("/v1/generate_batch", response_model=BatchGenerateResponse)
async def generate_hologram_batch(
    request: BatchGenerateRequest,
    api_key: APIKeyInfo = Depends(validate_api_key),
):
    """批量全息图推理。"""
    gpu_tier = request.gpu_tier if request.gpu_tier != "auto" else api_key.gpu_tier
    n_frames = len(request.frames)

    # 限流
    if rate_limiter and not rate_limiter.check(api_key.key_hash, gpu_tier):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # 额度
    credits_cost = {"t4": CREDITS_PER_FRAME_T4, "a10": CREDITS_PER_FRAME_A10,
                    "a100": CREDITS_PER_FRAME_A100}.get(gpu_tier, 1) * n_frames
    if api_key.credits < credits_cost:
        raise HTTPException(status_code=402, detail="Insufficient credits")

    # 解码
    try:
        rgb_list = []
        depth_list = []
        for frame_data in request.frames:
            rgb, depth = decode_input(frame_data)
            rgb_list.append(rgb)
            depth_list.append(depth)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {e}")

    # 推理
    if worker_pool is None:
        raise HTTPException(status_code=503, detail="GPU worker pool not initialized")

    result = await worker_pool.infer_batch(rgb_list, depth_list, tier=gpu_tier)
    if result is None:
        raise HTTPException(status_code=503, detail="No available GPU worker")

    # 编码
    encoded_results = []
    for phase in result["phases"]:
        encoded_results.append(encode_output(phase))

    # 更新额度
    api_key.credits -= credits_cost
    api_key.used_this_month += credits_cost

    return BatchGenerateResponse(
        results=encoded_results,
        credits_used=credits_cost,
        latency_ms=result["latency_ms"],
    )


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """健康检查。"""
    pool_status = worker_pool.get_status() if worker_pool else {}
    return {
        "status": "ok",
        "gpu_pool": pool_status,
        "redis": "connected" if redis_client else "not_configured",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
