"""
axcgh_pro/cloud_client.py — AXCGH 云 API 客户端。

功能:
  - connect(api_key, endpoint) — 连接云服务
  - generate_hologram(rgb, depth) — 远程推理
  - generate_hologram_batch(frames) — 批量远程推理
  - get_credits() — 查询剩余额度
  - 自动重试 (指数退避)
  - 本地缓存 (重复输入)
"""

import io
import time
import zlib
import hashlib
import threading
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

import numpy as np

try:
    import requests
except ImportError:
    raise ImportError("requests is required for cloud client: pip install requests")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_ENDPOINT = "https://api.axcgh.com"
API_VERSION = "v1"
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 30.0
REQUEST_TIMEOUT_SECONDS = 120
CACHE_MAX_ENTRIES = 256


# ---------------------------------------------------------------------------
# Local Cache
# ---------------------------------------------------------------------------

class _LocalCache:
    """基于内容哈希的本地缓存，避免重复推理。"""

    def __init__(self, max_entries: int = CACHE_MAX_ENTRIES):
        self._max_entries = max_entries
        self._cache: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    def _compute_key(self, rgb: np.ndarray, depth: np.ndarray) -> str:
        """计算输入数据的哈希键。"""
        h = hashlib.sha256()
        h.update(rgb.tobytes())
        h.update(depth.tobytes())
        h.update(str(rgb.shape).encode())
        h.update(str(depth.shape).encode())
        return h.hexdigest()

    def get(self, rgb: np.ndarray, depth: np.ndarray) -> Optional[np.ndarray]:
        """查询缓存。"""
        key = self._compute_key(rgb, depth)
        with self._lock:
            return self._cache.get(key)

    def put(self, rgb: np.ndarray, depth: np.ndarray, result: np.ndarray):
        """存入缓存。"""
        key = self._compute_key(rgb, depth)
        with self._lock:
            if len(self._cache) >= self._max_entries:
                # 简单 LRU: 删除最早的一半
                keys = list(self._cache.keys())
                for k in keys[:len(keys) // 2]:
                    del self._cache[k]
            self._cache[key] = result.copy()

    def clear(self):
        """清空缓存。"""
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# CloudEngineClient
# ---------------------------------------------------------------------------

@dataclass
class CloudConnectionInfo:
    """云服务连接信息。"""
    api_key: str = ""
    endpoint: str = DEFAULT_ENDPOINT
    connected: bool = False
    tier: str = ""
    credits_remaining: int = 0


class CloudEngineClient:
    """
    AXCGH 云 API 客户端。

    功能:
      - 远程全息图推理
      - 批量推理
      - 额度查询
      - 自动重试 (指数退避)
      - 本地缓存 (重复输入)

    Usage:
        client = CloudEngineClient()
        client.connect(api_key="axcgh-api-xxxxx")
        status, phase = client.generate_hologram(rgb, depth)
    """

    def __init__(self, enable_cache: bool = True, max_retries: int = MAX_RETRIES):
        self._conn = CloudConnectionInfo()
        self._max_retries = max_retries
        self._cache = _LocalCache() if enable_cache else None
        self._last_error = ""
        self._session: Optional[requests.Session] = None

    def connect(self, api_key: str, endpoint: str = DEFAULT_ENDPOINT) -> dict:
        """连接云服务。

        Args:
            api_key: API 密钥，格式 axcgh-api-xxxxx
            endpoint: 云服务端点

        Returns:
            {"connected": bool, "tier": str, "credits": int, "message": str}
        """
        self._conn.api_key = api_key
        self._conn.endpoint = endpoint.rstrip("/")

        # 创建 HTTP 会话
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "AXCGH-CloudClient/1.0.0",
        })

        # 验证连接
        try:
            resp = self._request_with_retry(
                "GET", f"{self._conn.endpoint}/{API_VERSION}/info"
            )
            if resp and resp.status_code == 200:
                data = resp.json()
                self._conn.connected = True
                self._conn.tier = data.get("tier", "standard")
                self._conn.credits_remaining = data.get("credits", 0)
                return {
                    "connected": True,
                    "tier": self._conn.tier,
                    "credits": self._conn.credits_remaining,
                    "message": "连接成功",
                }
            else:
                self._conn.connected = False
                return {
                    "connected": False,
                    "tier": "",
                    "credits": 0,
                    "message": f"连接失败: HTTP {resp.status_code if resp else 'N/A'}",
                }
        except Exception as e:
            self._conn.connected = False
            return {
                "connected": False,
                "tier": "",
                "credits": 0,
                "message": f"连接异常: {e}",
            }

    def generate_hologram(self, rgb: np.ndarray, depth: np.ndarray,
                          gpu_tier: str = "auto") -> Tuple[str, Optional[np.ndarray]]:
        """远程单帧推理。

        Args:
            rgb: uint8 [H, W, 3]
            depth: float32 [H, W]
            gpu_tier: GPU 层级 "t4" / "a10" / "a100" / "auto"

        Returns:
            ("ok", float32 [H, W] 相位图) 或 ("error", None)
        """
        if not self._conn.connected:
            self._last_error = "未连接到云服务"
            return "error", None

        # 检查本地缓存
        if self._cache is not None:
            cached = self._cache.get(rgb, depth)
            if cached is not None:
                return "ok", cached

        # 序列化输入
        payload = self._encode_input(rgb, depth, gpu_tier)

        # 发送请求
        try:
            resp = self._request_with_retry(
                "POST",
                f"{self._conn.endpoint}/{API_VERSION}/generate",
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            if resp is None:
                self._last_error = "请求失败 (重试耗尽)"
                return "error", None

            if resp.status_code != 200:
                error_msg = resp.json().get("error", f"HTTP {resp.status_code}")
                self._last_error = error_msg
                return "error", None

            # 解码输出
            result = self._decode_output(resp.json())
            if result is None:
                self._last_error = "输出解码失败"
                return "error", None

            # 更新额度
            credits_used = resp.json().get("credits_used", 1)
            self._conn.credits_remaining = max(
                0, self._conn.credits_remaining - credits_used
            )

            # 存入缓存
            if self._cache is not None:
                self._cache.put(rgb, depth, result)

            return "ok", result

        except Exception as e:
            self._last_error = str(e)
            return "error", None

    def generate_hologram_batch(self, frames: List[Tuple[np.ndarray, np.ndarray]],
                                gpu_tier: str = "auto") -> Tuple[str, Optional[List[np.ndarray]]]:
        """批量远程推理。

        Args:
            frames: [(rgb, depth), ...] 列表
            gpu_tier: GPU 层级

        Returns:
            ("ok", [phase, ...]) 或 ("error", None)
        """
        if not self._conn.connected:
            self._last_error = "未连接到云服务"
            return "error", None

        if not frames:
            self._last_error = "空帧列表"
            return "error", None

        # 检查缓存，分离已缓存和未缓存的帧
        results = [None] * len(frames)
        uncached_indices = []
        uncached_frames = []

        if self._cache is not None:
            for i, (rgb, depth) in enumerate(frames):
                cached = self._cache.get(rgb, depth)
                if cached is not None:
                    results[i] = cached
                else:
                    uncached_indices.append(i)
                    uncached_frames.append((rgb, depth))
        else:
            uncached_indices = list(range(len(frames)))
            uncached_frames = frames

        if not uncached_frames:
            return "ok", results

        # 序列化批量输入
        payload = self._encode_batch_input(uncached_frames, gpu_tier)

        try:
            resp = self._request_with_retry(
                "POST",
                f"{self._conn.endpoint}/{API_VERSION}/generate_batch",
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS * 3,
            )

            if resp is None:
                self._last_error = "批量请求失败 (重试耗尽)"
                return "error", None

            if resp.status_code != 200:
                error_msg = resp.json().get("error", f"HTTP {resp.status_code}")
                self._last_error = error_msg
                return "error", None

            # 解码批量输出
            batch_results = self._decode_batch_output(resp.json())
            if batch_results is None:
                self._last_error = "批量输出解码失败"
                return "error", None

            # 合并结果
            for i, idx in enumerate(uncached_indices):
                if i < len(batch_results):
                    results[idx] = batch_results[i]
                    # 存入缓存
                    if self._cache is not None:
                        rgb, depth = uncached_frames[i]
                        self._cache.put(rgb, depth, batch_results[i])

            # 更新额度
            credits_used = resp.json().get("credits_used", len(uncached_frames))
            self._conn.credits_remaining = max(
                0, self._conn.credits_remaining - credits_used
            )

            return "ok", results

        except Exception as e:
            self._last_error = str(e)
            return "error", None

    def get_credits(self) -> dict:
        """查询剩余推理额度。

        Returns:
            {"credits": int, "tier": str, "message": str}
        """
        if not self._conn.connected:
            return {"credits": 0, "tier": "", "message": "未连接"}

        try:
            resp = self._request_with_retry(
                "GET", f"{self._conn.endpoint}/{API_VERSION}/credits"
            )
            if resp and resp.status_code == 200:
                data = resp.json()
                self._conn.credits_remaining = data.get("credits", 0)
                return {
                    "credits": self._conn.credits_remaining,
                    "tier": data.get("tier", self._conn.tier),
                    "message": "ok",
                }
            return {"credits": 0, "tier": "", "message": "查询失败"}
        except Exception as e:
            return {"credits": 0, "tier": "", "message": str(e)}

    def disconnect(self):
        """断开云服务连接。"""
        if self._session:
            self._session.close()
            self._session = None
        self._conn.connected = False

    # -------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------

    @property
    def last_error(self) -> str:
        return self._last_error

    @property
    def is_connected(self) -> bool:
        return self._conn.connected

    @property
    def credits_remaining(self) -> int:
        return self._conn.credits_remaining

    @property
    def cache_size(self) -> int:
        return self._cache.size if self._cache else 0

    # -------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------

    def _request_with_retry(self, method: str, url: str, **kwargs) -> Optional[requests.Response]:
        """带指数退避重试的 HTTP 请求。"""
        backoff = INITIAL_BACKOFF_SECONDS

        for attempt in range(self._max_retries):
            try:
                if self._session is None:
                    return None
                resp = self._session.request(method, url, **kwargs)
                # 5xx 错误重试
                if resp.status_code >= 500:
                    if attempt < self._max_retries - 1:
                        time.sleep(backoff)
                        backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
                        continue
                return resp
            except requests.exceptions.ConnectionError:
                if attempt < self._max_retries - 1:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
                    continue
                return None
            except requests.exceptions.Timeout:
                if attempt < self._max_retries - 1:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
                    continue
                return None
            except Exception:
                return None

        return None

    def _encode_input(self, rgb: np.ndarray, depth: np.ndarray,
                      gpu_tier: str) -> dict:
        """编码单帧输入为 JSON 可序列化格式。"""
        rgb_bytes = zlib.compress(rgb.tobytes())
        depth_bytes = zlib.compress(depth.tobytes())

        import base64
        return {
            "rgb": base64.b64encode(rgb_bytes).decode("ascii"),
            "depth": base64.b64encode(depth_bytes).decode("ascii"),
            "shape": list(rgb.shape),
            "depth_shape": list(depth.shape),
            "dtype": str(rgb.dtype),
            "depth_dtype": str(depth.dtype),
            "gpu_tier": gpu_tier,
        }

    def _encode_batch_input(self, frames: List[Tuple[np.ndarray, np.ndarray]],
                            gpu_tier: str) -> dict:
        """编码批量输入。"""
        import base64
        encoded_frames = []
        for rgb, depth in frames:
            rgb_bytes = zlib.compress(rgb.tobytes())
            depth_bytes = zlib.compress(depth.tobytes())
            encoded_frames.append({
                "rgb": base64.b64encode(rgb_bytes).decode("ascii"),
                "depth": base64.b64encode(depth_bytes).decode("ascii"),
                "shape": list(rgb.shape),
                "depth_shape": list(depth.shape),
                "dtype": str(rgb.dtype),
                "depth_dtype": str(depth.dtype),
            })
        return {
            "frames": encoded_frames,
            "gpu_tier": gpu_tier,
        }

    def _decode_output(self, data: dict) -> Optional[np.ndarray]:
        """解码单帧输出。"""
        import base64
        try:
            phase_bytes = zlib.decompress(base64.b64decode(data["phase"]))
            shape = tuple(data["shape"])
            dtype = np.dtype(data.get("dtype", "float32"))
            return np.frombuffer(phase_bytes, dtype=dtype).reshape(shape)
        except Exception:
            return None

    def _decode_batch_output(self, data: dict) -> Optional[List[np.ndarray]]:
        """解码批量输出。"""
        import base64
        try:
            results = []
            for item in data["results"]:
                phase_bytes = zlib.decompress(base64.b64decode(item["phase"]))
                shape = tuple(item["shape"])
                dtype = np.dtype(item.get("dtype", "float32"))
                results.append(np.frombuffer(phase_bytes, dtype=dtype).reshape(shape))
            return results
        except Exception:
            return None
