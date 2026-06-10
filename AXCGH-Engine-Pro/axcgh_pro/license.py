"""
axcgh_pro/license.py — Pro 版本许可证管理系统。

功能:
  - validate_key(key) — 在线/离线验证许可证密钥
  - check_feature(feature) — 检查许可证层级是否支持指定功能
  - get_license_info() — 获取许可证详情
  - 离线验证 (机器指纹)
  - 网络故障宽限期
"""

import os
import json
import time
import hashlib
import platform
import uuid
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LICENSE_SERVER_URL = "https://api.axcgh.com/v1/license"
CACHE_FILE = os.path.join(os.path.expanduser("~"), ".axcgh", "license.json")
GRACE_PERIOD_SECONDS = 72 * 3600  # 72 小时宽限期
VALIDATION_INTERVAL_SECONDS = 6 * 3600  # 每 6 小时重新验证一次

# 功能层级定义
TIER_FEATURES = {
    "community": {
        "cpu_inference", "cuda_inference", "slm_driver", "realtime_preview",
        "multi_wavelength",
    },
    "pro": {
        "cpu_inference", "cuda_inference", "slm_driver", "realtime_preview",
        "multi_wavelength", "tensorrt_fp16", "tensorrt_int8", "multi_gpu",
        "dynamic_batch", "streaming", "auto_tune", "model_cache",
        "cloud_api",
    },
    "enterprise": {
        "cpu_inference", "cuda_inference", "slm_driver", "realtime_preview",
        "multi_wavelength", "tensorrt_fp16", "tensorrt_int8", "multi_gpu",
        "dynamic_batch", "streaming", "auto_tune", "model_cache",
        "cloud_api", "private_deploy", "custom_model", "sla",
    },
}


# ---------------------------------------------------------------------------
# Machine Fingerprint
# ---------------------------------------------------------------------------

def _get_machine_fingerprint() -> str:
    """生成机器唯一指纹，用于离线验证。"""
    components = [
        platform.node(),
        platform.machine(),
        platform.processor(),
        str(uuid.getnode()),  # MAC 地址
        platform.system(),
    ]
    raw = "|".join(components)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# License Data
# ---------------------------------------------------------------------------

@dataclass
class LicenseInfo:
    """许可证信息。"""
    key: str = ""
    tier: str = "community"
    valid: bool = False
    expires_at: float = 0.0  # Unix timestamp
    machine_fingerprint: str = ""
    features: set = field(default_factory=set)
    last_validated: float = 0.0
    last_error: str = ""

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "tier": self.tier,
            "valid": self.valid,
            "expires_at": self.expires_at,
            "machine_fingerprint": self.machine_fingerprint,
            "features": sorted(list(self.features)),
            "last_validated": self.last_validated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LicenseInfo":
        info = cls()
        info.key = data.get("key", "")
        info.tier = data.get("tier", "community")
        info.valid = data.get("valid", False)
        info.expires_at = data.get("expires_at", 0.0)
        info.machine_fingerprint = data.get("machine_fingerprint", "")
        info.features = set(data.get("features", []))
        info.last_validated = data.get("last_validated", 0.0)
        return info


# ---------------------------------------------------------------------------
# LicenseManager
# ---------------------------------------------------------------------------

class LicenseManager:
    """
    Pro 许可证管理器。

    功能:
      - 在线验证许可证密钥
      - 离线验证 (机器指纹绑定)
      - 网络故障宽限期
      - 功能层级检查
      - 许可证缓存到本地文件
    """

    def __init__(self, cache_file: Optional[str] = None):
        self._cache_file = cache_file or CACHE_FILE
        self._info = LicenseInfo()
        self._machine_fp = _get_machine_fingerprint()
        self._load_cached_license()

    def validate_key(self, key: str) -> dict:
        """验证许可证密钥。

        优先在线验证，失败时尝试离线验证。

        Args:
            key: 许可证密钥，格式 AXCGH-PRO-XXXX-XXXX-XXXX-XXXX

        Returns:
            {"valid": bool, "tier": str, "message": str}
        """
        # 基本格式检查
        if not self._validate_key_format(key):
            return {"valid": False, "tier": "community", "message": "密钥格式无效"}

        # 尝试在线验证
        online_result = self._validate_online(key)
        if online_result is not None:
            self._info = LicenseInfo(
                key=key,
                tier=online_result.get("tier", "pro"),
                valid=True,
                expires_at=online_result.get("expires_at", 0),
                machine_fingerprint=self._machine_fp,
                features=set(TIER_FEATURES.get(
                    online_result.get("tier", "pro"), set()
                )),
                last_validated=time.time(),
            )
            self._save_cached_license()
            return {
                "valid": True,
                "tier": self._info.tier,
                "message": "许可证验证成功",
            }

        # 在线验证失败，尝试离线验证
        offline_result = self._validate_offline(key)
        if offline_result is not None:
            return offline_result

        return {
            "valid": False,
            "tier": "community",
            "message": "许可证验证失败: 无法连接服务器且无有效缓存",
        }

    def check_feature(self, feature: str) -> bool:
        """检查当前许可证是否支持指定功能。

        Args:
            feature: 功能名称，如 "tensorrt_fp16", "multi_gpu" 等

        Returns:
            bool: 是否支持
        """
        if not self._info.valid:
            # 未验证或许可证无效，仅允许 community 功能
            return feature in TIER_FEATURES.get("community", set())

        # 检查是否过期
        if self._info.expires_at > 0 and time.time() > self._info.expires_at:
            return feature in TIER_FEATURES.get("community", set())

        # 检查宽限期
        if self._is_in_grace_period():
            return feature in self._info.features

        return feature in self._info.features

    def get_license_info(self) -> dict:
        """获取许可证详情。"""
        info = self._info.to_dict()
        info["machine_fingerprint"] = self._machine_fp
        info["in_grace_period"] = self._is_in_grace_period()
        info["grace_remaining_hours"] = self._grace_remaining_hours()
        info["needs_revalidation"] = self._needs_revalidation()
        return info

    def is_valid(self) -> bool:
        """检查许可证是否有效。"""
        if not self._info.valid:
            return False

        # 检查过期
        if self._info.expires_at > 0 and time.time() > self._info.expires_at:
            return False

        # 检查宽限期
        if self._is_in_grace_period():
            return True

        # 检查是否需要重新验证
        if self._needs_revalidation():
            # 尝试在线重新验证
            result = self._validate_online(self._info.key)
            if result is not None:
                self._info.last_validated = time.time()
                self._save_cached_license()
                return True
            # 在宽限期内仍然有效
            if self._is_in_grace_period():
                return True
            return False

        return True

    def has_cached_license(self) -> bool:
        """检查是否有缓存的许可证。"""
        return self._info.valid and bool(self._info.key)

    def invalidate(self):
        """使当前许可证失效。"""
        self._info.valid = False
        self._info.tier = "community"
        self._info.features = set()
        self._save_cached_license()

    # -------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------

    def _validate_key_format(self, key: str) -> bool:
        """验证密钥格式。"""
        if not key:
            return False
        parts = key.split("-")
        if len(parts) != 6:
            return False
        if parts[0] != "AXCGH":
            return False
        if parts[1] not in ("PRO", "ENT"):
            return False
        for part in parts[2:]:
            if len(part) != 4 or not part.isalnum():
                return False
        return True

    def _validate_online(self, key: str) -> Optional[dict]:
        """在线验证许可证密钥。"""
        try:
            import requests
            resp = requests.post(
                f"{LICENSE_SERVER_URL}/validate",
                json={
                    "key": key,
                    "machine_fingerprint": self._machine_fp,
                },
                timeout=10.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("valid", False):
                    return {
                        "tier": data.get("tier", "pro"),
                        "expires_at": data.get("expires_at", 0),
                    }
            return None
        except Exception:
            return None

    def _validate_offline(self, key: str) -> Optional[dict]:
        """离线验证 (基于缓存和机器指纹)。"""
        # 检查缓存
        if not self._info.valid or self._info.key != key:
            return None

        # 检查机器指纹
        if self._info.machine_fingerprint != self._machine_fp:
            return {
                "valid": False,
                "tier": "community",
                "message": "机器指纹不匹配，许可证绑定到其他设备",
            }

        # 检查过期
        if self._info.expires_at > 0 and time.time() > self._info.expires_at:
            return {
                "valid": False,
                "tier": "community",
                "message": "许可证已过期",
            }

        # 检查宽限期
        if self._is_in_grace_period():
            return {
                "valid": True,
                "tier": self._info.tier,
                "message": "离线验证成功 (宽限期内)",
            }

        # 上次验证时间过久
        elapsed = time.time() - self._info.last_validated
        if elapsed > GRACE_PERIOD_SECONDS:
            return {
                "valid": False,
                "tier": "community",
                "message": f"离线时间过长 ({elapsed / 3600:.1f} 小时)，需在线验证",
            }

        return {
            "valid": True,
            "tier": self._info.tier,
            "message": "离线验证成功",
        }

    def _is_in_grace_period(self) -> bool:
        """是否在宽限期内。"""
        if not self._info.valid or self._info.last_validated <= 0:
            return False
        elapsed = time.time() - self._info.last_validated
        return elapsed < GRACE_PERIOD_SECONDS

    def _grace_remaining_hours(self) -> float:
        """宽限期剩余小时数。"""
        if not self._info.valid or self._info.last_validated <= 0:
            return 0.0
        elapsed = time.time() - self._info.last_validated
        remaining = GRACE_PERIOD_SECONDS - elapsed
        return max(0.0, remaining / 3600.0)

    def _needs_revalidation(self) -> bool:
        """是否需要重新在线验证。"""
        if not self._info.valid or self._info.last_validated <= 0:
            return True
        elapsed = time.time() - self._info.last_validated
        return elapsed > VALIDATION_INTERVAL_SECONDS

    def _load_cached_license(self):
        """从本地缓存加载许可证。"""
        try:
            cache_path = Path(self._cache_file)
            if cache_path.exists():
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._info = LicenseInfo.from_dict(data)
        except Exception:
            self._info = LicenseInfo()

    def _save_cached_license(self):
        """保存许可证到本地缓存。"""
        try:
            cache_path = Path(self._cache_file)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(self._info.to_dict(), f, indent=2)
        except Exception:
            pass
