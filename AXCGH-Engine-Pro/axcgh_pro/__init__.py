"""
AXCGH-Engine Pro — 高性能全息渲染引擎专业版

功能:
  - TensorRT FP16/INT8 自动优化
  - 多 GPU 并行推理 (帧级分发)
  - 动态批处理推理
  - 流式推理 (Ring Buffer)
  - 自动调优 (Batch Size / Precision)
  - TRT 引擎磁盘缓存
  - 云 API 远程推理

使用前必须验证许可证密钥。
"""

import os

from .license import LicenseManager
from .pro_engine import ProEngineAPI
from .cloud_client import CloudEngineClient

__version__ = "1.0.0"

# ---------------------------------------------------------------------------
# License guard — 拒绝无许可证加载
# ---------------------------------------------------------------------------

_license_manager = LicenseManager()
_license_validated = False


def _check_license():
    """检查许可证是否已验证，未验证则拒绝加载 Pro 功能。"""
    global _license_validated
    if _license_validated:
        return True

    # 尝试从环境变量自动验证
    env_key = os.environ.get("AXCGH_LICENSE_KEY", "")
    if env_key:
        result = _license_manager.validate_key(env_key)
        if result.get("valid", False):
            _license_validated = True
            return True

    # 尝试从已保存的许可证文件加载
    if _license_manager.has_cached_license():
        info = _license_manager.get_license_info()
        if info.get("valid", False):
            _license_validated = True
            return True

    return False


def activate(key: str) -> dict:
    """激活 Pro 许可证。

    Args:
        key: 许可证密钥，格式 AXCGH-PRO-XXXX-XXXX-XXXX-XXXX

    Returns:
        dict: {"valid": bool, "tier": str, "message": str}
    """
    global _license_validated
    result = _license_manager.validate_key(key)
    if result.get("valid", False):
        _license_validated = True
    return result


def is_activated() -> bool:
    """检查 Pro 许可证是否已激活。"""
    return _check_license()


def get_license_info() -> dict:
    """获取当前许可证信息。"""
    return _license_manager.get_license_info()


# ---------------------------------------------------------------------------
# Import-time license check — 打印警告但不阻止导入
# ---------------------------------------------------------------------------

if not _check_license():
    import warnings
    warnings.warn(
        "AXCGH-Engine Pro: 许可证未激活。请调用 axcgh_pro.activate(key) 或设置 "
        "AXCGH_LICENSE_KEY 环境变量。Pro 功能在未激活状态下不可用。",
        UserWarning,
        stacklevel=2,
    )


__all__ = [
    "ProEngineAPI",
    "CloudEngineClient",
    "LicenseManager",
    "activate",
    "is_activated",
    "get_license_info",
]
