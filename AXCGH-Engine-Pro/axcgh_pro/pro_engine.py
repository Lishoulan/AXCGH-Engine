"""
axcgh_pro/pro_engine.py — Pro 版本高性能全息渲染引擎。

扩展 GPUEngineAPI:
  - TensorRT FP16/INT8 自动优化
  - 多 GPU 支持 (帧级分发)
  - 动态批处理推理
  - 流式推理 (Ring Buffer)
  - 自动调优: 寻找最优 batch size 和精度
  - 模型缓存: TRT 引擎持久化到磁盘
"""

import os
import time
import hashlib
import threading
import queue
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("onnxruntime-gpu is required for Pro engine: pip install onnxruntime-gpu>=1.16")

from deepcgh_engine import (
    GPUEngineAPI,
    GPUConfig,
    EngineConfig,
    Status,
    PhaseFormat,
)

from .license import LicenseManager


# ===========================================================================
# Pro Configuration
# ===========================================================================

@dataclass
class ProConfig:
    """Pro 引擎专用配置。"""
    # TensorRT 优化
    trt_fp16: bool = True
    trt_int8: bool = False
    trt_int8_calibration_data: Optional[str] = None  # INT8 校准数据路径
    trt_max_workspace_mb: int = 4096
    trt_cache_dir: str = ".trt_cache"  # TRT 引擎缓存目录

    # 多 GPU
    multi_gpu: bool = False
    gpu_ids: List[int] = field(default_factory=lambda: [0])

    # 批处理
    max_batch_size: int = 8
    dynamic_batch: bool = True  # 动态调整 batch size

    # 流式推理
    stream_buffer_size: int = 16
    stream_num_workers: int = 2

    # 自动调优
    auto_tune: bool = False
    auto_tune_warmup: int = 3
    auto_tune_runs: int = 20

    # 模型缓存
    enable_model_cache: bool = True


# ===========================================================================
# Multi-GPU Worker
# ===========================================================================

class _GPUWorker:
    """单个 GPU 上的推理工作器。"""

    def __init__(self, gpu_id: int, model_path: str, config: EngineConfig,
                 pro_config: ProConfig):
        self.gpu_id = gpu_id
        self.model_path = model_path
        self.config = config
        self.pro_config = pro_config
        self.engine: Optional[GPUEngineAPI] = None
        self._lock = threading.Lock()

    def start(self) -> Status:
        """初始化该 GPU 上的推理引擎。"""
        gpu_config = GPUConfig(
            provider_priority=['tensorrt', 'cuda', 'cpu'],
            enable_fp16=self.pro_config.trt_fp16,
            max_batch_size=self.pro_config.max_batch_size,
            device_id=self.gpu_id,
            trt_fp16=self.pro_config.trt_fp16,
            trt_int8=self.pro_config.trt_int8,
            trt_max_workspace_size_mb=self.pro_config.trt_max_workspace_mb,
            trt_cache_path=self._get_trt_cache_path(),
            preallocate_io_bindings=True,
        )

        self.engine = GPUEngineAPI(gpu_config)
        status = self.engine.init(self.model_path, self.config)
        return status

    def stop(self):
        """释放资源。"""
        if self.engine:
            self.engine.shutdown()
            self.engine = None

    def generate(self, rgb: np.ndarray, depth: np.ndarray,
                 benchmark: bool = False) -> Tuple[Status, Optional[np.ndarray]]:
        """单帧推理。"""
        if not self.engine or not self.engine.is_ready():
            return Status.NOT_INITIALIZED, None
        with self._lock:
            return self.engine.generate_hologram(rgb, depth, benchmark=benchmark)

    def generate_batch(self, rgb_list: List[np.ndarray],
                       depth_list: List[np.ndarray],
                       benchmark: bool = False) -> Tuple[Status, Optional[List[np.ndarray]]]:
        """批量推理。"""
        if not self.engine or not self.engine.is_ready():
            return Status.NOT_INITIALIZED, None
        with self._lock:
            return self.engine.generate_hologram_batch(rgb_list, depth_list, benchmark=benchmark)

    def _get_trt_cache_path(self) -> str:
        """获取该 GPU 的 TRT 缓存路径。"""
        cache_dir = self.pro_config.trt_cache_dir
        if self.pro_config.enable_model_cache:
            os.makedirs(cache_dir, exist_ok=True)
            return os.path.join(cache_dir, f"gpu{self.gpu_id}")
        return ""


# ===========================================================================
# Streaming Ring Buffer
# ===========================================================================

class _StreamBuffer:
    """流式推理的 Ring Buffer 实现。"""

    def __init__(self, engine: "ProEngineAPI", buffer_size: int = 16,
                 num_workers: int = 2):
        self.engine = engine
        self.buffer_size = buffer_size
        self._input_queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._output_queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._running = False
        self._workers: List[threading.Thread] = []
        self._num_workers = num_workers

    def start(self):
        """启动流式推理工作线程。"""
        self._running = True
        for i in range(self._num_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True, name=f"stream-worker-{i}")
            t.start()
            self._workers.append(t)

    def stop(self):
        """停止流式推理。"""
        self._running = False
        # 清空队列以解除阻塞
        while not self._input_queue.empty():
            try:
                self._input_queue.get_nowait()
            except queue.Empty:
                break
        while not self._output_queue.empty():
            try:
                self._output_queue.get_nowait()
            except queue.Empty:
                break
        for t in self._workers:
            t.join(timeout=5.0)
        self._workers.clear()

    def push(self, rgb: np.ndarray, depth: np.ndarray, frame_id: Optional[int] = None):
        """推入一帧到 Ring Buffer。"""
        if frame_id is None:
            frame_id = int(time.perf_counter() * 1000)
        self._input_queue.put((frame_id, rgb, depth), timeout=30.0)

    def pop(self, timeout: float = 1.0) -> Optional[Tuple[int, np.ndarray]]:
        """获取推理结果。返回 (frame_id, phase) 或 None。"""
        try:
            return self._output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _worker_loop(self):
        """工作线程主循环。"""
        while self._running:
            try:
                frame_id, rgb, depth = self._input_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            status, phase = self.engine.generate_hologram(rgb, depth)
            if status == Status.OK and phase is not None:
                self._output_queue.put((frame_id, phase), timeout=30.0)
            else:
                self._output_queue.put((frame_id, None), timeout=30.0)

    @property
    def pending_count(self) -> int:
        return self._input_queue.qsize()

    @property
    def ready_count(self) -> int:
        return self._output_queue.qsize()


# ===========================================================================
# Auto-Tuner
# ===========================================================================

class _AutoTuner:
    """自动调优: 寻找最优 batch size 和精度。"""

    def __init__(self, engine: "ProEngineAPI", config: EngineConfig):
        self.engine = engine
        self.config = config

    def tune(self, model_path: str, warmup: int = 3, runs: int = 20) -> Dict[str, Any]:
        """执行自动调优。

        Returns:
            {
                "best_batch_size": int,
                "best_precision": str,  # "fp16" or "int8"
                "results": {
                    "fp16_batch1": {"mean_ms": float, "fps": float},
                    ...
                }
            }
        """
        H, W = self.config.height, self.config.width
        rgb = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        depth = np.random.rand(H, W).astype(np.float32)

        results = {}
        best_config = {"batch_size": 1, "precision": "fp16", "mean_ms": float('inf')}

        # 测试不同精度和 batch size 组合
        for precision in ["fp16", "int8"]:
            for batch_size in [1, 2, 4, 8]:
                key = f"{precision}_batch{batch_size}"

                # 创建测试引擎
                pro_cfg = ProConfig(
                    trt_fp16=(precision == "fp16"),
                    trt_int8=(precision == "int8"),
                    max_batch_size=batch_size,
                    multi_gpu=False,
                    gpu_ids=[self.engine._pro_config.gpu_ids[0]] if self.engine._pro_config.gpu_ids else [0],
                )

                gpu_config = GPUConfig(
                    provider_priority=['tensorrt', 'cuda', 'cpu'],
                    enable_fp16=(precision == "fp16"),
                    max_batch_size=batch_size,
                    trt_fp16=(precision == "fp16"),
                    trt_int8=(precision == "int8"),
                    trt_max_workspace_size_mb=4096,
                    preallocate_io_bindings=True,
                )

                test_engine = GPUEngineAPI(gpu_config)
                status = test_engine.init(model_path, self.config)

                if status != Status.OK:
                    results[key] = {"status": "failed", "mean_ms": None, "fps": None}
                    continue

                # 准备批量数据
                rgb_list = [rgb] * batch_size
                depth_list = [depth] * batch_size

                # Warmup
                for _ in range(warmup):
                    if batch_size == 1:
                        test_engine.generate_hologram(rgb, depth)
                    else:
                        test_engine.generate_hologram_batch(rgb_list, depth_list)

                # Benchmark
                latencies = []
                for _ in range(runs):
                    t0 = time.perf_counter()
                    if batch_size == 1:
                        test_engine.generate_hologram(rgb, depth)
                    else:
                        test_engine.generate_hologram_batch(rgb_list, depth_list)
                    elapsed = (time.perf_counter() - t0) * 1000
                    latencies.append(elapsed / batch_size)  # 每帧延迟

                test_engine.shutdown()

                mean_ms = float(np.mean(latencies))
                fps = 1000.0 / mean_ms if mean_ms > 0 else 0

                results[key] = {
                    "status": "ok",
                    "mean_ms": round(mean_ms, 3),
                    "fps": round(fps, 1),
                }

                if mean_ms < best_config["mean_ms"]:
                    best_config = {
                        "batch_size": batch_size,
                        "precision": precision,
                        "mean_ms": mean_ms,
                    }

        return {
            "best_batch_size": best_config["batch_size"],
            "best_precision": best_config["precision"],
            "best_mean_ms": round(best_config["mean_ms"], 3),
            "results": results,
        }


# ===========================================================================
# ProEngineAPI
# ===========================================================================

class ProEngineAPI:
    """
    Pro 版本高性能全息渲染引擎。

    扩展 GPUEngineAPI:
      - TensorRT FP16/INT8 自动优化
      - 多 GPU 支持 (帧级分发)
      - 动态批处理推理
      - 流式推理 (Ring Buffer)
      - 自动调优
      - 模型缓存

    Usage:
        engine = ProEngineAPI()
        engine.init("model.onnx", EngineConfig(height=256, width=256), auto_tune=True)
        status, phase = engine.generate_hologram(rgb, depth)
        engine.shutdown()
    """

    def __init__(self, pro_config: Optional[ProConfig] = None,
                 multi_gpu: bool = False, gpu_ids: Optional[List[int]] = None):
        self._pro_config = pro_config or ProConfig()
        if multi_gpu:
            self._pro_config.multi_gpu = True
        if gpu_ids is not None:
            self._pro_config.gpu_ids = gpu_ids

        self._license = LicenseManager()
        self._initialized = False
        self._model_path: Optional[str] = None
        self._config: Optional[EngineConfig] = None
        self._last_error = ""

        # 单 GPU 模式
        self._engine: Optional[GPUEngineAPI] = None

        # 多 GPU 模式
        self._workers: List[_GPUWorker] = []
        self._worker_pool: Optional[ThreadPoolExecutor] = None
        self._next_gpu: int = 0  # 轮询调度计数器

        # 流式推理
        self._stream_buffer: Optional[_StreamBuffer] = None
        self._streaming = False

        # 自动调优结果
        self._tune_result: Optional[Dict[str, Any]] = None

        # 性能统计
        self._perf_stats: Dict[str, list] = {
            'preprocess': [], 'inference': [], 'postprocess': [],
            'total': [], 'batch_total': [],
        }

    def init(self, model_path: str, config: Optional[EngineConfig] = None,
             auto_tune: bool = False) -> Status:
        """初始化 Pro 引擎。

        Args:
            model_path: ONNX 模型路径
            config: 引擎配置
            auto_tune: 是否自动调优 (寻找最优 batch size 和精度)

        Returns:
            Status.OK 或错误状态
        """
        # 许可证检查
        if not self._license.is_valid():
            self._last_error = "许可证未激活，Pro 功能不可用"
            return Status.NOT_INITIALIZED

        if config is None:
            config = EngineConfig()

        try:
            config.validate()
        except AssertionError as e:
            self._last_error = str(e)
            return Status.INVALID_INPUT

        self._config = config
        self._model_path = model_path

        # 确保 TRT 缓存目录存在
        if self._pro_config.enable_model_cache:
            os.makedirs(self._pro_config.trt_cache_dir, exist_ok=True)

        if self._pro_config.multi_gpu and len(self._pro_config.gpu_ids) > 1:
            # 多 GPU 模式
            status = self._init_multi_gpu(model_path, config)
        else:
            # 单 GPU 模式
            status = self._init_single_gpu(model_path, config)

        if status != Status.OK:
            return status

        self._initialized = True

        # 自动调优
        if auto_tune:
            self._run_auto_tune(model_path, config)

        return Status.OK

    def _init_single_gpu(self, model_path: str, config: EngineConfig) -> Status:
        """初始化单 GPU 引擎。"""
        gpu_config = GPUConfig(
            provider_priority=['tensorrt', 'cuda', 'cpu'],
            enable_fp16=self._pro_config.trt_fp16,
            max_batch_size=self._pro_config.max_batch_size,
            device_id=self._pro_config.gpu_ids[0] if self._pro_config.gpu_ids else 0,
            trt_fp16=self._pro_config.trt_fp16,
            trt_int8=self._pro_config.trt_int8,
            trt_max_workspace_size_mb=self._pro_config.trt_max_workspace_mb,
            trt_cache_path=self._get_trt_cache_path(0),
            preallocate_io_bindings=True,
        )

        self._engine = GPUEngineAPI(gpu_config)
        status = self._engine.init(model_path, config)

        if status != Status.OK:
            self._last_error = f"引擎初始化失败: {self._engine.last_error}"
            return status

        provider = self._engine.active_provider.upper()
        fp16 = "FP16" if self._engine.gpu_uses_fp16 else "FP32"
        print(f"[ProEngine] Provider: {provider} | Precision: {fp16} | "
              f"Batch: {self._pro_config.max_batch_size}")

        return Status.OK

    def _init_multi_gpu(self, model_path: str, config: EngineConfig) -> Status:
        """初始化多 GPU 工作器。"""
        self._workers = []
        self._worker_pool = ThreadPoolExecutor(
            max_workers=len(self._pro_config.gpu_ids),
            thread_name_prefix="gpu-worker",
        )

        for gpu_id in self._pro_config.gpu_ids:
            worker = _GPUWorker(gpu_id, model_path, config, self._pro_config)
            status = worker.start()
            if status != Status.OK:
                # 清理已启动的工作器
                for w in self._workers:
                    w.stop()
                self._workers.clear()
                self._last_error = f"GPU {gpu_id} 初始化失败"
                return status
            self._workers.append(worker)

        print(f"[ProEngine] Multi-GPU: {len(self._workers)} GPUs "
              f"({self._pro_config.gpu_ids})")

        return Status.OK

    def _get_trt_cache_path(self, gpu_id: int) -> str:
        """获取 TRT 引擎缓存路径。"""
        if not self._pro_config.enable_model_cache:
            return ""

        cache_dir = self._pro_config.trt_cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # 基于模型路径 + GPU ID 生成唯一缓存键
        model_hash = hashlib.md5(
            f"{self._model_path or 'model'}_gpu{gpu_id}".encode()
        ).hexdigest()[:12]

        return os.path.join(cache_dir, f"gpu{gpu_id}_{model_hash}")

    def _run_auto_tune(self, model_path: str, config: EngineConfig):
        """执行自动调优并应用最优配置。"""
        print("[ProEngine] 自动调优中...")
        tuner = _AutoTuner(self, config)
        self._tune_result = tuner.tune(
            model_path,
            warmup=self._pro_config.auto_tune_warmup,
            runs=self._pro_config.auto_tune_runs,
        )

        best = self._tune_result
        print(f"[ProEngine] 调优结果: 最佳配置 batch={best['best_batch_size']}, "
              f"precision={best['best_precision']}, "
              f"latency={best['best_mean_ms']:.3f}ms")

        # 如果最优配置与当前不同，重新初始化
        if (best['best_batch_size'] != self._pro_config.max_batch_size or
                best['best_precision'] != ("fp16" if self._pro_config.trt_fp16 else "int8")):
            self._pro_config.max_batch_size = best['best_batch_size']
            self._pro_config.trt_fp16 = (best['best_precision'] == "fp16")
            self._pro_config.trt_int8 = (best['best_precision'] == "int8")

            # 重新初始化引擎
            self.shutdown()
            self._init_single_gpu(model_path, config)
            self._initialized = True

    # -----------------------------------------------------------------------
    # Inference API
    # -----------------------------------------------------------------------

    def generate_hologram(self, rgb: np.ndarray, depth: np.ndarray,
                          benchmark: bool = False) -> Tuple[Status, Optional[np.ndarray]]:
        """生成全息相位图 (单帧)。

        Args:
            rgb: uint8 [H, W, 3]
            depth: float32 [H, W]
            benchmark: 是否记录性能数据

        Returns:
            (Status, float32 [H, W] 相位图) 或 (Status, None)
        """
        if not self._initialized:
            self._last_error = "引擎未初始化"
            return Status.NOT_INITIALIZED, None

        if self._pro_config.multi_gpu and self._workers:
            return self._generate_multi_gpu_single(rgb, depth, benchmark)

        if self._engine is None:
            self._last_error = "引擎未初始化"
            return Status.NOT_INITIALIZED, None

        return self._engine.generate_hologram(rgb, depth, benchmark=benchmark)

    def generate_hologram_batch(self, rgb_list: List[np.ndarray],
                                depth_list: List[np.ndarray],
                                benchmark: bool = False) -> Tuple[Status, Optional[List[np.ndarray]]]:
        """批量生成全息相位图。

        在多 GPU 模式下，帧会自动分发到不同 GPU。

        Args:
            rgb_list: uint8 [H, W, 3] 列表
            depth_list: float32 [H, W] 列表
            benchmark: 是否记录性能数据

        Returns:
            (Status, List[float32 [H, W]]) 或 (Status, None)
        """
        if not self._initialized:
            self._last_error = "引擎未初始化"
            return Status.NOT_INITIALIZED, None

        if len(rgb_list) != len(depth_list):
            self._last_error = "rgb_list 和 depth_list 长度不一致"
            return Status.INVALID_INPUT, None

        if self._pro_config.multi_gpu and self._workers:
            return self._generate_multi_gpu_batch(rgb_list, depth_list, benchmark)

        if self._engine is None:
            self._last_error = "引擎未初始化"
            return Status.NOT_INITIALIZED, None

        return self._engine.generate_hologram_batch(rgb_list, depth_list, benchmark=benchmark)

    def _generate_multi_gpu_single(self, rgb: np.ndarray, depth: np.ndarray,
                                   benchmark: bool) -> Tuple[Status, Optional[np.ndarray]]:
        """多 GPU 轮询调度单帧推理。"""
        worker = self._workers[self._next_gpu % len(self._workers)]
        self._next_gpu += 1
        return worker.generate(rgb, depth, benchmark=benchmark)

    def _generate_multi_gpu_batch(self, rgb_list: List[np.ndarray],
                                  depth_list: List[np.ndarray],
                                  benchmark: bool) -> Tuple[Status, Optional[List[np.ndarray]]]:
        """多 GPU 帧级分发批量推理。"""
        n_frames = len(rgb_list)
        n_gpus = len(self._workers)

        if n_gpus == 0:
            self._last_error = "无可用 GPU 工作器"
            return Status.NOT_INITIALIZED, None

        # 将帧均匀分配到各 GPU
        chunks = [[] for _ in range(n_gpus)]
        for i in range(n_frames):
            gpu_idx = i % n_gpus
            chunks[gpu_idx].append(i)

        # 并行推理
        futures = {}
        for gpu_idx, indices in enumerate(chunks):
            if not indices:
                continue
            worker = self._workers[gpu_idx]
            chunk_rgb = [rgb_list[i] for i in indices]
            chunk_depth = [depth_list[i] for i in indices]
            future = self._worker_pool.submit(
                worker.generate_batch, chunk_rgb, chunk_depth, benchmark
            )
            futures[gpu_idx] = (indices, future)

        # 收集结果
        results = [None] * n_frames
        for gpu_idx, (indices, future) in futures.items():
            try:
                status, phases = future.result(timeout=60.0)
                if status != Status.OK or phases is None:
                    self._last_error = f"GPU {gpu_idx} 推理失败"
                    return status, None
                for i, phase in zip(indices, phases):
                    results[i] = phase
            except Exception as e:
                self._last_error = f"GPU {gpu_idx} 推理异常: {e}"
                return Status.INFERENCE_FAILED, None

        return Status.OK, results

    # -----------------------------------------------------------------------
    # Streaming API
    # -----------------------------------------------------------------------

    def start_stream(self, buffer_size: int = 16, num_workers: int = 2):
        """启动流式推理模式。

        Args:
            buffer_size: Ring Buffer 大小
            num_workers: 推理工作线程数
        """
        if not self._initialized:
            raise RuntimeError("引擎未初始化")

        if self._streaming:
            self.stop_stream()

        self._stream_buffer = _StreamBuffer(self, buffer_size, num_workers)
        self._stream_buffer.start()
        self._streaming = True
        print(f"[ProEngine] 流式推理已启动 (buffer={buffer_size}, workers={num_workers})")

    def stop_stream(self):
        """停止流式推理。"""
        if self._stream_buffer:
            self._stream_buffer.stop()
            self._stream_buffer = None
        self._streaming = False
        print("[ProEngine] 流式推理已停止")

    def push_frame(self, rgb: np.ndarray, depth: np.ndarray,
                   frame_id: Optional[int] = None):
        """向流式推理推入一帧。

        Args:
            rgb: uint8 [H, W, 3]
            depth: float32 [H, W]
            frame_id: 帧标识 (可选)
        """
        if not self._streaming or not self._stream_buffer:
            raise RuntimeError("流式推理未启动")
        self._stream_buffer.push(rgb, depth, frame_id)

    def pop_result(self, timeout: float = 1.0) -> Optional[Tuple[int, Optional[np.ndarray]]]:
        """获取流式推理结果。

        Returns:
            (frame_id, phase) 或 None (超时)
        """
        if not self._streaming or not self._stream_buffer:
            return None
        return self._stream_buffer.pop(timeout=timeout)

    def is_streaming(self) -> bool:
        """是否正在流式推理。"""
        return self._streaming

    @property
    def stream_pending(self) -> int:
        """流式推理待处理帧数。"""
        if self._stream_buffer:
            return self._stream_buffer.pending_count
        return 0

    @property
    def stream_ready(self) -> int:
        """流式推理已就绪结果数。"""
        if self._stream_buffer:
            return self._stream_buffer.ready_count
        return 0

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def is_ready(self) -> bool:
        return self._initialized

    def is_licensed(self) -> bool:
        return self._license.is_valid()

    def shutdown(self):
        """释放所有资源。"""
        if self._streaming:
            self.stop_stream()

        if self._engine:
            self._engine.shutdown()
            self._engine = None

        for worker in self._workers:
            worker.stop()
        self._workers.clear()

        if self._worker_pool:
            self._worker_pool.shutdown(wait=True)
            self._worker_pool = None

        self._initialized = False

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def last_error(self) -> str:
        return self._last_error

    @property
    def config(self) -> Optional[EngineConfig]:
        return self._config

    @property
    def pro_config(self) -> ProConfig:
        return self._pro_config

    @property
    def tune_result(self) -> Optional[Dict[str, Any]]:
        return self._tune_result

    @property
    def num_gpus(self) -> int:
        if self._pro_config.multi_gpu and self._workers:
            return len(self._workers)
        return 1

    def get_perf_stats(self) -> Dict[str, Dict[str, float]]:
        """获取性能统计。"""
        if self._engine:
            return self._engine.get_perf_stats()
        return {}
