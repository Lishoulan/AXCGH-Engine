"""
deepcgh_engine/server.py — FastAPI REST API server for AXCGH-Engine.

Provides HTTP endpoints for holographic phase map generation from RGB-D inputs.

Endpoints:
  POST /generate       — Upload RGB-D numpy arrays, get phase map as JSON
  POST /generate_file  — Upload image files, get phase map as PNG
  GET  /health         — Health check
  GET  /info           — Engine info and configuration

Usage:
  python -m deepcgh_engine.server

  Or with uvicorn directly:
  uvicorn deepcgh_engine.server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import io
import os
import time
import logging
from typing import Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from .engine import (
    EngineAPI,
    EngineConfig,
    ExecutionProvider,
    ColorSpace,
    PhaseFormat,
    Status,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("axcgh-server")

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AXCGH-Engine API",
    description="Deep-learning Computer-Generated Holography Engine — "
                "RGB-D to SLM phase map via U-Net inference",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Global engine state
# ---------------------------------------------------------------------------
_engine: Optional[EngineAPI] = None
_engine_lock = asyncio.Lock()
_request_queue: asyncio.Queue = asyncio.Queue(maxsize=16)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    """Request body for /generate endpoint — raw numpy array data."""
    rgb: list = Field(..., description="Flattened uint8 RGB array [H*W*3]")
    depth: list = Field(..., description="Flattened float32 depth array [H*W]")
    height: int = Field(256, description="Frame height")
    width: int = Field(256, description="Frame width")


class PhaseMapResponse(BaseModel):
    """Response containing the generated phase map."""
    status: int = Field(..., description="Status code (0=OK)")
    phase: list = Field(..., description="Flattened float32 phase map [H*W]")
    height: int = Field(..., description="Phase map height")
    width: int = Field(..., description="Phase map width")
    elapsed_ms: float = Field(..., description="Generation time in ms")


class HealthResponse(BaseModel):
    status: str
    engine_ready: bool
    device: str
    uptime_s: float


class InfoResponse(BaseModel):
    version: str
    device: str
    model_path: str
    config: dict
    onnx_providers: list
    cupy_available: bool


# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------
_start_time: float = 0.0


@app.on_event("startup")
async def startup():
    global _engine, _start_time
    _start_time = time.time()

    device = os.environ.get("DEEPCGH_DEVICE", "cpu").lower()
    model_path = os.environ.get("DEEPCGH_MODEL_PATH", "models/deepcgh_unet.onnx")
    height = int(os.environ.get("DEEPCGH_HEIGHT", "256"))
    width = int(os.environ.get("DEEPCGH_WIDTH", "256"))
    num_planes = int(os.environ.get("DEEPCGH_NUM_PLANES", "5"))
    intra_threads = int(os.environ.get("DEEPCGH_INTRA_OP_THREADS", "4"))
    inter_threads = int(os.environ.get("DEEPCGH_INTER_OP_THREADS", "1"))
    use_cupy = os.environ.get("DEEPCGH_USE_CUPY", "0") == "1"

    provider = ExecutionProvider.CUDA if device == "gpu" else ExecutionProvider.CPU

    config = EngineConfig(
        height=height,
        width=width,
        num_planes=num_planes,
        provider=provider,
        intra_op_threads=intra_threads,
        inter_op_threads=inter_threads,
        use_cupy=use_cupy,
    )

    logger.info(f"Initializing AXCGH-Engine (device={device}, model={model_path})")

    _engine = EngineAPI()
    status = _engine.init(model_path, config)

    if status != Status.OK:
        logger.error(f"Engine initialization failed: {_engine.last_error}")
        _engine = None
    else:
        logger.info(
            f"Engine ready: {width}x{height}, {num_planes} planes, "
            f"provider={provider.value}, cupy={_engine.gpu_enabled}"
        )


@app.on_event("shutdown")
async def shutdown():
    global _engine
    if _engine is not None:
        _engine.shutdown()
        _engine = None
        logger.info("Engine shut down")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok" if _engine is not None and _engine.is_ready() else "unavailable",
        engine_ready=_engine is not None and _engine.is_ready(),
        device=os.environ.get("DEEPCGH_DEVICE", "cpu"),
        uptime_s=time.time() - _start_time,
    )


@app.get("/info", response_model=InfoResponse)
async def info():
    """Return engine information and configuration."""
    if _engine is None or not _engine.is_ready():
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
    except Exception:
        providers = []

    try:
        import cupy
        cupy_avail = True
    except ImportError:
        cupy_avail = False

    cfg = _engine.config
    config_dict = {
        "height": cfg.height,
        "width": cfg.width,
        "num_planes": cfg.num_planes,
        "provider": cfg.provider.value,
        "color_space": cfg.color_space.value,
        "phase_format": cfg.phase_format.value,
        "quantization_bits": cfg.quantization_bits,
        "wavelength_mm": cfg.wavelength,
        "pixel_size_mm": cfg.pixel_size,
        "focal_length_mm": cfg.focal_length,
        "plane_distance_mm": cfg.plane_distance,
        "gpu_enabled": _engine.gpu_enabled,
    }

    return InfoResponse(
        version="1.0.0",
        device=os.environ.get("DEEPCGH_DEVICE", "cpu"),
        model_path=os.environ.get("DEEPCGH_MODEL_PATH", "models/deepcgh_unet.onnx"),
        config=config_dict,
        onnx_providers=providers,
        cupy_available=cupy_avail,
    )


@app.post("/generate", response_model=PhaseMapResponse)
async def generate(request: GenerateRequest):
    """
    Generate a hologram phase map from RGB-D data.

    Accepts flattened numpy arrays (JSON-friendly) and returns
    the phase map as a flattened float32 array.
    """
    if _engine is None or not _engine.is_ready():
        raise HTTPException(status_code=503, detail="Engine not initialized")

    # Reconstruct arrays
    try:
        rgb = np.array(request.rgb, dtype=np.uint8).reshape(
            request.height, request.width, 3
        )
        depth = np.array(request.depth, dtype=np.float32).reshape(
            request.height, request.width
        )
    except (ValueError, TypeError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input data: {e}. "
                   f"Expected rgb[{request.height}*{request.width}*3] and "
                   f"depth[{request.height}*{request.width}]",
        )

    # Acquire lock for thread-safe inference
    async with _engine_lock:
        t0 = time.perf_counter()
        status, phase = _engine.generate_hologram(rgb, depth)
        elapsed = (time.perf_counter() - t0) * 1000

    if status != Status.OK:
        raise HTTPException(
            status_code=500,
            detail=f"Hologram generation failed (status={status}): {_engine.last_error}",
        )

    return PhaseMapResponse(
        status=int(status),
        phase=phase.flatten().tolist(),
        height=phase.shape[0],
        width=phase.shape[1],
        elapsed_ms=round(elapsed, 2),
    )


@app.post("/generate_file")
async def generate_file(
    rgb_file: UploadFile = File(..., description="RGB image file (PNG/JPG/BMP)"),
    depth_file: UploadFile = File(None, description="Depth map image file (optional)"),
    height: int = Query(256, description="Target height"),
    width: int = Query(256, description="Target width"),
    format: str = Query("png", description="Output format: png or npy"),
):
    """
    Generate a hologram phase map from uploaded image files.

    Upload an RGB image (and optionally a depth map) and receive
    the generated phase map as a downloadable PNG image.
    """
    if _engine is None or not _engine.is_ready():
        raise HTTPException(status_code=503, detail="Engine not initialized")

    # Read and resize RGB image
    try:
        rgb_bytes = await rgb_file.read()
        rgb_img = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
        rgb_img = rgb_img.resize((width, height), Image.BILINEAR)
        rgb = np.array(rgb_img, dtype=np.uint8)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid RGB image: {e}")

    # Read depth map or generate uniform depth
    if depth_file is not None:
        try:
            depth_bytes = await depth_file.read()
            depth_img = Image.open(io.BytesIO(depth_bytes)).convert("L")
            depth_img = depth_img.resize((width, height), Image.BILINEAR)
            depth = np.array(depth_img, dtype=np.float32) / 255.0
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid depth image: {e}")
    else:
        # Default: uniform depth (all planes at center)
        depth = np.ones((height, width), dtype=np.float32) * 0.5

    # Run inference
    async with _engine_lock:
        t0 = time.perf_counter()
        status, phase = _engine.generate_hologram(rgb, depth)
        elapsed = (time.perf_counter() - t0) * 1000

    if status != Status.OK:
        raise HTTPException(
            status_code=500,
            detail=f"Hologram generation failed (status={status}): {_engine.last_error}",
        )

    logger.info(f"Generated phase map: {phase.shape}, {elapsed:.2f} ms")

    # Return in requested format
    if format.lower() == "npy":
        buf = io.BytesIO()
        np.save(buf, phase)
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": "attachment; filename=phase_map.npy",
                "X-Elapsed-Ms": f"{elapsed:.2f}",
            },
        )
    else:
        # Convert phase [-pi, pi] -> [0, 255] uint8 PNG
        normalized = ((phase + np.pi) / (2 * np.pi) * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(normalized, mode="L")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="image/png",
            headers={
                "Content-Disposition": "attachment; filename=phase_map.png",
                "X-Elapsed-Ms": f"{elapsed:.2f}",
                "X-Phase-Height": str(phase.shape[0]),
                "X-Phase-Width": str(phase.shape[1]),
            },
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    """Run the server with uvicorn."""
    import uvicorn

    host = os.environ.get("DEEPCGH_HOST", "0.0.0.0")
    port = int(os.environ.get("DEEPCGH_PORT", "8000"))

    uvicorn.run(
        "deepcgh_engine.server:app",
        host=host,
        port=port,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
