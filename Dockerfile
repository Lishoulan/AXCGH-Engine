# ============================================================================
# AXCGH-Engine Dockerfile — Multi-stage build
# ============================================================================
# Supports both CPU and GPU (NVIDIA CUDA) variants via build args:
#
#   CPU build (default):
#     docker build -t axcgh-engine:cpu .
#
#   GPU build:
#     docker build --build-arg DEVICE=gpu -t axcgh-engine:gpu .
#
#   Run CPU:
#     docker run -p 8000:8000 axcgh-engine:cpu
#
#   Run GPU:
#     docker run --gpus all -p 8000:8000 axcgh-engine:gpu
# ============================================================================

# ---------------------------------------------------------------------------
# Build arguments
# ---------------------------------------------------------------------------
ARG DEVICE=cpu
ARG PYTHON_VERSION=3.10
ARG UBUNTU_VERSION=22.04

# ============================================================================
# Stage 1: Builder — Compile C++ engine with CMake
# ============================================================================
FROM python:${PYTHON_VERSION}-slim AS builder

ARG DEVICE

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    pkg-config \
    libfftw3-dev \
    libfftw3-single3 \
    && rm -rf /var/lib/apt/lists/*

# Install Python build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /build

# Copy C++ source and CMake configuration
COPY DeepCGHEngine/CMakeLists.txt ./
COPY DeepCGHEngine/include/ ./include/
COPY DeepCGHEngine/src/ ./src/
COPY DeepCGHEngine/bindings/ ./bindings/
COPY DeepCGHEngine/apps/ ./apps/
COPY DeepCGHEngine/cmake/ ./cmake/

# Build C++ engine (static library + optional Python bindings)
# ONNX Runtime SDK will be installed via pip in runtime stage
RUN mkdir -p build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DDEEPCGH_BUILD_PYTHON=OFF \
        -DDEEPCGH_BUILD_APPS=ON \
        -DDEEPCGH_BUILD_TESTS=OFF \
    && cmake --build . --config Release -j$(nproc)

# ============================================================================
# Stage 2: Runtime — Minimal image with Python + runtime deps
# ============================================================================
FROM python:${PYTHON_VERSION}-slim AS runtime-cpu

ARG DEVICE

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfftw3-single3 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY DeepCGHEngine/pyproject.toml ./DeepCGHEngine/pyproject.toml
COPY DeepCGHEngine/setup.py ./DeepCGHEngine/setup.py
COPY DeepCGHEngine/MANIFEST.in ./DeepCGHEngine/MANIFEST.in

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    onnxruntime>=1.16 \
    numpy>=1.20 \
    Pillow>=9.0 \
    fastapi>=0.100 \
    uvicorn[standard]>=0.23 \
    python-multipart>=0.0.6

# Copy Python package
COPY DeepCGHEngine/deepcgh_engine/ ./DeepCGHEngine/deepcgh_engine/

# Copy C++ built artifacts from builder
COPY --from=builder /build/build/ ./DeepCGHEngine/build/

# Copy ONNX model
COPY DeepCGHEngine/models/ ./models/

# Install the Python package
RUN cd DeepCGHEngine && pip install --no-cache-dir -e .

# Environment configuration
ENV DEEPCGH_DEVICE=cpu \
    DEEPCGH_MODEL_PATH=/app/models/deepcgh_unet.onnx \
    DEEPCGH_HOST=0.0.0.0 \
    DEEPCGH_PORT=8000 \
    DEEPCGH_HEIGHT=256 \
    DEEPCGH_WIDTH=256 \
    DEEPCGH_NUM_PLANES=5 \
    PYTHONUNBUFFERED=1

# Expose REST API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start server
CMD ["python", "-m", "deepcgh_engine.server"]

# ============================================================================
# GPU variant — NVIDIA CUDA base
# ============================================================================
FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu${UBUNTU_VERSION} AS runtime-gpu

ARG PYTHON_VERSION
ARG DEVICE

# Install Python and runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    libfftw3-single3 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python

WORKDIR /app

# Install Python dependencies with GPU support
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir \
    onnxruntime-gpu>=1.16 \
    numpy>=1.20 \
    Pillow>=9.0 \
    fastapi>=0.100 \
    uvicorn[standard]>=0.23 \
    python-multipart>=0.0.6 \
    cupy-cuda11x

# Copy Python package
COPY DeepCGHEngine/deepcgh_engine/ ./DeepCGHEngine/deepcgh_engine/
COPY DeepCGHEngine/pyproject.toml ./DeepCGHEngine/pyproject.toml
COPY DeepCGHEngine/setup.py ./DeepCGHEngine/setup.py
COPY DeepCGHEngine/MANIFEST.in ./DeepCGHEngine/MANIFEST.in

# Copy C++ built artifacts from builder
COPY --from=builder /build/build/ ./DeepCGHEngine/build/

# Copy ONNX model
COPY DeepCGHEngine/models/ ./models/

# Install the Python package
RUN cd DeepCGHEngine && pip install --no-cache-dir -e .

# Environment configuration
ENV DEEPCGH_DEVICE=gpu \
    DEEPCGH_MODEL_PATH=/app/models/deepcgh_unet.onnx \
    DEEPCGH_HOST=0.0.0.0 \
    DEEPCGH_PORT=8000 \
    DEEPCGH_HEIGHT=256 \
    DEEPCGH_WIDTH=256 \
    DEEPCGH_NUM_PLANES=5 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONUNBUFFERED=1

# Expose REST API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start server
CMD ["python3", "-m", "deepcgh_engine.server"]

# ============================================================================
# Final stage selection based on DEVICE arg
# ---------------------------------------------------------------------------
# Docker does not support conditional FROM, so we use a trick:
# The CPU image is the default. For GPU, use Dockerfile.gpu instead.
# ============================================================================
FROM runtime-${DEVICE} AS final
