"""
setup.py — Build script for DeepCGHEngine Python package.

Supports two build modes:
  1. Pure Python (default): Installs the Python-only engine using ONNX Runtime.
  2. C++ Extension: Compiles the PyBind11 module when a C++ compiler is available.

Usage:
  pip install .                          # Pure Python install
  pip install . --install-option="--cpp" # C++ extension (requires compiler)
"""

from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
import os
import sys

# Read version
version = "1.0.0"

# Check if C++ build is requested
BUILD_CPP = '--cpp' in sys.argv or os.environ.get('DEEPCGH_BUILD_CPP', '0') == '1'
if '--cpp' in sys.argv:
    sys.argv.remove('--cpp')


def get_cpp_extension():
    """Configure PyBind11 C++ extension if compiler is available."""
    try:
        import pybind11
    except ImportError:
        print("[WARN] pybind11 not found, skipping C++ extension build")
        return []

    if not BUILD_CPP:
        return []

    ort_root = os.environ.get('ONNXRUNTIME_ROOT', '')
    if not ort_root or not os.path.isdir(ort_root):
        print("[WARN] ONNXRUNTIME_ROOT not set, skipping C++ extension build")
        return []

    from pybind11.setup_helpers import Pybind11Extension

    ext_modules = [
        Pybind11Extension(
            "_deepcgh_engine",
            ["bindings/pybind_module.cpp"],
            include_dirs=[
                os.path.join(ort_root, "include"),
                os.path.join("include"),
            ],
            library_dirs=[os.path.join(ort_root, "lib")],
            libraries=["onnxruntime"],
            cxx_std=20,
            extra_compile_args=["/W4"] if sys.platform == "win32" else ["-Wall", "-Wextra"],
        )
    ]
    return ext_modules


setup(
    name="deepcgh-engine",
    version=version,
    description="Deep-learning Computer-Generated Holography Engine",
    author="DeepCGHEngine Project",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20",
        "onnxruntime>=1.12",
    ],
    extras_require={
        "gpu": ["cupy-cuda12x"],
        "dev": ["pybind11>=2.12", "pytest"],
    },
    package_data={
        "deepcgh_engine": ["py.typed"],
    },
    ext_modules=get_cpp_extension(),
    entry_points={
        "console_scripts": [
            "deepcgh-benchmark=deepcgh_engine.cli:benchmark_main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
    ],
)
