/**
 * @file Types.h
 * @brief Core data structures for the DeepCGH Engine.
 *
 * This header defines the fundamental data types used throughout the
 * DeepCGHEngine pipeline: RGB-D frame input, depth maps, phase map
 * output for SLM display, and engine configuration structures.
 *
 * Copyright (c) 2026 DeepCGHEngine Project
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace deepcgh {

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------

/// Pixel format of the output phase map, dictated by the SLM hardware.
enum class PhaseFormat : uint8_t {
    Uint8  = 0,  ///< 8-bit grayscale [0, 255] — common for nematic SLMs
    Uint16 = 1,  ///< 16-bit grayscale [0, 65535] — high-precision SLMs
    Float  = 2   ///< 32-bit float [-pi, pi] — raw phase in radians (debug / export)
};

/// Color-space conversion strategy applied during preprocessing.
enum class ColorSpace : uint8_t {
    RGB   = 0,  ///< Keep raw RGB channels (3-channel input to network)
    YCbCr = 1,  ///< Convert to YCbCr and use luminance channel (1-channel)
    Gray  = 2   ///< Convert to grayscale (1-channel, simplest)
};

/// Execution provider for ONNX Runtime inference.
enum class ExecutionProvider : uint8_t {
    CPU = 0,   ///< CPU-only execution (portable, no GPU required)
    CUDA = 1,  ///< NVIDIA CUDA GPU acceleration
    DML  = 2   ///< DirectML (Windows GPU, vendor-agnostic)
};

// ---------------------------------------------------------------------------
// Configuration Structures
// ---------------------------------------------------------------------------

/// Engine-wide configuration passed at initialization.
struct EngineConfig {
    // ---- Input geometry ----
    int32_t height        = 512;       ///< Input frame height in pixels
    int32_t width         = 512;       ///< Input frame width in pixels
    int32_t num_planes    = 5;         ///< Number of depth planes (channels)

    // ---- Preprocessing ----
    ColorSpace color_space = ColorSpace::YCbCr;  ///< Color-space conversion mode
    float norm_mean        = 0.0f;    ///< Per-channel mean for normalization
    float norm_std         = 1.0f;    ///< Per-channel std-dev for normalization

    // ---- Inference ----
    ExecutionProvider provider = ExecutionProvider::CPU;  ///< ONNX execution provider
    int32_t device_id          = 0;     ///< GPU device index (when using CUDA/DML)
    bool enable_memory_pool    = true;  ///< Reuse IO buffers across frames
    int32_t intra_op_threads   = 4;    ///< ORT intra-op thread count (CPU mode)
    int32_t inter_op_threads   = 1;    ///< ORT inter-op thread count (CPU mode)

    // ---- Output ----
    PhaseFormat phase_format = PhaseFormat::Uint8;  ///< SLM output pixel format
    float wavelength         = 532e-6f;  ///< Laser wavelength in mm (default 532 nm)
    float pixel_size         = 8e-3f;    ///< SLM pixel pitch in mm
    float focal_length       = 200.0f;   ///< Focal length in mm
    float plane_distance     = 10.0f;    ///< Inter-plane distance in mm
    int32_t quantization_bits = 8;       ///< SLM quantization levels (2^bits)

    // ---- Interleave factor (matches DeepCGH int_factor) ----
    int32_t int_factor = 2;  ///< Space-to-depth / depth-to-space block size

    /// Validate configuration values; throws std::invalid_argument on failure.
    void validate() const {
        if (height <= 0 || width <= 0)
            throw std::invalid_argument("EngineConfig: height and width must be positive");
        if (num_planes <= 0)
            throw std::invalid_argument("EngineConfig: num_planes must be positive");
        if (norm_std <= 0.0f)
            throw std::invalid_argument("EngineConfig: norm_std must be positive");
        if (quantization_bits < 1 || quantization_bits > 16)
            throw std::invalid_argument("EngineConfig: quantization_bits must be in [1, 16]");
        if (int_factor < 1)
            throw std::invalid_argument("EngineConfig: int_factor must be >= 1");
        if (height % int_factor != 0 || width % int_factor != 0)
            throw std::invalid_argument(
                "EngineConfig: height and width must be divisible by int_factor");
    }
};

// ---------------------------------------------------------------------------
// Frame Input Structure
// ---------------------------------------------------------------------------

/**
 * @brief RGB-D frame captured from a sensor or rendered scene.
 *
 * Memory layout:
 *   - rgb: row-major, interleaved RGB uint8 [H * W * 3]
 *   - depth: row-major, float32 in meters [H * W]
 *
 * The structure owns its data via std::vector for safe lifetime management.
 * For zero-copy scenarios, use the pointer-based overloads in EngineAPI.
 */
struct RGBDFrame {
    int32_t height = 0;
    int32_t width  = 0;

    /// RGB pixel data, 3 channels, uint8, row-major [H * W * 3].
    std::vector<uint8_t> rgb;

    /// Depth map, single channel, float32 in meters, row-major [H * W].
    std::vector<float> depth;

    /// Default constructor — empty frame.
    RGBDFrame() = default;

    /// Construct a frame with pre-allocated buffers.
    RGBDFrame(int32_t h, int32_t w)
        : height(h), width(w),
          rgb(static_cast<size_t>(h) * w * 3, 0),
          depth(static_cast<size_t>(h) * w, 0.0f) {}

    /// Total number of RGB bytes.
    [[nodiscard]] size_t rgb_size() const noexcept {
        return static_cast<size_t>(height) * width * 3;
    }

    /// Total number of depth floats.
    [[nodiscard]] size_t depth_size() const noexcept {
        return static_cast<size_t>(height) * width;
    }

    /// Validate internal consistency.
    [[nodiscard]] bool is_valid() const noexcept {
        return height > 0 && width > 0 &&
               rgb.size() == rgb_size() &&
               depth.size() == depth_size();
    }
};

// ---------------------------------------------------------------------------
// Phase Map Output Structure
// ---------------------------------------------------------------------------

/**
 * @brief Phase map produced by the engine for SLM display.
 *
 * The raw phase is stored as float32 in radians [-pi, pi].
 * Quantized views are generated on demand based on PhaseFormat.
 */
struct PhaseMap {
    int32_t height = 0;
    int32_t width  = 0;

    /// Raw phase values in radians, float32, row-major [H * W].
    std::vector<float> data;

    /// Phase format used for quantized output.
    PhaseFormat format = PhaseFormat::Uint8;

    /// Default constructor.
    PhaseMap() = default;

    /// Construct with dimensions.
    PhaseMap(int32_t h, int32_t w, PhaseFormat fmt = PhaseFormat::Uint8)
        : height(h), width(w),
          data(static_cast<size_t>(h) * w, 0.0f),
          format(fmt) {}

    /// Total number of phase pixels.
    [[nodiscard]] size_t size() const noexcept {
        return static_cast<size_t>(height) * width;
    }

    /**
     * @brief Quantize the raw phase to the configured format.
     * @return Byte buffer containing quantized pixels.
     *
     * Uint8:  maps [-pi, pi] -> [0, 255]
     * Uint16: maps [-pi, pi] -> [0, 65535]
     * Float:  returns raw bytes of the float32 buffer (no transform)
     */
    [[nodiscard]] std::vector<uint8_t> quantize() const {
        std::vector<uint8_t> out;

        switch (format) {
            case PhaseFormat::Uint8: {
                out.resize(size());
                for (size_t i = 0; i < size(); ++i) {
                    // Map [-pi, pi] -> [0, 255]
                    float normalized = (data[i] + 3.14159265358979323846f) /
                                       (2.0f * 3.14159265358979323846f);
                    out[i] = static_cast<uint8_t>(
                        std::min(255.0f, std::max(0.0f, normalized * 255.0f)));
                }
                break;
            }
            case PhaseFormat::Uint16: {
                out.resize(size() * 2);
                auto* ptr = reinterpret_cast<uint16_t*>(out.data());
                for (size_t i = 0; i < size(); ++i) {
                    float normalized = (data[i] + 3.14159265358979323846f) /
                                       (2.0f * 3.14159265358979323846f);
                    ptr[i] = static_cast<uint16_t>(
                        std::min(65535.0f, std::max(0.0f, normalized * 65535.0f)));
                }
                break;
            }
            case PhaseFormat::Float: {
                // Direct byte copy of the float buffer
                size_t bytes = size() * sizeof(float);
                out.resize(bytes);
                std::memcpy(out.data(), data.data(), bytes);
                break;
            }
        }
        return out;
    }
};

// ---------------------------------------------------------------------------
// Error / Status Codes
// ---------------------------------------------------------------------------

/// Engine operation result codes.
enum class Status : int32_t {
    OK                  = 0,
    NOT_INITIALIZED     = 1,
    MODEL_LOAD_FAILED   = 2,
    INVALID_INPUT       = 3,
    INFERENCE_FAILED    = 4,
    SIZE_MISMATCH       = 5,
    PROVIDER_UNAVAIL    = 6,
    ALLOCATION_FAILED   = 7
};

/// Convert a Status code to a human-readable string.
[[nodiscard]] inline const char* status_to_string(Status s) noexcept {
    switch (s) {
        case Status::OK:                return "OK";
        case Status::NOT_INITIALIZED:   return "Engine not initialized";
        case Status::MODEL_LOAD_FAILED: return "Model loading failed";
        case Status::INVALID_INPUT:     return "Invalid input data";
        case Status::INFERENCE_FAILED:  return "Inference execution failed";
        case Status::SIZE_MISMATCH:     return "Input size mismatch";
        case Status::PROVIDER_UNAVAIL:  return "Execution provider unavailable";
        case Status::ALLOCATION_FAILED: return "Memory allocation failed";
        default:                        return "Unknown error";
    }
}

// ---------------------------------------------------------------------------
// Convenience Aliases
// ---------------------------------------------------------------------------

template <typename T>
using UniquePtr = std::unique_ptr<T>;

template <typename T>
using SharedPtr = std::shared_ptr<T>;

}  // namespace deepcgh
