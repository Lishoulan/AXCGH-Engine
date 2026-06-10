/**
 * @file PreProcessor.cpp
 * @brief Implementation of the RGB-D preprocessing pipeline.
 *
 * All color-space conversions follow ITU-R BT.601 standards.
 * The multi-plane volume assembly mirrors the DeepCGH Python logic
 * where the target image is distributed across depth planes.
 */

#include "deepcgh/PreProcessor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace deepcgh {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

namespace {

// ITU-R BT.601 conversion coefficients
constexpr float KR = 0.299f;
constexpr float KG = 0.587f;
constexpr float KB = 0.114f;

constexpr float PI = 3.14159265358979323846f;

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

PreProcessor::PreProcessor(const EngineConfig& config) : config_(config) {
    config_.validate();

    // Determine the number of color channels based on color-space mode
    switch (config_.color_space) {
        case ColorSpace::YCbCr:
        case ColorSpace::RGB:
            color_channels_ = 3;
            break;
        case ColorSpace::Gray:
            color_channels_ = 1;
            break;
    }
}

// ---------------------------------------------------------------------------
// Primary API
// ---------------------------------------------------------------------------

std::vector<float> PreProcessor::process(const RGBDFrame& frame) const {
    std::vector<float> tensor(tensor_size());
    process_into(frame, tensor.data(), tensor.size());
    return tensor;
}

size_t PreProcessor::process_into(const RGBDFrame& frame,
                                  float* out_tensor,
                                  size_t out_capacity) const {
    if (!frame.is_valid()) {
        throw std::invalid_argument("PreProcessor: invalid RGBDFrame");
    }
    if (frame.height != config_.height || frame.width != config_.width) {
        throw std::invalid_argument(
            "PreProcessor: frame dimensions [" +
            std::to_string(frame.height) + "x" + std::to_string(frame.width) +
            "] do not match config [" +
            std::to_string(config_.height) + "x" + std::to_string(config_.width) + "]");
    }
    if (out_capacity < tensor_size()) {
        throw std::invalid_argument(
            "PreProcessor: output buffer too small (" +
            std::to_string(out_capacity) + " < " +
            std::to_string(tensor_size()) + ")");
    }

    return process_raw(frame.rgb.data(), frame.depth.data(),
                       frame.height, frame.width,
                       out_tensor, out_capacity);
}

size_t PreProcessor::process_raw(const uint8_t* rgb_ptr,
                                 const float* depth_ptr,
                                 int32_t h, int32_t w,
                                 float* out_tensor,
                                 size_t out_capacity) const {
    if (!rgb_ptr || !depth_ptr || !out_tensor) {
        throw std::invalid_argument("PreProcessor: null pointer passed");
    }
    if (h != config_.height || w != config_.width) {
        throw std::invalid_argument(
            "PreProcessor: input dimensions do not match config");
    }
    if (out_capacity < tensor_size()) {
        throw std::invalid_argument("PreProcessor: output buffer too small");
    }

    const size_t pixel_count = static_cast<size_t>(h) * w;

    // Temporary buffers for intermediate results
    // We allocate these on the stack for small frames, or on the heap for large ones.
    std::vector<float> color_buf(pixel_count * color_channels_);
    std::vector<float> depth_buf(pixel_count);

    // Step 1: Color-space conversion
    switch (config_.color_space) {
        case ColorSpace::YCbCr:
            rgb_to_ycbcr(rgb_ptr, color_buf.data(), h, w);
            break;
        case ColorSpace::Gray:
            rgb_to_gray(rgb_ptr, color_buf.data(), h, w);
            break;
        case ColorSpace::RGB:
            // Direct uint8 -> float32 conversion with [0, 1] normalization
            for (size_t i = 0; i < pixel_count * 3; ++i) {
                color_buf[i] = static_cast<float>(rgb_ptr[i]) / 255.0f;
            }
            break;
    }

    // Step 2: Depth normalization to [0, 1]
    normalize_depth(depth_ptr, depth_buf.data(), h, w);

    // Step 3: Assemble multi-plane volume in NCHW layout
    assemble_volume(color_buf.data(), depth_buf.data(), out_tensor);

    // Step 4: Apply per-channel mean/std normalization
    if (config_.norm_mean != 0.0f || config_.norm_std != 1.0f) {
        const size_t total = tensor_size();
        for (size_t i = 0; i < total; ++i) {
            out_tensor[i] = (out_tensor[i] - config_.norm_mean) / config_.norm_std;
        }
    }

    return tensor_size();
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

int32_t PreProcessor::output_channels() const noexcept {
    // For DeepCGH, the model input is [H, W, num_planes] — a single
    // multi-plane volume. In NCHW layout this becomes [1, num_planes, H, W].
    // When color_channels_ > 1, we replicate the structure per color channel.
    return color_channels_ * config_.num_planes;
}

size_t PreProcessor::tensor_size() const noexcept {
    return static_cast<size_t>(1) * output_channels() * config_.height * config_.width;
}

const EngineConfig& PreProcessor::config() const noexcept {
    return config_;
}

// ---------------------------------------------------------------------------
// Color-Space Conversion
// ---------------------------------------------------------------------------

void PreProcessor::rgb_to_ycbcr(const uint8_t* rgb, float* ycbcr_out,
                                int32_t h, int32_t w) const {
    const size_t n = static_cast<size_t>(h) * w;
    float* y_ptr  = ycbcr_out;
    float* cb_ptr = ycbcr_out + n;
    float* cr_ptr = ycbcr_out + 2 * n;

    for (size_t i = 0; i < n; ++i) {
        const float r = static_cast<float>(rgb[i * 3 + 0]);
        const float g = static_cast<float>(rgb[i * 3 + 1]);
        const float b = static_cast<float>(rgb[i * 3 + 2]);

        // Luminance
        y_ptr[i] = (KR * r + KG * g + KB * b) / 255.0f;

        // Chrominance Blue
        cb_ptr[i] = (-0.168736f * r - 0.331264f * g + 0.5f * b) / 255.0f + 0.5f;

        // Chrominance Red
        cr_ptr[i] = (0.5f * r - 0.418688f * g - 0.081312f * b) / 255.0f + 0.5f;
    }
}

void PreProcessor::rgb_to_gray(const uint8_t* rgb, float* gray_out,
                               int32_t h, int32_t w) const {
    const size_t n = static_cast<size_t>(h) * w;

    for (size_t i = 0; i < n; ++i) {
        const float r = static_cast<float>(rgb[i * 3 + 0]);
        const float g = static_cast<float>(rgb[i * 3 + 1]);
        const float b = static_cast<float>(rgb[i * 3 + 2]);

        gray_out[i] = (KR * r + KG * g + KB * b) / 255.0f;
    }
}

// ---------------------------------------------------------------------------
// Depth Normalization
// ---------------------------------------------------------------------------

void PreProcessor::normalize_depth(const float* depth, float* depth_out,
                                   int32_t h, int32_t w) const {
    const size_t n = static_cast<size_t>(h) * w;

    // Find min and max for min-max normalization
    float d_min = std::numeric_limits<float>::max();
    float d_max = std::numeric_limits<float>::lowest();

    for (size_t i = 0; i < n; ++i) {
        if (depth[i] < d_min) d_min = depth[i];
        if (depth[i] > d_max) d_max = depth[i];
    }

    const float range = d_max - d_min;
    const float inv_range = (range > 1e-6f) ? (1.0f / range) : 0.0f;

    for (size_t i = 0; i < n; ++i) {
        depth_out[i] = (depth[i] - d_min) * inv_range;
    }
}

// ---------------------------------------------------------------------------
// Multi-Plane Volume Assembly
// ---------------------------------------------------------------------------

void PreProcessor::assemble_volume(const float* color_data,
                                   const float* depth_data,
                                   float* out_nchw) const {
    const int32_t H = config_.height;
    const int32_t W = config_.width;
    const int32_t C = config_.num_planes;
    const size_t pixel_count = static_cast<size_t>(H) * W;

    // The DeepCGH model expects input of shape [H, W, num_planes].
    // In NCHW layout: [1, num_planes, H, W].
    //
    // Strategy for volume assembly:
    //   - Center plane (C//2): luminance / grayscale image
    //   - Other planes: depth-weighted copies of the color data
    //     This encodes the 3D structure across depth planes.
    //
    // This mirrors the Python implementation where target volumes are
    // constructed as [H, W, C] with the image distributed across planes.

    const int32_t center_plane = C / 2;

    for (int32_t p = 0; p < C; ++p) {
        // Pointer to the start of this plane in NCHW output
        float* plane_ptr = out_nchw + static_cast<size_t>(p) * H * W;

        if (color_channels_ == 1) {
            // Single-channel mode: use luminance directly
            if (p == center_plane) {
                // Center plane: full intensity color data
                std::memcpy(plane_ptr, color_data, pixel_count * sizeof(float));
            } else {
                // Off-center planes: modulate by depth
                // Planes closer to center get more intensity from the image,
                // farther planes get more contribution from depth structure.
                const float plane_offset = static_cast<float>(p - center_plane);
                const float depth_weight = std::abs(plane_offset) /
                                           static_cast<float>(C / 2);

                for (size_t i = 0; i < pixel_count; ++i) {
                    // Blend color and depth based on plane distance from center
                    plane_ptr[i] = color_data[i] * (1.0f - depth_weight) +
                                   depth_data[i] * depth_weight;
                }
            }
        } else {
            // Multi-channel mode (YCbCr / RGB): replicate across color channels
            // For DeepCGH, the model input is [H, W, num_planes], so each
            // plane gets the luminance (Y) channel data.
            // Color channels are handled separately if needed by the model.
            const float* y_channel = color_data;  // Y is always the first channel

            if (p == center_plane) {
                std::memcpy(plane_ptr, y_channel, pixel_count * sizeof(float));
            } else {
                const float plane_offset = static_cast<float>(p - center_plane);
                const float depth_weight = std::abs(plane_offset) /
                                           static_cast<float>(C / 2);

                for (size_t i = 0; i < pixel_count; ++i) {
                    plane_ptr[i] = y_channel[i] * (1.0f - depth_weight) +
                                   depth_data[i] * depth_weight;
                }
            }
        }
    }
}

}  // namespace deepcgh
