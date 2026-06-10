/**
 * @file PreProcessor.h
 * @brief RGB-D preprocessing pipeline for DeepCGHEngine.
 *
 * Converts raw RGB-D sensor data into normalized tensors suitable for
 * neural network inference. Supports RGB -> YCbCr / Grayscale conversion,
 * depth normalization, and multi-plane volume assembly.
 */

#pragma once

#include "deepcgh/Types.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace deepcgh {

// ---------------------------------------------------------------------------
// PreProcessor
// ---------------------------------------------------------------------------

/**
 * @brief Stateless preprocessing module that transforms RGB-D frames into
 *        model-ready input tensors.
 *
 * The processor handles:
 *   1. Color-space conversion (RGB -> YCbCr / Gray / passthrough)
 *   2. Depth map normalization to [0, 1] range
 *   3. Assembly of the multi-plane input volume [H, W, C]
 *   4. Per-channel mean/std normalization
 *   5. Conversion to a contiguous float32 tensor in NCHW layout
 *
 * All operations are designed to be allocation-free when the caller
 * provides pre-allocated output buffers (via the `process_into` overloads).
 */
class PreProcessor {
public:
    /// Construct a processor bound to the given engine configuration.
    explicit PreProcessor(const EngineConfig& config);

    /// Default destructor.
    ~PreProcessor() = default;

    // Non-copyable, movable
    PreProcessor(const PreProcessor&) = delete;
    PreProcessor& operator=(const PreProcessor&) = delete;
    PreProcessor(PreProcessor&&) noexcept = default;
    PreProcessor& operator=(PreProcessor&&) noexcept = default;

    // -----------------------------------------------------------------------
    // Primary API
    // -----------------------------------------------------------------------

    /**
     * @brief Process an RGBDFrame into a contiguous float32 tensor.
     *
     * @param frame  Input RGB-D frame (owns its data).
     * @return Vector of float32 in NCHW layout [1, C, H, W] where C
     *         depends on color_space and num_planes.
     *
     * @throws std::invalid_argument if frame dimensions don't match config.
     */
    [[nodiscard]] std::vector<float> process(const RGBDFrame& frame) const;

    /**
     * @brief Zero-copy variant: process into a caller-provided buffer.
     *
     * @param frame      Input RGB-D frame.
     * @param out_tensor Pre-allocated output buffer (must be large enough).
     * @return Number of floats written.
     *
     * @throws std::invalid_argument on dimension / buffer size mismatch.
     */
    size_t process_into(const RGBDFrame& frame,
                        float* out_tensor,
                        size_t out_capacity) const;

    /**
     * @brief Process raw pointer data (for interop with external pipelines).
     *
     * @param rgb_ptr   Pointer to RGB uint8 data [H * W * 3].
     * @param depth_ptr Pointer to depth float32 data [H * W].
     * @param h         Frame height.
     * @param w         Frame width.
     * @param out_tensor Output buffer (NCHW float32).
     * @param out_capacity Capacity of the output buffer in floats.
     * @return Number of floats written.
     */
    size_t process_raw(const uint8_t* rgb_ptr,
                       const float* depth_ptr,
                       int32_t h, int32_t w,
                       float* out_tensor,
                       size_t out_capacity) const;

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Number of channels in the output tensor (depends on color_space + num_planes).
    [[nodiscard]] int32_t output_channels() const noexcept;

    /// Total number of floats in the output tensor per frame.
    [[nodiscard]] size_t tensor_size() const noexcept;

    /// The bound configuration.
    [[nodiscard]] const EngineConfig& config() const noexcept;

private:
    EngineConfig config_;

    /// Number of color channels after color-space conversion (1 or 3).
    int32_t color_channels_ = 1;

    // -----------------------------------------------------------------------
    // Internal conversion routines
    // -----------------------------------------------------------------------

    /**
     * @brief Convert RGB uint8 to YCbCr float32.
     * Output: Y at [0..H*W], Cb at [H*W..2*H*W], Cr at [2*H*W..3*H*W].
     *
     * ITU-R BT.601 coefficients:
     *   Y  =  0.299*R + 0.587*G + 0.114*B
     *   Cb = -0.169*R - 0.331*G + 0.500*B + 0.5
     *   Cr =  0.500*R - 0.419*G - 0.081*B + 0.5
     */
    void rgb_to_ycbcr(const uint8_t* rgb, float* ycbcr_out,
                      int32_t h, int32_t w) const;

    /// Convert RGB uint8 to grayscale float32 using luminance.
    void rgb_to_gray(const uint8_t* rgb, float* gray_out,
                     int32_t h, int32_t w) const;

    /// Normalize depth map to [0, 1] range (min-max normalization).
    void normalize_depth(const float* depth, float* depth_out,
                         int32_t h, int32_t w) const;

    /**
     * @brief Assemble the multi-plane volume from color and depth channels.
     *
     * The DeepCGH model expects input of shape [H, W, num_planes] where
     * the center plane is the 2D target and surrounding planes encode
     * depth information. This function distributes the color + depth data
     * across the plane dimension.
     *
     * Layout in the output NCHW tensor:
     *   - If color_channels_ == 1:  [1, num_planes, H, W]
     *   - If color_channels_ == 3:  [1, 3 * num_planes, H, W]
     */
    void assemble_volume(const float* color_data,
                         const float* depth_data,
                         float* out_nchw) const;
};

}  // namespace deepcgh
