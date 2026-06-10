/**
 * @file InferenceCore.h
 * @brief ONNX Runtime inference engine for DeepCGHEngine.
 *
 * Encapsulates model loading, memory pooling, and single-pass forward
 * inference for the DeepCGH U-Net model. Designed for real-time
 * frame-by-frame hologram generation with minimal latency.
 *
 * The ONNX model produces TWO outputs:
 *   - amp_0: [1, H, W, 1] amplitude
 *   - phi_0: [1, H, W, 1] initial phase
 * These are post-processed by IFFTPostProcessor to produce the final
 * SLM phase map.
 */

#pragma once

#include "deepcgh/Types.h"

#include <memory>
#include <string>
#include <vector>

// Forward-declare ONNX Runtime types to avoid leaking the header.
namespace Ort {
class Session;
class SessionOptions;
class Env;
class MemoryInfo;
class Allocator;
}  // namespace Ort

namespace deepcgh {

// ---------------------------------------------------------------------------
// InferenceCore
// ---------------------------------------------------------------------------

/**
 * @brief Manages ONNX Runtime model loading, memory pooling, and inference.
 *
 * Key design decisions:
 *   - **Memory Pooling**: Input and output tensor buffers are allocated once
 *     during initialization and reused for every frame, eliminating per-frame
 *     heap allocation overhead.
 *   - **Dual Output**: The DeepCGH model outputs amp_0 and phi_0; both are
 *     populated by forward() into separate pooled buffers.
 *   - **Single Forward Pass**: The DeepCGH model is a non-iterative U-Net;
 *     each call to `forward()` performs exactly one inference pass.
 *   - **Thread Safety**: The underlying ORT session is thread-safe for
 *     concurrent reads. However, the pooled buffers are NOT thread-safe;
 *     use one InferenceCore per thread, or synchronize externally.
 *
 * Typical lifecycle:
 *   1. Construct with config.
 *   2. Call `load_model()` to initialize the ORT session.
 *   3. Call `forward()` per frame.
 *   4. Destruction releases all resources.
 */
class InferenceCore {
public:
    /// Construct with engine configuration.
    explicit InferenceCore(const EngineConfig& config);

    /// Destructor — releases ORT session and pooled memory.
    ~InferenceCore();

    // Non-copyable, non-movable (owns ORT session with internal state)
    InferenceCore(const InferenceCore&) = delete;
    InferenceCore& operator=(const InferenceCore&) = delete;
    InferenceCore(InferenceCore&&) = delete;
    InferenceCore& operator=(InferenceCore&&) = delete;

    // -----------------------------------------------------------------------
    // Model Management
    // -----------------------------------------------------------------------

    /**
     * @brief Load an ONNX model from disk and initialize the inference session.
     *
     * This method:
     *   1. Creates the ORT session with the configured execution provider.
     *   2. Inspects model input/output shapes and validates them against config.
     *   3. Allocates pooled I/O buffers if memory pooling is enabled.
     *
     * @param model_path Filesystem path to the .onnx model file.
     * @return Status::OK on success, or an appropriate error code.
     */
    Status load_model(const std::string& model_path);

    /// Returns true if a model has been successfully loaded.
    [[nodiscard]] bool is_loaded() const noexcept;

    /// Release the current model and all associated resources.
    void unload_model();

    // -----------------------------------------------------------------------
    // Inference
    // -----------------------------------------------------------------------

    /**
     * @brief Execute a single forward pass through the network.
     *
     * The model produces two outputs (amp_0 and phi_0), both of which
     * are written into the internal pooled buffers when memory pooling
     * is enabled, or into the caller-provided buffers otherwise.
     *
     * @param input_tensor  Pointer to the input data in NCHW float32 layout.
     *                      Must contain exactly `input_size()` floats.
     * @param amp_output    Pointer to the amplitude output buffer. Must have
     *                      capacity for at least `amp_size()` floats.
     *                      If memory pooling is enabled, this parameter is
     *                      ignored and the internal pooled buffer is used.
     * @param phi_output    Pointer to the phase output buffer. Must have
     *                      capacity for at least `phi_size()` floats.
     *                      If memory pooling is enabled, this parameter is
     *                      ignored and the internal pooled buffer is used.
     * @return Status::OK on success, or an error code.
     *
     * @note If memory pooling is disabled, the caller must ensure that
     *       both output buffers point to valid memory of sufficient size.
     */
    Status forward(const float* input_tensor,
                   float* amp_output,
                   float* phi_output);

    /**
     * @brief Get a pointer to the pooled amplitude output from the last forward pass.
     *
     * Only valid when memory pooling is enabled and after a successful
     * call to `forward()`.
     *
     * @return Pointer to float32 amplitude data, or nullptr if unavailable.
     */
    [[nodiscard]] const float* pooled_amp() const noexcept;

    /**
     * @brief Get a pointer to the pooled phase output from the last forward pass.
     *
     * Only valid when memory pooling is enabled and after a successful
     * call to `forward()`.
     *
     * @return Pointer to float32 phase data, or nullptr if unavailable.
     */
    [[nodiscard]] const float* pooled_phi() const noexcept;

    // -----------------------------------------------------------------------
    // Shape / Size Queries
    // -----------------------------------------------------------------------

    /// Number of floats in the model input tensor.
    [[nodiscard]] size_t input_size() const noexcept;

    /// Number of floats in the amplitude output tensor (amp_0).
    [[nodiscard]] size_t amp_size() const noexcept;

    /// Number of floats in the phase output tensor (phi_0).
    [[nodiscard]] size_t phi_size() const noexcept;

    /// Model input shape as a vector of dimensions [N, C, H, W].
    [[nodiscard]] const std::vector<int64_t>& input_shape() const noexcept;

    /// Amplitude output shape as a vector of dimensions.
    [[nodiscard]] const std::vector<int64_t>& amp_shape() const noexcept;

    /// Phase output shape as a vector of dimensions.
    [[nodiscard]] const std::vector<int64_t>& phi_shape() const noexcept;

    /// The bound configuration.
    [[nodiscard]] const EngineConfig& config() const noexcept;

    /// Number of model outputs (expected: 2 for amp_0 and phi_0).
    [[nodiscard]] size_t num_outputs() const noexcept;

private:
    EngineConfig config_;

    // ORT components (hidden behind PIMPL to avoid header leakage)
    struct OrtImpl;
    std::unique_ptr<OrtImpl> ort_;

    // Model I/O metadata
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> amp_shape_;
    std::vector<int64_t> phi_shape_;
    size_t input_size_  = 0;
    size_t amp_size_    = 0;
    size_t phi_size_    = 0;
    size_t num_outputs_ = 0;
    bool loaded_        = false;

    // Pooled memory buffers (allocated once, reused per frame)
    std::vector<float> pooled_input_;
    std::vector<float> pooled_amp_;
    std::vector<float> pooled_phi_;
    bool use_memory_pool_ = true;

    // -----------------------------------------------------------------------
    // Internal Helpers
    // -----------------------------------------------------------------------

    /// Configure the ORT session options based on EngineConfig.
    void configure_session(/* Ort::SessionOptions& options */);

    /// Validate that model I/O shapes are compatible with the engine config.
    Status validate_model_io() const;

    /// Allocate pooled I/O buffers.
    Status allocate_pools();
};

}  // namespace deepcgh
