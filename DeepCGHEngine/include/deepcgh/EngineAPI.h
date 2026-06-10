/**
 * @file EngineAPI.h
 * @brief High-level API wrapper for the DeepCGH Engine.
 *
 * Provides a simple, C-style interface for initializing the engine,
 * feeding RGB-D frames, and retrieving SLM phase maps. This is the
 * primary entry point for both C++ consumers and Python bindings.
 *
 * Pipeline: RGBDFrame -> PreProcessor -> InferenceCore (amp_0, phi_0)
 *           -> IFFTPostProcessor -> PhaseMap
 */

#pragma once

#include "deepcgh/Types.h"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace deepcgh {

// Forward declarations
class PreProcessor;
class InferenceCore;

// ---------------------------------------------------------------------------
// IFFTPostProcessor
// ---------------------------------------------------------------------------

/**
 * @brief Post-processes the dual-output (amp_0, phi_0) from the neural
 *        network into a final SLM phase map via IFFT2D.
 *
 * Replicates the _ifft_AmPh Lambda from DeepCGH Python:
 *   1. Construct complex field: amp * exp(j * phi)
 *   2. IFFT shift + IFFT2D (via FFTW3f)
 *   3. Extract angle (phase)
 *   4. Quantize to SLM levels
 *
 * Uses FFTW3f (single-precision) for fast 2D inverse FFT.
 */
class IFFTPostProcessor {
public:
    /**
     * @brief Construct with SLM quantization bits.
     * @param quantization_bits Number of bits for SLM phase quantization
     *                          (e.g. 8 for 256 levels).
     */
    explicit IFFTPostProcessor(int32_t quantization_bits = 8);

    /// Destructor — releases FFTW plans and buffers.
    ~IFFTPostProcessor();

    // Non-copyable, movable
    IFFTPostProcessor(const IFFTPostProcessor&) = delete;
    IFFTPostProcessor& operator=(const IFFTPostProcessor&) = delete;
    IFFTPostProcessor(IFFTPostProcessor&&) noexcept;
    IFFTPostProcessor& operator=(IFFTPostProcessor&&) noexcept;

    /**
     * @brief Run IFFT post-processing on model outputs.
     *
     * @param amp_data  Amplitude data, shape [H * W] (squeezed from [1,H,W,1]).
     * @param phi_data  Initial phase data, shape [H * W] (squeezed from [1,H,W,1]).
     * @param H         Spatial height.
     * @param W         Spatial width.
     * @param out_phase Output phase buffer [H * W], values in [-pi, pi].
     * @return Status::OK on success.
     */
    Status process(const float* amp_data,
                   const float* phi_data,
                   int32_t H, int32_t W,
                   float* out_phase);

    /// Number of SLM quantization levels (2^quantization_bits).
    [[nodiscard]] int32_t quantization_levels() const noexcept;

private:
    int32_t quant_levels_;

    // FFTW3 working buffers (reused across calls for same dimensions)
    float (*fft_in_)[2]  = nullptr;   // Input complex array [N] (FFTW complex format)
    float (*fft_out_)[2] = nullptr;   // Output complex array [N]
    void*  fft_plan_     = nullptr;   // fftwf_plan (opaque pointer)
    int32_t plan_h_      = 0;         // Height used for current plan
    int32_t plan_w_      = 0;         // Width used for current plan

    // -----------------------------------------------------------------------
    // Internal routines
    // -----------------------------------------------------------------------

    /**
     * @brief Ensure FFTW plan and buffers are allocated for given dimensions.
     * Only reallocates if dimensions changed since last call.
     */
    Status ensure_plan(int32_t H, int32_t W);

    /**
     * @brief Release FFTW plan and buffers.
     */
    void release_plan();

    /**
     * @brief Build complex field from amplitude and phase into fft_in_.
     * fft_in_[2*k]   = amp[k] * cos(phi[k])
     * fft_in_[2*k+1] = amp[k] * sin(phi[k])
     */
    void build_complex_field(const float* amp, const float* phi, size_t N);

    /**
     * @brief In-place IFFT shift (swap quadrants) on interleaved complex data.
     */
    void ifft_shift(int32_t H, int32_t W);

    /**
     * @brief Quantize phase from [-pi, pi] to discrete SLM levels and back.
     */
    void quantize_phase(float* phase, size_t N);
};

// ---------------------------------------------------------------------------
// EngineAPI
// ---------------------------------------------------------------------------

/**
 * @brief Top-level facade that orchestrates the full DeepCGH pipeline.
 *
 * Pipeline: RGBDFrame -> PreProcessor -> InferenceCore (amp_0, phi_0)
 *           -> IFFTPostProcessor -> PhaseMap
 */
class EngineAPI {
public:
    /// Factory function — creates a new engine instance.
    [[nodiscard]] static std::unique_ptr<EngineAPI> create();

    /// Destructor — releases all resources.
    ~EngineAPI();

    // Non-copyable
    EngineAPI(const EngineAPI&) = delete;
    EngineAPI& operator=(const EngineAPI&) = delete;

    // -----------------------------------------------------------------------
    // Initialization
    // -----------------------------------------------------------------------

    Status init(const std::string& model_path, const EngineConfig& config);
    Status init(const EngineConfig& config);

    /// Returns true if the engine is fully initialized and ready.
    [[nodiscard]] bool is_ready() const noexcept;

    /// Shut down the engine and release all resources.
    void shutdown();

    // -----------------------------------------------------------------------
    // Hologram Generation
    // -----------------------------------------------------------------------

    Status generate_hologram(const RGBDFrame& frame, PhaseMap& phase);
    Status generate_hologram(const uint8_t* rgb_ptr,
                             const float* depth_ptr,
                             float* out_phase_ptr,
                             int32_t h, int32_t w);

    /**
     * @brief Run preprocessing + inference only, returning raw amp_0 and phi_0.
     * Bypasses the C++ IFFT post-processor.
     */
    Status infer_only(const uint8_t* rgb_ptr,
                      const float* depth_ptr,
                      float* amp_out,
                      float* phi_out,
                      int32_t h, int32_t w);

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    [[nodiscard]] const EngineConfig& config() const noexcept;
    [[nodiscard]] const std::string& last_error() const noexcept;

private:
    EngineAPI();

    std::unique_ptr<PreProcessor>       preprocessor_;
    std::unique_ptr<InferenceCore>      inference_core_;
    std::unique_ptr<IFFTPostProcessor>  ifft_postprocessor_;
    EngineConfig                        config_;
    bool                                initialized_ = false;
    std::string                         last_error_;

    Status set_error(Status s, const std::string& msg);
};

}  // namespace deepcgh
