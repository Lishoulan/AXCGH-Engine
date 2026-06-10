/**
 * @file EngineAPI.cpp
 * @brief Implementation of the high-level DeepCGH Engine API.
 *
 * Orchestrates the full pipeline: RGB-D input -> preprocessing ->
 * neural network inference (dual output: amp_0, phi_0) -> IFFT
 * post-processing (via FFTW3f) -> quantized SLM phase map.
 */

#define _USE_MATH_DEFINES
#include <cmath>

#include "deepcgh/EngineAPI.h"
#include "deepcgh/PreProcessor.h"
#include "deepcgh/InferenceCore.h"

#include <fftw3.h>

#include <cstring>
#include <sstream>

namespace deepcgh {

// ===========================================================================
// IFFTPostProcessor Implementation (FFTW3-based)
// ===========================================================================

IFFTPostProcessor::IFFTPostProcessor(int32_t quantization_bits)
    : quant_levels_(1 << quantization_bits) {}

IFFTPostProcessor::~IFFTPostProcessor() {
    release_plan();
}

IFFTPostProcessor::IFFTPostProcessor(IFFTPostProcessor&& other) noexcept
    : quant_levels_(other.quant_levels_)
    , fft_in_(other.fft_in_)
    , fft_out_(other.fft_out_)
    , fft_plan_(other.fft_plan_)
    , plan_h_(other.plan_h_)
    , plan_w_(other.plan_w_) {
    other.fft_in_   = nullptr;
    other.fft_out_  = nullptr;
    other.fft_plan_ = nullptr;
    other.plan_h_   = 0;
    other.plan_w_   = 0;
}

IFFTPostProcessor& IFFTPostProcessor::operator=(IFFTPostProcessor&& other) noexcept {
    if (this != &other) {
        release_plan();
        quant_levels_ = other.quant_levels_;
        fft_in_   = other.fft_in_;
        fft_out_  = other.fft_out_;
        fft_plan_ = other.fft_plan_;
        plan_h_   = other.plan_h_;
        plan_w_   = other.plan_w_;
        other.fft_in_   = nullptr;
        other.fft_out_  = nullptr;
        other.fft_plan_ = nullptr;
        other.plan_h_   = 0;
        other.plan_w_   = 0;
    }
    return *this;
}

Status IFFTPostProcessor::process(const float* amp_data,
                                   const float* phi_data,
                                   int32_t H, int32_t W,
                                   float* out_phase) {
    if (!amp_data || !phi_data || !out_phase) {
        return Status::INVALID_INPUT;
    }
    if (H <= 0 || W <= 0) {
        return Status::INVALID_INPUT;
    }

    const size_t N = static_cast<size_t>(H) * W;

    // Step 1: Ensure FFTW plan and buffers are ready
    Status plan_status = ensure_plan(H, W);
    if (plan_status != Status::OK) {
        return plan_status;
    }

    // Step 2: Build complex field  amp * exp(j * phi)  into fft_in_
    build_complex_field(amp_data, phi_data, N);

    // Step 3: IFFT shift (swap quadrants) on the complex data
    ifft_shift(H, W);

    // Step 4: Execute FFTW3 2D inverse FFT (complex-to-complex)
    fftwf_execute(reinterpret_cast<fftwf_plan>(fft_plan_));

    // Step 5: Extract angle (phase) from fft_out_ and apply 1/N normalization
    // FFTW's backward transform does NOT include the 1/N factor
    const float inv_norm = 1.0f / static_cast<float>(N);
    for (size_t i = 0; i < N; ++i) {
        const float re = fft_out_[i][0] * inv_norm;
        const float im = fft_out_[i][1] * inv_norm;
        out_phase[i] = std::atan2(im, re);
    }

    // Step 6: Quantize phase to SLM levels
    quantize_phase(out_phase, N);

    return Status::OK;
}

int32_t IFFTPostProcessor::quantization_levels() const noexcept {
    return quant_levels_;
}

// ---------------------------------------------------------------------------
// IFFTPostProcessor — Internal routines
// ---------------------------------------------------------------------------

Status IFFTPostProcessor::ensure_plan(int32_t H, int32_t W) {
    // Reuse existing plan if dimensions haven't changed
    if (fft_plan_ && plan_h_ == H && plan_w_ == W) {
        return Status::OK;
    }

    // Release any existing plan first
    release_plan();

    const int N = H * W;

    // Allocate FFTW3 complex arrays
    fft_in_  = fftwf_alloc_complex(N);
    fft_out_ = fftwf_alloc_complex(N);

    if (!fft_in_ || !fft_out_) {
        release_plan();
        return Status::ALLOCATION_FAILED;
    }

    // Create FFTW3 plan for 2D backward (inverse) FFT
    // FFTW_BACKWARD = inverse FFT (without 1/N normalization)
    fftwf_plan plan = fftwf_plan_dft_2d(
        H, W,
        fft_in_,
        fft_out_,
        FFTW_BACKWARD,
        FFTW_MEASURE);

    if (!plan) {
        release_plan();
        return Status::ALLOCATION_FAILED;
    }

    fft_plan_ = reinterpret_cast<void*>(plan);
    plan_h_ = H;
    plan_w_ = W;

    return Status::OK;
}

void IFFTPostProcessor::release_plan() {
    if (fft_plan_) {
        fftwf_destroy_plan(reinterpret_cast<fftwf_plan>(fft_plan_));
        fft_plan_ = nullptr;
    }
    if (fft_in_) {
        fftwf_free(fft_in_);
        fft_in_ = nullptr;
    }
    if (fft_out_) {
        fftwf_free(fft_out_);
        fft_out_ = nullptr;
    }
    plan_h_ = 0;
    plan_w_ = 0;
}

void IFFTPostProcessor::build_complex_field(const float* amp,
                                             const float* phi,
                                             size_t N) {
    // Write FFTW complex data: fft_in_[k][0] = re, fft_in_[k][1] = im
    for (size_t i = 0; i < N; ++i) {
        const float a = amp[i];
        const float p = phi[i];
        fft_in_[i][0] = a * std::cos(p);   // real part
        fft_in_[i][1] = a * std::sin(p);   // imaginary part
    }
}

void IFFTPostProcessor::ifft_shift(int32_t H, int32_t W) {
    // Swap quadrants on FFTW complex data.
    // Equivalent to numpy.fft.ifftshift on a 2D array.
    const int32_t half_h = H / 2;
    const int32_t half_w = W / 2;

    for (int32_t r = 0; r < half_h; ++r) {
        for (int32_t c = 0; c < half_w; ++c) {
            // Swap top-left <-> bottom-right
            const size_t tl = static_cast<size_t>(r) * W + c;
            const size_t br = static_cast<size_t>(r + half_h) * W + (c + half_w);

            std::swap(fft_in_[tl][0], fft_in_[br][0]);
            std::swap(fft_in_[tl][1], fft_in_[br][1]);

            // Swap top-right <-> bottom-left
            const size_t tr = static_cast<size_t>(r) * W + (c + half_w);
            const size_t bl = static_cast<size_t>(r + half_h) * W + c;

            std::swap(fft_in_[tr][0], fft_in_[bl][0]);
            std::swap(fft_in_[tr][1], fft_in_[bl][1]);
        }
    }
}

void IFFTPostProcessor::quantize_phase(float* phase, size_t N) {
    // Quantize: [-pi, pi] -> discrete levels -> [-pi, pi]
    const float two_pi = 2.0f * static_cast<float>(M_PI);
    const float L_minus_1 = static_cast<float>(quant_levels_ - 1);
    const float PI = static_cast<float>(M_PI);

    for (size_t i = 0; i < N; ++i) {
        float normalized = (phase[i] + PI) / two_pi;
        // Clamp to [0, 1] for safety
        normalized = std::max(0.0f, std::min(1.0f, normalized));
        float q = std::round(normalized * L_minus_1);
        phase[i] = q / L_minus_1 * two_pi - PI;
    }
}

// ===========================================================================
// EngineAPI Implementation
// ===========================================================================

// ---------------------------------------------------------------------------
// Factory & Construction
// ---------------------------------------------------------------------------

std::unique_ptr<EngineAPI> EngineAPI::create() {
    return std::unique_ptr<EngineAPI>(new EngineAPI());
}

EngineAPI::EngineAPI() = default;

EngineAPI::~EngineAPI() {
    shutdown();
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

Status EngineAPI::init(const std::string& model_path, const EngineConfig& config) {
    Status status = init(config);
    if (status != Status::OK) {
        return status;
    }

    if (!model_path.empty()) {
        status = inference_core_->load_model(model_path);
        if (status != Status::OK) {
            initialized_ = false;
            return set_error(status,
                "Failed to load model: " + model_path +
                " (" + status_to_string(status) + ")");
        }
    }

    initialized_ = true;
    last_error_.clear();
    return Status::OK;
}

Status EngineAPI::init(const EngineConfig& config) {
    try {
        config_ = config;
        config_.validate();
    } catch (const std::invalid_argument& e) {
        return set_error(Status::INVALID_INPUT,
            std::string("Invalid config: ") + e.what());
    }

    try {
        preprocessor_ = std::make_unique<PreProcessor>(config_);
    } catch (const std::exception& e) {
        return set_error(Status::ALLOCATION_FAILED,
            std::string("Failed to create PreProcessor: ") + e.what());
    }

    try {
        inference_core_ = std::make_unique<InferenceCore>(config_);
    } catch (const std::exception& e) {
        return set_error(Status::ALLOCATION_FAILED,
            std::string("Failed to create InferenceCore: ") + e.what());
    }

    try {
        ifft_postprocessor_ = std::make_unique<IFFTPostProcessor>(
            config_.quantization_bits);
    } catch (const std::exception& e) {
        return set_error(Status::ALLOCATION_FAILED,
            std::string("Failed to create IFFTPostProcessor: ") + e.what());
    }

    initialized_ = false;
    last_error_.clear();
    return Status::OK;
}

bool EngineAPI::is_ready() const noexcept {
    return initialized_ && inference_core_ && inference_core_->is_loaded();
}

void EngineAPI::shutdown() {
    ifft_postprocessor_.reset();
    if (inference_core_) {
        inference_core_->unload_model();
    }
    preprocessor_.reset();
    inference_core_.reset();
    initialized_ = false;
}

// ---------------------------------------------------------------------------
// Hologram Generation
// ---------------------------------------------------------------------------

Status EngineAPI::generate_hologram(const RGBDFrame& frame, PhaseMap& phase) {
    if (!is_ready()) {
        return set_error(Status::NOT_INITIALIZED,
            "Engine is not initialized. Call init() first.");
    }

    if (!frame.is_valid()) {
        return set_error(Status::INVALID_INPUT, "Invalid RGBDFrame");
    }
    if (frame.height != config_.height || frame.width != config_.width) {
        std::ostringstream oss;
        oss << "Frame size [" << frame.height << "x" << frame.width
            << "] does not match config ["
            << config_.height << "x" << config_.width << "]";
        return set_error(Status::SIZE_MISMATCH, oss.str());
    }

    std::vector<float> input_tensor;
    try {
        input_tensor = preprocessor_->process(frame);
    } catch (const std::invalid_argument& e) {
        return set_error(Status::INVALID_INPUT,
            std::string("Preprocessing failed: ") + e.what());
    }

    const size_t amp_sz = inference_core_->amp_size();
    const size_t phi_sz = inference_core_->phi_size();
    std::vector<float> raw_amp(amp_sz, 0.0f);
    std::vector<float> raw_phi(phi_sz, 0.0f);

    Status infer_status = inference_core_->forward(
        input_tensor.data(), raw_amp.data(), raw_phi.data());

    if (infer_status != Status::OK) {
        return set_error(infer_status, "Inference forward pass failed");
    }

    const size_t H = static_cast<size_t>(config_.height);
    const size_t W = static_cast<size_t>(config_.width);
    const size_t pixel_count = H * W;

    if (raw_amp.size() < pixel_count || raw_phi.size() < pixel_count) {
        return set_error(Status::SIZE_MISMATCH,
            "Model output size is smaller than expected spatial dimensions");
    }

    phase = PhaseMap(config_.height, config_.width, config_.phase_format);

    Status ifft_status = ifft_postprocessor_->process(
        raw_amp.data(), raw_phi.data(),
        config_.height, config_.width,
        phase.data.data());

    if (ifft_status != Status::OK) {
        return set_error(ifft_status, "IFFT post-processing failed");
    }

    last_error_.clear();
    return Status::OK;
}

Status EngineAPI::generate_hologram(const uint8_t* rgb_ptr,
                                    const float* depth_ptr,
                                    float* out_phase_ptr,
                                    int32_t h, int32_t w) {
    if (!is_ready()) {
        return set_error(Status::NOT_INITIALIZED,
            "Engine is not initialized. Call init() first.");
    }
    if (!rgb_ptr || !depth_ptr || !out_phase_ptr) {
        return set_error(Status::INVALID_INPUT, "Null pointer argument");
    }
    if (h != config_.height || w != config_.width) {
        return set_error(Status::SIZE_MISMATCH,
            "Input dimensions do not match config");
    }

    const size_t tensor_sz = preprocessor_->tensor_size();
    std::vector<float> input_tensor(tensor_sz);

    try {
        preprocessor_->process_raw(rgb_ptr, depth_ptr, h, w,
                                   input_tensor.data(), tensor_sz);
    } catch (const std::invalid_argument& e) {
        return set_error(Status::INVALID_INPUT,
            std::string("Preprocessing failed: ") + e.what());
    }

    const size_t amp_sz = inference_core_->amp_size();
    const size_t phi_sz = inference_core_->phi_size();
    std::vector<float> raw_amp(amp_sz, 0.0f);
    std::vector<float> raw_phi(phi_sz, 0.0f);

    Status infer_status = inference_core_->forward(
        input_tensor.data(), raw_amp.data(), raw_phi.data());

    if (infer_status != Status::OK) {
        return set_error(infer_status, "Inference forward pass failed");
    }

    const size_t pixel_count = static_cast<size_t>(h) * w;
    if (raw_amp.size() < pixel_count || raw_phi.size() < pixel_count) {
        return set_error(Status::SIZE_MISMATCH,
            "Model output size is smaller than expected spatial dimensions");
    }

    Status ifft_status = ifft_postprocessor_->process(
        raw_amp.data(), raw_phi.data(),
        h, w, out_phase_ptr);

    if (ifft_status != Status::OK) {
        return set_error(ifft_status, "IFFT post-processing failed");
    }

    last_error_.clear();
    return Status::OK;
}

Status EngineAPI::infer_only(const uint8_t* rgb_ptr,
                             const float* depth_ptr,
                             float* amp_out,
                             float* phi_out,
                             int32_t h, int32_t w) {
    if (!is_ready()) {
        return set_error(Status::NOT_INITIALIZED,
            "Engine is not initialized. Call init() first.");
    }
    if (!rgb_ptr || !depth_ptr || !amp_out || !phi_out) {
        return set_error(Status::INVALID_INPUT, "Null pointer argument");
    }
    if (h != config_.height || w != config_.width) {
        return set_error(Status::SIZE_MISMATCH,
            "Input dimensions do not match config");
    }

    const size_t tensor_sz = preprocessor_->tensor_size();
    std::vector<float> input_tensor(tensor_sz);

    try {
        preprocessor_->process_raw(rgb_ptr, depth_ptr, h, w,
                                   input_tensor.data(), tensor_sz);
    } catch (const std::invalid_argument& e) {
        return set_error(Status::INVALID_INPUT,
            std::string("Preprocessing failed: ") + e.what());
    }

    Status infer_status = inference_core_->forward(
        input_tensor.data(), amp_out, phi_out);

    if (infer_status != Status::OK) {
        return set_error(infer_status, "Inference forward pass failed");
    }

    last_error_.clear();
    return Status::OK;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

const EngineConfig& EngineAPI::config() const noexcept {
    return config_;
}

const std::string& EngineAPI::last_error() const noexcept {
    return last_error_;
}

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

Status EngineAPI::set_error(Status s, const std::string& msg) {
    last_error_ = msg;
    return s;
}

}  // namespace deepcgh
