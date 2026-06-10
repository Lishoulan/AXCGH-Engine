/**
 * @file InferenceCore.cpp
 * @brief Implementation of the ONNX Runtime inference engine.
 *
 * Uses the ONNX Runtime C++ API to load models, configure execution
 * providers (CPU / CUDA / DirectML), manage memory pooling, and
 * execute single-pass forward inference with dual outputs (amp_0, phi_0).
 */

#include "deepcgh/InferenceCore.h"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>

// ONNX Runtime C++ API headers
#include <onnxruntime_cxx_api.h>

namespace deepcgh {

// ---------------------------------------------------------------------------
// PIMPL Implementation — hides ORT types from the public header
// ---------------------------------------------------------------------------

struct InferenceCore::OrtImpl {
    Ort::Env            env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo     memory_info;

    // Input / output node names (cached after model load)
    std::vector<std::string>          input_names_str;
    std::vector<std::string>          output_names_str;
    std::vector<const char*>          input_names;
    std::vector<const char*>          output_names;

    // Per-output shapes (populated during load_model)
    std::vector<std::vector<int64_t>> output_shapes;

    OrtImpl()
        : env(ORT_LOGGING_LEVEL_WARNING, "DeepCGHEngine"),
          memory_info(Ort::MemoryInfo::CreateCpu(
              OrtAllocatorType::OrtArenaAllocator,
              OrtMemType::OrtMemTypeDefault)) {}
};

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

InferenceCore::InferenceCore(const EngineConfig& config)
    : config_(config),
      ort_(std::make_unique<OrtImpl>()),
      use_memory_pool_(config.enable_memory_pool) {
    config_.validate();
}

InferenceCore::~InferenceCore() {
    unload_model();
}

// ---------------------------------------------------------------------------
// Model Management
// ---------------------------------------------------------------------------

Status InferenceCore::load_model(const std::string& model_path) {
    if (model_path.empty()) {
        return Status::MODEL_LOAD_FAILED;
    }

    try {
        // ---------------------------------------------------------------
        // Step 1: Configure session options
        // ---------------------------------------------------------------
        ort_->session_options = Ort::SessionOptions();

        // Set thread counts for CPU execution
        ort_->session_options.SetIntraOpNumThreads(config_.intra_op_threads);
        ort_->session_options.SetInterOpNumThreads(config_.inter_op_threads);

        // Enable optimized graph execution
        ort_->session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);

        // ---------------------------------------------------------------
        // Step 2: Append execution provider
        // ---------------------------------------------------------------
        switch (config_.provider) {
            case ExecutionProvider::CPU:
                // Default — no additional provider needed
                break;

            case ExecutionProvider::CUDA: {
#ifdef USE_ORT_CUDA
                // CUDA execution provider options
                OrtCUDAProviderOptionsV2 cuda_options;
                cuda_options.device_id = config_.device_id;
                cuda_options.arena_extend_strategy = 0;  // kNextPowerOfTwo
                cuda_options.gpu_mem_limit = 0;           // No limit
                cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
                cuda_options.do_copy_in_default_stream = true;

                ort_->session_options.AppendExecutionProvider_CUDA_V2(cuda_options);
#else
                return Status::PROVIDER_UNAVAIL;
#endif
                break;
            }

            case ExecutionProvider::DML: {
#ifdef USE_ORT_DML
                // DirectML execution provider (Windows GPU)
                OrtDmlProviderOptions dml_options;
                dml_options.device_id = config_.device_id;

                ort_->session_options.AppendExecutionProvider_DML(dml_options);
#else
                return Status::PROVIDER_UNAVAIL;
#endif
                break;
            }
        }

        // ---------------------------------------------------------------
        // Step 3: Create the session
        // ---------------------------------------------------------------
#ifdef _WIN32
        // Windows: ORT expects wide-string paths
        std::wstring wide_path(model_path.begin(), model_path.end());
        ort_->session = std::make_unique<Ort::Session>(
            ort_->env, wide_path.c_str(), ort_->session_options);
#else
        ort_->session = std::make_unique<Ort::Session>(
            ort_->env, model_path.c_str(), ort_->session_options);
#endif

        // ---------------------------------------------------------------
        // Step 4: Inspect model I/O metadata
        // ---------------------------------------------------------------
        Ort::AllocatorWithDefaultOptions allocator;

        // --- Input ---
        const size_t num_inputs = ort_->session->GetInputCount();
        ort_->input_names_str.clear();
        input_shape_.clear();

        for (size_t i = 0; i < num_inputs; ++i) {
            auto name_alloc = ort_->session->GetInputNameAllocated(i, allocator);
            ort_->input_names_str.emplace_back(name_alloc.get());

            auto type_info = ort_->session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();

            if (i == 0) {
                input_shape_ = shape;
                input_size_ = 1;
                for (auto dim : shape) {
                    // Dynamic batch dimension (-1) defaults to 1
                    input_size_ *= (dim > 0) ? static_cast<size_t>(dim) : 1;
                }
            }
        }

        // --- Outputs (expect 2: amp_0 and phi_0) ---
        num_outputs_ = ort_->session->GetOutputCount();
        ort_->output_names_str.clear();
        ort_->output_shapes.clear();
        amp_shape_.clear();
        phi_shape_.clear();
        amp_size_ = 0;
        phi_size_ = 0;

        for (size_t i = 0; i < num_outputs_; ++i) {
            auto name_alloc = ort_->session->GetOutputNameAllocated(i, allocator);
            ort_->output_names_str.emplace_back(name_alloc.get());

            auto type_info = ort_->session->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();

            // Compute element count for this output
            size_t elem_count = 1;
            for (auto dim : shape) {
                elem_count *= (dim > 0) ? static_cast<size_t>(dim) : 1;
            }

            // Store shape for all outputs
            ort_->output_shapes.push_back(shape);

            // Assign the first two outputs to amp_0 and phi_0
            if (i == 0) {
                amp_shape_ = shape;
                amp_size_ = elem_count;
            } else if (i == 1) {
                phi_shape_ = shape;
                phi_size_ = elem_count;
            }
        }

        // ---------------------------------------------------------------
        // Step 5: Validate model I/O against engine config
        // ---------------------------------------------------------------
        Status validation = validate_model_io();
        if (validation != Status::OK) {
            ort_->session.reset();
            return validation;
        }

        // ---------------------------------------------------------------
        // Step 5.5: Build c_str pointer vectors AFTER all strings are stored
        // ---------------------------------------------------------------
        // This must happen after all push_backs to input_names_str and
        // output_names_str, because vector reallocation invalidates
        // previously obtained c_str() pointers.
        ort_->input_names.clear();
        for (const auto& s : ort_->input_names_str) {
            ort_->input_names.push_back(s.c_str());
        }
        ort_->output_names.clear();
        for (const auto& s : ort_->output_names_str) {
            ort_->output_names.push_back(s.c_str());
        }

        // ---------------------------------------------------------------
        // Step 6: Allocate memory pools
        // ---------------------------------------------------------------
        if (use_memory_pool_) {
            Status pool_status = allocate_pools();
            if (pool_status != Status::OK) {
                ort_->session.reset();
                return pool_status;
            }
        }

        loaded_ = true;
        return Status::OK;

    } catch (const Ort::Exception& e) {
        // ORT-specific error
        ort_->session.reset();
        // If the error message mentions provider not found, return PROVIDER_UNAVAIL
        std::string msg(e.what());
        if (msg.find("provider") != std::string::npos ||
            msg.find("Provider") != std::string::npos ||
            msg.find("CUDA") != std::string::npos ||
            msg.find("DML") != std::string::npos) {
            return Status::PROVIDER_UNAVAIL;
        }
        return Status::MODEL_LOAD_FAILED;
    } catch (const std::exception& e) {
        ort_->session.reset();
        return Status::MODEL_LOAD_FAILED;
    }
}

bool InferenceCore::is_loaded() const noexcept {
    return loaded_ && ort_->session != nullptr;
}

void InferenceCore::unload_model() {
    ort_->session.reset();
    pooled_input_.clear();
    pooled_input_.shrink_to_fit();
    pooled_amp_.clear();
    pooled_amp_.shrink_to_fit();
    pooled_phi_.clear();
    pooled_phi_.shrink_to_fit();
    input_shape_.clear();
    amp_shape_.clear();
    phi_shape_.clear();
    input_size_ = 0;
    amp_size_ = 0;
    phi_size_ = 0;
    num_outputs_ = 0;
    loaded_ = false;
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

Status InferenceCore::forward(const float* input_tensor,
                               float* amp_output,
                               float* phi_output) {
    if (!is_loaded()) {
        return Status::NOT_INITIALIZED;
    }
    if (!input_tensor) {
        return Status::INVALID_INPUT;
    }

    try {
        // ---------------------------------------------------------------
        // Prepare input tensor
        // ---------------------------------------------------------------
        // Copy input data into pooled buffer or use directly
        const float* input_data = input_tensor;
        if (use_memory_pool_) {
            if (pooled_input_.size() < input_size_) {
                return Status::ALLOCATION_FAILED;
            }
            std::memcpy(pooled_input_.data(), input_tensor,
                        input_size_ * sizeof(float));
            input_data = pooled_input_.data();
        }

        // Create ORT input value tensor
        // Note: We use the model's actual input shape, replacing dynamic
        // dimensions (-1) with 1 for single-frame inference.
        std::vector<int64_t> actual_input_shape = input_shape_;
        for (auto& dim : actual_input_shape) {
            if (dim <= 0) dim = 1;
        }

        auto input_ort_tensor = Ort::Value::CreateTensor<float>(
            ort_->memory_info,
            const_cast<float*>(input_data),  // ORT API requires non-const
            input_size_,
            actual_input_shape.data(),
            actual_input_shape.size());

        // ---------------------------------------------------------------
        // Prepare output buffers for amp_0 and phi_0
        // ---------------------------------------------------------------
        float* amp_data = amp_output;
        float* phi_data = phi_output;
        if (use_memory_pool_) {
            amp_data = pooled_amp_.data();
            phi_data = pooled_phi_.data();
        }
        if (!amp_data || !phi_data) {
            return Status::ALLOCATION_FAILED;
        }

        // Create ORT output value tensors (pre-allocated)
        // amp_0
        std::vector<int64_t> actual_amp_shape = amp_shape_;
        for (auto& dim : actual_amp_shape) {
            if (dim <= 0) dim = 1;
        }

        auto amp_ort_tensor = Ort::Value::CreateTensor<float>(
            ort_->memory_info,
            amp_data,
            amp_size_,
            actual_amp_shape.data(),
            actual_amp_shape.size());

        // phi_0
        std::vector<int64_t> actual_phi_shape = phi_shape_;
        for (auto& dim : actual_phi_shape) {
            if (dim <= 0) dim = 1;
        }

        auto phi_ort_tensor = Ort::Value::CreateTensor<float>(
            ort_->memory_info,
            phi_data,
            phi_size_,
            actual_phi_shape.data(),
            actual_phi_shape.size());

        // ---------------------------------------------------------------
        // Run inference (single forward pass, dual outputs)
        // ---------------------------------------------------------------
        // Let ORT allocate output tensors internally, then copy results
        // to the caller's buffers. This is more portable than pre-allocating.
        auto output_tensors = ort_->session->Run(
            Ort::RunOptions{nullptr},
            ort_->input_names.data(),
            &input_ort_tensor,
            1,  // number of inputs
            ort_->output_names.data(),
            ort_->output_names.size());

        // Copy results to caller's buffers
        if (output_tensors.size() >= 2) {
            const float* amp_result = output_tensors[0].GetTensorData<float>();
            const float* phi_result = output_tensors[1].GetTensorData<float>();

            if (amp_result && amp_data) {
                std::memcpy(amp_data, amp_result, amp_size_ * sizeof(float));
            }
            if (phi_result && phi_data) {
                std::memcpy(phi_data, phi_result, phi_size_ * sizeof(float));
            }

            // When memory pool is active, also copy to the caller's buffers
            if (use_memory_pool_) {
                if (amp_output && amp_data) {
                    std::memcpy(amp_output, amp_data, amp_size_ * sizeof(float));
                }
                if (phi_output && phi_data) {
                    std::memcpy(phi_output, phi_data, phi_size_ * sizeof(float));
                }
            }
        }

        return Status::OK;

    } catch (const Ort::Exception& e) {
        fprintf(stderr, "[ORT ERROR] InferenceCore::forward: %s\n", e.what());
        return Status::INFERENCE_FAILED;
    } catch (const std::exception& e) {
        fprintf(stderr, "[ERROR] InferenceCore::forward: %s\n", e.what());
        return Status::INFERENCE_FAILED;
    }
}

const float* InferenceCore::pooled_amp() const noexcept {
    if (use_memory_pool_ && !pooled_amp_.empty()) {
        return pooled_amp_.data();
    }
    return nullptr;
}

const float* InferenceCore::pooled_phi() const noexcept {
    if (use_memory_pool_ && !pooled_phi_.empty()) {
        return pooled_phi_.data();
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Shape / Size Queries
// ---------------------------------------------------------------------------

size_t InferenceCore::input_size() const noexcept {
    return input_size_;
}

size_t InferenceCore::amp_size() const noexcept {
    return amp_size_;
}

size_t InferenceCore::phi_size() const noexcept {
    return phi_size_;
}

const std::vector<int64_t>& InferenceCore::input_shape() const noexcept {
    return input_shape_;
}

const std::vector<int64_t>& InferenceCore::amp_shape() const noexcept {
    return amp_shape_;
}

const std::vector<int64_t>& InferenceCore::phi_shape() const noexcept {
    return phi_shape_;
}

size_t InferenceCore::num_outputs() const noexcept {
    return num_outputs_;
}

const EngineConfig& InferenceCore::config() const noexcept {
    return config_;
}

// ---------------------------------------------------------------------------
// Internal Helpers
// ---------------------------------------------------------------------------

Status InferenceCore::validate_model_io() const {
    // The DeepCGH model expects input of shape [N, H, W, C] (NHWC) or
    // [N, C, H, W] (NCHW) depending on export format.
    // We validate that the spatial dimensions match the engine config.

    if (input_shape_.size() != 4) {
        // Expected 4D input [N, C/H, H/W, W/C]
        return Status::SIZE_MISMATCH;
    }

    // Find spatial dimensions — they should match config height/width.
    // For NHWC: shape = [N, H, W, C]  -> dims[1]=H, dims[2]=W
    // For NCHW: shape = [N, C, H, W]  -> dims[2]=H, dims[3]=W
    //
    // We check both layouts and accept whichever matches.
    bool nhwc_match = (input_shape_[1] == config_.height &&
                       input_shape_[2] == config_.width);
    bool nchw_match = (input_shape_[2] == config_.height &&
                       input_shape_[3] == config_.width);

    if (!nhwc_match && !nchw_match) {
        // Neither layout matches — the model may have been exported with
        // different spatial dimensions. This is a warning, not a hard error,
        // because the model might still work with resized input.
        return Status::SIZE_MISMATCH;
    }

    // Validate that we have at least 2 outputs (amp_0 and phi_0)
    if (num_outputs_ < 2) {
        return Status::SIZE_MISMATCH;
    }

    return Status::OK;
}

Status InferenceCore::allocate_pools() {
    try {
        if (input_size_ > 0) {
            pooled_input_.resize(input_size_, 0.0f);
        }
        if (amp_size_ > 0) {
            pooled_amp_.resize(amp_size_, 0.0f);
        }
        if (phi_size_ > 0) {
            pooled_phi_.resize(phi_size_, 0.0f);
        }
    } catch (const std::bad_alloc&) {
        return Status::ALLOCATION_FAILED;
    }
    return Status::OK;
}

}  // namespace deepcgh
