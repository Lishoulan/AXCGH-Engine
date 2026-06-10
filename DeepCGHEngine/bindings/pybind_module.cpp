/**
 * @file pybind_module.cpp
 * @brief PyBind11 bindings for DeepCGHEngine.
 *
 * Exposes the engine's C++ API to Python, enabling:
 *   - Engine initialization with model path and configuration
 *   - Per-frame hologram generation from numpy RGB-D arrays
 *   - Phase map output as numpy arrays
 *
 * IFFT post-processing is performed via NumPy's FFT (np.fft.ifft2)
 * instead of the naive C++ DFT, ensuring both correctness and speed.
 *
 * Build:
 *   The module is compiled as _deepcgh_engine.<pyd/.so> and is
 *   imported from the Python package as:
 *       from deepcgh_engine import DeepCGHEngine
 */

#define _USE_MATH_DEFINES
#include <cmath>

#include "deepcgh/EngineAPI.h"
#include "deepcgh/Types.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <cstring>
#include <memory>
#include <string>
#include <complex>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Numpy <-> C++ Buffer Conversion Helpers
// ---------------------------------------------------------------------------

/**
 * @brief Validate and extract a contiguous uint8 numpy array as a raw pointer.
 */
static const uint8_t* numpy_to_uint8_ptr(const py::array_t<uint8_t>& arr,
                                         size_t expected_size) {
    auto buf = arr.request();
    if (buf.ndim != 1 && buf.ndim != 3) {
        throw std::invalid_argument(
            "Expected 1D or 3D uint8 array, got " +
            std::to_string(buf.ndim) + "D");
    }
    if (buf.size != static_cast<ssize_t>(expected_size)) {
        throw std::invalid_argument(
            "Array size mismatch: expected " + std::to_string(expected_size) +
            ", got " + std::to_string(buf.size));
    }
    return static_cast<const uint8_t*>(buf.ptr);
}

/**
 * @brief Validate and extract a contiguous float32 numpy array as a raw pointer.
 */
static const float* numpy_to_float_ptr(const py::array_t<float>& arr,
                                       size_t expected_size) {
    auto buf = arr.request();
    if (buf.ndim != 1 && buf.ndim != 2) {
        throw std::invalid_argument(
            "Expected 1D or 2D float32 array, got " +
            std::to_string(buf.ndim) + "D");
    }
    if (buf.size != static_cast<ssize_t>(expected_size)) {
        throw std::invalid_argument(
            "Array size mismatch: expected " + std::to_string(expected_size) +
            ", got " + std::to_string(buf.size));
    }
    return static_cast<const float*>(buf.ptr);
}

// ---------------------------------------------------------------------------
// NumPy-based IFFT Post-Processing
// ---------------------------------------------------------------------------

/**
 * @brief Perform IFFT post-processing using NumPy FFT.
 *
 * Replicates the _ifft_AmPh Lambda from DeepCGH Python:
 *   1. Squeeze trailing dim: [1, H, W, 1] -> [1, H, W]
 *   2. Construct complex field: amp * exp(j * phi)
 *   3. IFFT shift + IFFT2D (via np.fft.ifft2)
 *   4. Extract angle (phase)
 *   5. Quantize to SLM levels
 *
 * @param amp_data  Raw amplitude data pointer (H*W floats)
 * @param phi_data  Raw phase data pointer (H*W floats)
 * @param H         Spatial height
 * @param W         Spatial width
 * @param quantization_bits  Number of SLM quantization bits
 * @return float32 numpy array of shape [H, W] with phase in [-pi, pi]
 */
static py::array_t<float> ifft_postprocess_numpy(
        const float* amp_data,
        const float* phi_data,
        int32_t H, int32_t W,
        int32_t quantization_bits = 8) {

    const size_t pixel_count = static_cast<size_t>(H) * W;

    // Step 1: Build complex field amp * exp(j * phi) as numpy complex64 array
    std::vector<ssize_t> complex_shape = {1, H, W};
    auto complex_arr = py::array_t<std::complex<float>>(complex_shape);
    auto complex_buf = complex_arr.request();
    auto* complex_ptr = static_cast<std::complex<float>*>(complex_buf.ptr);

    for (size_t i = 0; i < pixel_count; ++i) {
        float a = amp_data[i];
        float p = phi_data[i];
        complex_ptr[i] = std::complex<float>(a * std::cos(p), a * std::sin(p));
    }

    // Step 2: IFFT shift + IFFT2D using numpy
    py::module np = py::module::import("numpy");
    py::module np_fft = py::module::import("numpy.fft");

    // ifftshift on spatial axes (1, 2)
    py::object shifted = np_fft.attr("ifftshift")(complex_arr, py::make_tuple(1, 2));

    // IFFT2D
    py::object slm_field = np_fft.attr("ifft2")(shifted);

    // Step 3: Extract angle (phase)
    py::object modulation = np.attr("angle")(slm_field);

    // Step 4: Squeeze to [H, W]
    py::object phase = np.attr("squeeze")(modulation);

    // Step 5: Quantize phase to SLM levels
    const int32_t quant_levels = 1 << quantization_bits;
    const float q_scale = static_cast<float>(quant_levels - 1) / (2.0f * static_cast<float>(M_PI));
    const float q_inv = 2.0f * static_cast<float>(M_PI) / static_cast<float>(quant_levels - 1);
    const float PI = static_cast<float>(M_PI);

    // Convert to float32 numpy array for quantization
    py::object phase_f32 = np.attr("asarray")(phase, py::arg("dtype") = py::dtype("float32"));
    auto phase_float = py::array_t<float>::ensure(phase_f32);
    auto phase_buf = phase_float.request();
    auto* phase_ptr = static_cast<float*>(phase_buf.ptr);
    ssize_t phase_size = phase_buf.size;

    for (ssize_t i = 0; i < phase_size; ++i) {
        float normalized = (phase_ptr[i] + PI) * q_scale;
        if (normalized < 0.0f) normalized = 0.0f;
        if (normalized > static_cast<float>(quant_levels - 1))
            normalized = static_cast<float>(quant_levels - 1);
        float q = std::round(normalized);
        phase_ptr[i] = q * q_inv - PI;
    }

    // Ensure output is [H, W] float32
    if (phase_buf.ndim != 2 || phase_buf.shape[0] != H || phase_buf.shape[1] != W) {
        std::vector<ssize_t> hw_shape = {H, W};
        return py::array_t<float>(hw_shape, phase_ptr);
    }

    return phase_float;
}

// ---------------------------------------------------------------------------
// Python Wrapper Class
// ---------------------------------------------------------------------------

/**
 * @brief Python-facing wrapper around EngineAPI.
 *
 * Handles numpy array conversion and provides a Pythonic interface.
 * IFFT post-processing is done via NumPy FFT for correctness and speed.
 */
class PyDeepCGHEngine {
public:
    PyDeepCGHEngine() : engine_(deepcgh::EngineAPI::create()) {}

    /// Initialize the engine with model path and keyword configuration.
    void init(const std::string& model_path,
              int height, int width, int num_planes,
              const std::string& color_space = "ycbcr",
              const std::string& provider = "cpu",
              int device_id = 0,
              bool enable_memory_pool = true,
              const std::string& phase_format = "uint8",
              float wavelength = 532e-6f,
              float pixel_size = 8e-3f,
              float focal_length = 200.0f,
              float plane_distance = 10.0f,
              int quantization_bits = 8,
              int int_factor = 2,
              float norm_mean = 0.0f,
              float norm_std = 1.0f,
              int intra_op_threads = 4,
              int inter_op_threads = 1) {
        deepcgh::EngineConfig config;
        config.height             = height;
        config.width              = width;
        config.num_planes         = num_planes;
        config.device_id          = device_id;
        config.enable_memory_pool = enable_memory_pool;
        config.wavelength         = wavelength;
        config.pixel_size         = pixel_size;
        config.focal_length       = focal_length;
        config.plane_distance     = plane_distance;
        config.quantization_bits  = quantization_bits;
        config.int_factor         = int_factor;
        config.norm_mean          = norm_mean;
        config.norm_std           = norm_std;
        config.intra_op_threads   = intra_op_threads;
        config.inter_op_threads   = inter_op_threads;

        // Parse color space string
        if (color_space == "rgb" || color_space == "RGB") {
            config.color_space = deepcgh::ColorSpace::RGB;
        } else if (color_space == "gray" || color_space == "Gray") {
            config.color_space = deepcgh::ColorSpace::Gray;
        } else {
            config.color_space = deepcgh::ColorSpace::YCbCr;
        }

        // Parse execution provider string
        if (provider == "cuda" || provider == "CUDA") {
            config.provider = deepcgh::ExecutionProvider::CUDA;
        } else if (provider == "dml" || provider == "DML") {
            config.provider = deepcgh::ExecutionProvider::DML;
        } else {
            config.provider = deepcgh::ExecutionProvider::CPU;
        }

        // Parse phase format string
        if (phase_format == "uint16" || phase_format == "Uint16") {
            config.phase_format = deepcgh::PhaseFormat::Uint16;
        } else if (phase_format == "float" || phase_format == "Float") {
            config.phase_format = deepcgh::PhaseFormat::Float;
        } else {
            config.phase_format = deepcgh::PhaseFormat::Uint8;
        }

        deepcgh::Status status = engine_->init(model_path, config);
        if (status != deepcgh::Status::OK) {
            throw std::runtime_error(
                "Engine init failed: " + std::string(deepcgh::status_to_string(status)) +
                " — " + engine_->last_error());
        }
    }

    /// Generate hologram from numpy RGB and depth arrays using NumPy FFT.
    py::array_t<float> generate_hologram(
            const py::array_t<uint8_t>& rgb,
            const py::array_t<float>& depth) {
        if (!engine_->is_ready()) {
            throw std::runtime_error("Engine not initialized. Call init() first.");
        }

        const auto& cfg = engine_->config();
        const size_t H = static_cast<size_t>(cfg.height);
        const size_t W = static_cast<size_t>(cfg.width);

        // Validate and extract pointers
        const uint8_t* rgb_ptr = numpy_to_uint8_ptr(rgb, H * W * 3);
        const float* depth_ptr = numpy_to_float_ptr(depth, H * W);

        // Step 1: Run preprocessing + inference (bypass C++ IFFT)
        const size_t pixel_count = H * W;
        std::vector<float> raw_amp(pixel_count, 0.0f);
        std::vector<float> raw_phi(pixel_count, 0.0f);

        deepcgh::Status status = engine_->infer_only(
            rgb_ptr, depth_ptr,
            raw_amp.data(), raw_phi.data(),
            cfg.height, cfg.width);

        if (status != deepcgh::Status::OK) {
            throw std::runtime_error(
                "Inference failed: " + std::string(deepcgh::status_to_string(status)) +
                " — " + engine_->last_error());
        }

        // Step 2: IFFT post-processing via NumPy FFT
        return ifft_postprocess_numpy(
            raw_amp.data(), raw_phi.data(),
            cfg.height, cfg.width,
            cfg.quantization_bits);
    }

    /// Generate hologram and return quantized phase map for SLM display.
    py::array generate_hologram_quantized(
            const py::array_t<uint8_t>& rgb,
            const py::array_t<float>& depth) {
        if (!engine_->is_ready()) {
            throw std::runtime_error("Engine not initialized. Call init() first.");
        }

        const auto& cfg = engine_->config();

        // Get float phase via NumPy FFT pipeline
        auto phase = generate_hologram(rgb, depth);
        auto phase_buf = phase.request();
        const float* phase_ptr = static_cast<const float*>(phase_buf.ptr);
        const size_t N = static_cast<size_t>(phase_buf.size);

        switch (cfg.phase_format) {
            case deepcgh::PhaseFormat::Uint8: {
                std::vector<ssize_t> hw_shape = {static_cast<ssize_t>(cfg.height),
                                                  static_cast<ssize_t>(cfg.width)};
                auto result = py::array_t<uint8_t>(hw_shape);
                auto res_buf = result.request();
                auto* out = static_cast<uint8_t*>(res_buf.ptr);
                for (size_t i = 0; i < N; ++i) {
                    float normalized = (phase_ptr[i] + static_cast<float>(M_PI)) /
                                       (2.0f * static_cast<float>(M_PI));
                    out[i] = static_cast<uint8_t>(
                        std::max(0.0f, std::min(255.0f, normalized * 255.0f)));
                }
                return result;
            }
            case deepcgh::PhaseFormat::Uint16: {
                std::vector<ssize_t> hw_shape = {static_cast<ssize_t>(cfg.height),
                                                  static_cast<ssize_t>(cfg.width)};
                auto result = py::array_t<uint16_t>(hw_shape);
                auto res_buf = result.request();
                auto* out = static_cast<uint16_t*>(res_buf.ptr);
                for (size_t i = 0; i < N; ++i) {
                    float normalized = (phase_ptr[i] + static_cast<float>(M_PI)) /
                                       (2.0f * static_cast<float>(M_PI));
                    out[i] = static_cast<uint16_t>(
                        std::max(0.0f, std::min(65535.0f, normalized * 65535.0f)));
                }
                return result;
            }
            case deepcgh::PhaseFormat::Float:
            default:
                return phase;
        }
    }

    /// Run inference only, returning raw amp_0 and phi_0 numpy arrays.
    py::tuple infer_raw(
            const py::array_t<uint8_t>& rgb,
            const py::array_t<float>& depth) {
        if (!engine_->is_ready()) {
            throw std::runtime_error("Engine not initialized. Call init() first.");
        }

        const auto& cfg = engine_->config();
        const size_t H = static_cast<size_t>(cfg.height);
        const size_t W = static_cast<size_t>(cfg.width);

        const uint8_t* rgb_ptr = numpy_to_uint8_ptr(rgb, H * W * 3);
        const float* depth_ptr = numpy_to_float_ptr(depth, H * W);

        const size_t pixel_count = H * W;
        std::vector<float> raw_amp(pixel_count, 0.0f);
        std::vector<float> raw_phi(pixel_count, 0.0f);

        deepcgh::Status status = engine_->infer_only(
            rgb_ptr, depth_ptr,
            raw_amp.data(), raw_phi.data(),
            cfg.height, cfg.width);

        if (status != deepcgh::Status::OK) {
            throw std::runtime_error(
                "Inference failed: " + std::string(deepcgh::status_to_string(status)) +
                " — " + engine_->last_error());
        }

        // Return as numpy arrays with shape [1, H, W, 1] matching model output
        std::vector<ssize_t> shape_4d = {1, static_cast<ssize_t>(H), static_cast<ssize_t>(W), 1};
        auto amp_arr = py::array_t<float>(shape_4d, raw_amp.data());
        auto phi_arr = py::array_t<float>(shape_4d, raw_phi.data());

        return py::make_tuple(amp_arr, phi_arr);
    }

    /// Check if the engine is ready.
    bool is_ready() const { return engine_->is_ready(); }

    /// Shut down the engine.
    void shutdown() { engine_->shutdown(); }

    /// Get the last error message.
    std::string last_error() const { return engine_->last_error(); }

private:
    std::unique_ptr<deepcgh::EngineAPI> engine_;
};

// ---------------------------------------------------------------------------
// Module Definition
// ---------------------------------------------------------------------------

PYBIND11_MODULE(_deepcgh_engine, m) {
    m.doc() = "DeepCGHEngine — Deep-learning Computer-Generated Holography";

    // ---- Enums ----
    py::enum_<deepcgh::PhaseFormat>(m, "PhaseFormat")
        .value("Uint8",  deepcgh::PhaseFormat::Uint8)
        .value("Uint16", deepcgh::PhaseFormat::Uint16)
        .value("Float",  deepcgh::PhaseFormat::Float)
        .export_values();

    py::enum_<deepcgh::ColorSpace>(m, "ColorSpace")
        .value("RGB",   deepcgh::ColorSpace::RGB)
        .value("YCbCr", deepcgh::ColorSpace::YCbCr)
        .value("Gray",  deepcgh::ColorSpace::Gray)
        .export_values();

    py::enum_<deepcgh::ExecutionProvider>(m, "ExecutionProvider")
        .value("CPU",  deepcgh::ExecutionProvider::CPU)
        .value("CUDA", deepcgh::ExecutionProvider::CUDA)
        .value("DML",  deepcgh::ExecutionProvider::DML)
        .export_values();

    py::enum_<deepcgh::Status>(m, "Status")
        .value("OK",                deepcgh::Status::OK)
        .value("NOT_INITIALIZED",   deepcgh::Status::NOT_INITIALIZED)
        .value("MODEL_LOAD_FAILED", deepcgh::Status::MODEL_LOAD_FAILED)
        .value("INVALID_INPUT",     deepcgh::Status::INVALID_INPUT)
        .value("INFERENCE_FAILED",  deepcgh::Status::INFERENCE_FAILED)
        .value("SIZE_MISMATCH",     deepcgh::Status::SIZE_MISMATCH)
        .value("PROVIDER_UNAVAIL",  deepcgh::Status::PROVIDER_UNAVAIL)
        .value("ALLOCATION_FAILED", deepcgh::Status::ALLOCATION_FAILED)
        .export_values();

    // ---- EngineConfig ----
    py::class_<deepcgh::EngineConfig>(m, "EngineConfig")
        .def(py::init<>())
        .def_readwrite("height",             &deepcgh::EngineConfig::height)
        .def_readwrite("width",              &deepcgh::EngineConfig::width)
        .def_readwrite("num_planes",         &deepcgh::EngineConfig::num_planes)
        .def_readwrite("color_space",        &deepcgh::EngineConfig::color_space)
        .def_readwrite("norm_mean",          &deepcgh::EngineConfig::norm_mean)
        .def_readwrite("norm_std",           &deepcgh::EngineConfig::norm_std)
        .def_readwrite("provider",           &deepcgh::EngineConfig::provider)
        .def_readwrite("device_id",          &deepcgh::EngineConfig::device_id)
        .def_readwrite("enable_memory_pool", &deepcgh::EngineConfig::enable_memory_pool)
        .def_readwrite("intra_op_threads",   &deepcgh::EngineConfig::intra_op_threads)
        .def_readwrite("inter_op_threads",   &deepcgh::EngineConfig::inter_op_threads)
        .def_readwrite("phase_format",       &deepcgh::EngineConfig::phase_format)
        .def_readwrite("wavelength",         &deepcgh::EngineConfig::wavelength)
        .def_readwrite("pixel_size",         &deepcgh::EngineConfig::pixel_size)
        .def_readwrite("focal_length",       &deepcgh::EngineConfig::focal_length)
        .def_readwrite("plane_distance",     &deepcgh::EngineConfig::plane_distance)
        .def_readwrite("quantization_bits",  &deepcgh::EngineConfig::quantization_bits)
        .def_readwrite("int_factor",         &deepcgh::EngineConfig::int_factor)
        .def("validate", &deepcgh::EngineConfig::validate);

    // ---- PyDeepCGHEngine (Python-facing wrapper) ----
    py::class_<PyDeepCGHEngine>(m, "DeepCGHEngine")
        .def(py::init<>())
        .def("init",
            &PyDeepCGHEngine::init,
            py::arg("model_path"),
            py::arg("height")            = 256,
            py::arg("width")             = 256,
            py::arg("num_planes")        = 5,
            py::arg("color_space")       = "ycbcr",
            py::arg("provider")          = "cpu",
            py::arg("device_id")         = 0,
            py::arg("enable_memory_pool")= true,
            py::arg("phase_format")      = "uint8",
            py::arg("wavelength")        = 532e-6f,
            py::arg("pixel_size")        = 8e-3f,
            py::arg("focal_length")      = 200.0f,
            py::arg("plane_distance")    = 10.0f,
            py::arg("quantization_bits") = 8,
            py::arg("int_factor")        = 2,
            py::arg("norm_mean")         = 0.0f,
            py::arg("norm_std")          = 1.0f,
            py::arg("intra_op_threads")  = 4,
            py::arg("inter_op_threads")  = 1,
            R"doc(
Initialize the DeepCGH engine with a model and configuration.

Args:
    model_path: Path to the ONNX model file.
    height: Input frame height (default: 256).
    width: Input frame width (default: 256).
    num_planes: Number of depth planes (default: 5).
    color_space: Color space conversion ("ycbcr", "rgb", "gray").
    provider: Execution provider ("cpu", "cuda", "dml").
    device_id: GPU device index (default: 0).
    enable_memory_pool: Reuse IO buffers across frames (default: True).
    phase_format: Output phase format ("uint8", "uint16", "float").
    wavelength: Laser wavelength in mm (default: 532e-6).
    pixel_size: SLM pixel pitch in mm (default: 8e-3).
    focal_length: Focal length in mm (default: 200.0).
    plane_distance: Inter-plane distance in mm (default: 10.0).
    quantization_bits: SLM quantization levels (default: 8).
    int_factor: Interleave factor for U-Net (default: 2).
    norm_mean: Per-channel normalization mean (default: 0.0).
    norm_std: Per-channel normalization std (default: 1.0).
    intra_op_threads: ORT intra-op thread count (default: 4).
    inter_op_threads: ORT inter-op thread count (default: 1).

Raises:
    RuntimeError: If initialization fails.
)doc")
        .def("generate_hologram",
            &PyDeepCGHEngine::generate_hologram,
            py::arg("rgb"),
            py::arg("depth"),
            R"doc(
Generate a hologram phase map from RGB-D input.

Uses NumPy FFT for IFFT post-processing (correct and fast).

Args:
    rgb: numpy uint8 array of shape [H, W, 3].
    depth: numpy float32 array of shape [H, W].

Returns:
    numpy float32 array of shape [H, W] with phase in radians [-pi, pi].
)doc")
        .def("generate_hologram_quantized",
            &PyDeepCGHEngine::generate_hologram_quantized,
            py::arg("rgb"),
            py::arg("depth"),
            R"doc(
Generate a hologram and return quantized phase map for SLM display.

Args:
    rgb: numpy uint8 array of shape [H, W, 3].
    depth: numpy float32 array of shape [H, W].

Returns:
    numpy array with quantized phase:
      - uint8 [H, W] for PhaseFormat.Uint8
      - uint16 [H, W] for PhaseFormat.Uint16
      - float32 [H, W] for PhaseFormat.Float
)doc")
        .def("infer_raw",
            &PyDeepCGHEngine::infer_raw,
            py::arg("rgb"),
            py::arg("depth"),
            R"doc(
Run preprocessing + inference only, returning raw amp_0 and phi_0.

Args:
    rgb: numpy uint8 array of shape [H, W, 3].
    depth: numpy float32 array of shape [H, W].

Returns:
    Tuple of (amp_0, phi_0) as numpy float32 arrays [1, H, W, 1].
)doc")
        .def("is_ready", &PyDeepCGHEngine::is_ready)
        .def("shutdown", &PyDeepCGHEngine::shutdown)
        .def("last_error", &PyDeepCGHEngine::last_error);
}
