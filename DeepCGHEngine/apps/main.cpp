/**
 * @file main.cpp
 * @brief Standalone C++ demo for DeepCGHEngine — generates holograms
 *        without any Python dependency.
 *
 * Pipeline: RGB-D input → PreProcessor → InferenceCore → FFTW3 IFFT → PhaseMap
 *
 * Usage:
 *   deepcgh_demo --model <path> [options]
 *
 * Options:
 *   --model <path>     ONNX model path (required)
 *   --width <int>      Input width  (default: 256)
 *   --height <int>     Input height (default: 256)
 *   --planes <int>     Number of depth planes (default: 5)
 *   --output <path>    Output PGM file path (default: "output_phase.pgm")
 *   --benchmark <int>  Run N frames for benchmarking (default: 0, disabled)
 *   --rgb <path>       Load RGB from raw binary file (H*W*3 uint8)
 *   --depth <path>     Load depth from raw binary file (H*W float32)
 */

#include <deepcgh/EngineAPI.h>
#include <deepcgh/Types.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Command-line argument parsing
// ---------------------------------------------------------------------------

struct CliArgs {
    std::string model_path;
    int32_t     width      = 256;
    int32_t     height     = 256;
    int32_t     planes     = 5;
    std::string output     = "output_phase.pgm";
    int32_t     benchmark  = 0;
    std::string rgb_path;
    std::string depth_path;
};

static void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " --model <path> [options]\n"
        << "\nOptions:\n"
        << "  --model <path>     ONNX model path (required)\n"
        << "  --width <int>      Input width  (default: 256)\n"
        << "  --height <int>     Input height (default: 256)\n"
        << "  --planes <int>     Number of depth planes (default: 5)\n"
        << "  --output <path>    Output PGM file path (default: output_phase.pgm)\n"
        << "  --benchmark <int>  Run N frames for benchmarking (default: 0)\n"
        << "  --rgb <path>       Load RGB from raw binary file (H*W*3 uint8)\n"
        << "  --depth <path>     Load depth from raw binary file (H*W float32)\n";
}

static CliArgs parse_args(int argc, char* argv[]) {
    CliArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            args.model_path = argv[++i];
        } else if (arg == "--width" && i + 1 < argc) {
            args.width = std::atoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            args.height = std::atoi(argv[++i]);
        } else if (arg == "--planes" && i + 1 < argc) {
            args.planes = std::atoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            args.output = argv[++i];
        } else if (arg == "--benchmark" && i + 1 < argc) {
            args.benchmark = std::atoi(argv[++i]);
        } else if (arg == "--rgb" && i + 1 < argc) {
            args.rgb_path = argv[++i];
        } else if (arg == "--depth" && i + 1 < argc) {
            args.depth_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            std::exit(1);
        }
    }
    return args;
}

// ---------------------------------------------------------------------------
// File I/O helpers
// ---------------------------------------------------------------------------

static bool load_raw_rgb(const std::string& path, uint8_t* buffer, size_t expected_size) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        std::cerr << "Error: cannot open RGB file: " << path << "\n";
        return false;
    }
    ifs.read(reinterpret_cast<char*>(buffer), expected_size);
    if (static_cast<size_t>(ifs.gcount()) != expected_size) {
        std::cerr << "Error: RGB file size mismatch. Expected "
                  << expected_size << " bytes, got " << ifs.gcount() << "\n";
        return false;
    }
    return true;
}

static bool load_raw_depth(const std::string& path, float* buffer, size_t expected_size) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        std::cerr << "Error: cannot open depth file: " << path << "\n";
        return false;
    }
    ifs.read(reinterpret_cast<char*>(buffer), expected_size * sizeof(float));
    if (static_cast<size_t>(ifs.gcount()) != expected_size * sizeof(float)) {
        std::cerr << "Error: depth file size mismatch. Expected "
                  << expected_size * sizeof(float) << " bytes, got " << ifs.gcount() << "\n";
        return false;
    }
    return true;
}

static bool save_pgm(const std::string& path, const float* data, int32_t w, int32_t h) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        std::cerr << "Error: cannot create PGM file: " << path << "\n";
        return false;
    }

    // PGM P5 header
    ofs << "P5\n" << w << " " << h << "\n255\n";

    // Quantize [-pi, pi] -> [0, 255]
    const float PI = 3.14159265358979323846f;
    const float two_pi = 2.0f * PI;
    for (int32_t i = 0; i < w * h; ++i) {
        float normalized = (data[i] + PI) / two_pi;
        normalized = std::max(0.0f, std::min(1.0f, normalized));
        uint8_t pixel = static_cast<uint8_t>(normalized * 255.0f);
        ofs.write(reinterpret_cast<char*>(&pixel), 1);
    }
    return true;
}

static bool save_raw_float32(const std::string& path, const float* data, size_t count) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        std::cerr << "Error: cannot create raw file: " << path << "\n";
        return false;
    }
    ofs.write(reinterpret_cast<const char*>(data), count * sizeof(float));
    return true;
}

// ---------------------------------------------------------------------------
// Test data generation
// ---------------------------------------------------------------------------

static void generate_random_rgb(uint8_t* buffer, size_t count, unsigned int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    for (size_t i = 0; i < count; ++i) {
        buffer[i] = static_cast<uint8_t>(dist(gen));
    }
}

static void generate_random_depth(float* buffer, size_t count, unsigned int seed = 123) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.1f, 5.0f);
    for (size_t i = 0; i < count; ++i) {
        buffer[i] = dist(gen);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    CliArgs args = parse_args(argc, argv);

    if (args.model_path.empty()) {
        std::cerr << "Error: --model is required.\n";
        print_usage(argv[0]);
        return 1;
    }

    if (args.width <= 0 || args.height <= 0 || args.planes <= 0) {
        std::cerr << "Error: width, height, and planes must be positive integers.\n";
        return 1;
    }

    std::cout << "=== DeepCGHEngine Standalone Demo ===\n";
    std::cout << "Model:    " << args.model_path << "\n";
    std::cout << "Size:     " << args.width << " x " << args.height << "\n";
    std::cout << "Planes:   " << args.planes << "\n";
    std::cout << "Output:   " << args.output << "\n";
    if (args.benchmark > 0) {
        std::cout << "Bench:    " << args.benchmark << " frames\n";
    }
    std::cout << "\n";

    // -----------------------------------------------------------------------
    // Prepare input data
    // -----------------------------------------------------------------------
    const size_t rgb_size   = static_cast<size_t>(args.height) * args.width * 3;
    const size_t depth_size = static_cast<size_t>(args.height) * args.width;

    std::vector<uint8_t> rgb_data(rgb_size);
    std::vector<float>   depth_data(depth_size);

    bool rgb_loaded   = false;
    bool depth_loaded = false;

    if (!args.rgb_path.empty()) {
        rgb_loaded = load_raw_rgb(args.rgb_path, rgb_data.data(), rgb_size);
        if (!rgb_loaded) return 1;
    }
    if (!args.depth_path.empty()) {
        depth_loaded = load_raw_depth(args.depth_path, depth_data.data(), depth_size);
        if (!depth_loaded) return 1;
    }

    if (!rgb_loaded) {
        std::cout << "Generating random RGB data (" << rgb_size << " bytes)...\n";
        generate_random_rgb(rgb_data.data(), rgb_size);
    }
    if (!depth_loaded) {
        std::cout << "Generating random depth data (" << depth_size << " floats)...\n";
        generate_random_depth(depth_data.data(), depth_size);
    }

    // -----------------------------------------------------------------------
    // Initialize engine
    // -----------------------------------------------------------------------
    deepcgh::EngineConfig config;
    config.height     = args.height;
    config.width      = args.width;
    config.num_planes = args.planes;

    auto engine = deepcgh::EngineAPI::create();
    if (!engine) {
        std::cerr << "Error: failed to create EngineAPI instance.\n";
        return 1;
    }

    std::cout << "Initializing engine...\n";
    deepcgh::Status status = engine->init(args.model_path, config);
    if (status != deepcgh::Status::OK) {
        std::cerr << "Error: engine init failed: "
                  << deepcgh::status_to_string(status)
                  << " — " << engine->last_error() << "\n";
        return 1;
    }

    if (!engine->is_ready()) {
        std::cerr << "Error: engine is not ready after init.\n";
        return 1;
    }

    std::cout << "Engine initialized successfully.\n\n";

    // -----------------------------------------------------------------------
    // Single-frame generation
    // -----------------------------------------------------------------------
    const size_t pixel_count = static_cast<size_t>(args.height) * args.width;
    std::vector<float> phase_buffer(pixel_count);

    auto t0 = std::chrono::high_resolution_clock::now();

    status = engine->generate_hologram(
        rgb_data.data(), depth_data.data(),
        phase_buffer.data(),
        args.height, args.width);

    auto t1 = std::chrono::high_resolution_clock::now();

    if (status != deepcgh::Status::OK) {
        std::cerr << "Error: hologram generation failed: "
                  << deepcgh::status_to_string(status)
                  << " — " << engine->last_error() << "\n";
        engine->shutdown();
        return 1;
    }

    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Hologram generated in " << elapsed_ms << " ms\n";

    // -----------------------------------------------------------------------
    // Save output
    // -----------------------------------------------------------------------
    // Save PGM
    if (save_pgm(args.output, phase_buffer.data(), args.width, args.height)) {
        std::cout << "PGM saved to: " << args.output << "\n";
    }

    // Save raw float32
    std::string raw_path = args.output;
    auto dot_pos = raw_path.rfind('.');
    if (dot_pos != std::string::npos) {
        raw_path = raw_path.substr(0, dot_pos) + ".raw";
    } else {
        raw_path += ".raw";
    }
    if (save_raw_float32(raw_path, phase_buffer.data(), phase_buffer.size())) {
        std::cout << "Raw float32 saved to: " << raw_path << "\n";
    }

    // Print some phase statistics
    float min_phase = phase_buffer[0], max_phase = phase_buffer[0], sum = 0.0f;
    for (const auto& v : phase_buffer) {
        min_phase = std::min(min_phase, v);
        max_phase = std::max(max_phase, v);
        sum += v;
    }
    float mean_phase = sum / static_cast<float>(phase_buffer.size());
    std::cout << "Phase stats: min=" << min_phase
              << " max=" << max_phase
              << " mean=" << mean_phase << "\n";

    // -----------------------------------------------------------------------
    // Benchmark mode
    // -----------------------------------------------------------------------
    if (args.benchmark > 0) {
        std::cout << "\n--- Benchmark: " << args.benchmark << " frames ---\n";

        // Re-allocate phase buffer for each iteration
        std::vector<float> bench_phase(pixel_count);

        auto bench_start = std::chrono::high_resolution_clock::now();

        for (int32_t i = 0; i < args.benchmark; ++i) {
            status = engine->generate_hologram(
                rgb_data.data(), depth_data.data(),
                bench_phase.data(),
                args.height, args.width);

            if (status != deepcgh::Status::OK) {
                std::cerr << "Error: benchmark frame " << i << " failed: "
                          << deepcgh::status_to_string(status) << "\n";
                break;
            }
        }

        auto bench_end = std::chrono::high_resolution_clock::now();
        double bench_total_ms = std::chrono::duration<double, std::milli>(bench_end - bench_start).count();
        double bench_avg_ms   = bench_total_ms / static_cast<double>(args.benchmark);
        double fps            = 1000.0 / bench_avg_ms;

        std::cout << "Total time:  " << bench_total_ms << " ms\n";
        std::cout << "Avg latency: " << bench_avg_ms << " ms/frame\n";
        std::cout << "Avg FPS:     " << fps << "\n";
    }

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    engine->shutdown();
    std::cout << "\nDone.\n";
    return 0;
}
