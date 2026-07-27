#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "postprocess.hpp"
#include "pointcloud.hpp"
#include "preprocess.hpp"
#include "rknn.hpp"

static void printUsage(const char *prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "  --model <path>           .rknn model path\n"
              << "  --image <path>           Input image\n"
              << "  --output <path>          Save colormap overlay (PNG)\n"
              << "  --mode <disparity|metric> Colormap mode (default: disparity)\n"
              << "  --benchmark <N>          Benchmark N iterations\n"
              << "  --save-depth <path>      Save raw depth (float32 binary)\n"
              << "  --save-ply <path>        Save point cloud (PLY)\n"
              << "  --pc-downsample <N>      Point cloud downsample (default: 2)\n"
              << "  --core <auto|0|1|2|012>  NPU core selection (default: auto)\n";
}

static NpuCore parseCore(const std::string &s) {
    if (s == "auto") return NpuCore::Auto;
    if (s == "0")    return NpuCore::Core0;
    if (s == "1")    return NpuCore::Core1;
    if (s == "2")    return NpuCore::Core2;
    if (s == "012")  return NpuCore::Core012;
    throw std::invalid_argument("Invalid core: " + s);
}

struct Args {
    std::string model;
    std::string image;
    std::string output;
    std::string mode = "disparity";
    int benchmark = 0;
    std::string save_depth;
    std::string save_ply;
    int pc_downsample = 2;
    NpuCore core = NpuCore::Auto;
};

static Args parseArgs(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            args.model = argv[++i];
        else if (std::strcmp(argv[i], "--image") == 0 && i + 1 < argc)
            args.image = argv[++i];
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            args.output = argv[++i];
        else if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc)
            args.mode = argv[++i];
        else if (std::strcmp(argv[i], "--benchmark") == 0 && i + 1 < argc)
            args.benchmark = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--save-depth") == 0 && i + 1 < argc)
            args.save_depth = argv[++i];
        else if (std::strcmp(argv[i], "--save-ply") == 0 && i + 1 < argc)
            args.save_ply = argv[++i];
        else if (std::strcmp(argv[i], "--pc-downsample") == 0 && i + 1 < argc)
            args.pc_downsample = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--core") == 0 && i + 1 < argc)
            args.core = parseCore(argv[++i]);
        else if (std::strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            std::exit(0);
        }
    }
    if (args.model.empty() || args.image.empty()) {
        printUsage(argv[0]);
        throw std::runtime_error("Missing required arguments");
    }
    return args;
}

static void runBenchmark(DepthModel &model, const cv::Mat &rgb, int iterations) {
    // Warmup
    std::cout << "Warmup (3)..." << std::endl;
    for (int i = 0; i < 3; ++i) {
        model.infer(rgb.data, rgb.total() * rgb.elemSize());
    }

    std::vector<double> times;
    times.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        model.infer(rgb.data, rgb.total() * rgb.elemSize());
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times.push_back(ms);
    }

    double sum = 0, min_t = times[0], max_t = times[0];
    for (double t : times) {
        sum += t;
        if (t < min_t) min_t = t;
        if (t > max_t) max_t = t;
    }
    double avg = sum / times.size();

    std::cout << "\nBenchmark (" << iterations << " iterations):" << std::endl;
    std::cout << "  Latency: " << avg << " ms (" << (1000.0 / avg) << " FPS)" << std::endl;
    std::cout << "  Min: " << min_t << " ms  Max: " << max_t << " ms" << std::endl;
}

int main(int argc, char **argv) {
    try {
        Args args = parseArgs(argc, argv);

        cv::Mat orig = loadImage(args.image);
        int orig_w = orig.cols;
        int orig_h = orig.rows;
        std::cout << "Image: " << orig_w << "x" << orig_h << std::endl;

        std::cout << "Loading: " << args.model << std::endl;
        DepthModel model(args.model, args.core);

        cv::Mat rgb = preprocess(orig, model.input_w, model.input_h);
        size_t input_size = static_cast<size_t>(rgb.total()) * rgb.elemSize();

        if (args.benchmark > 0) {
            runBenchmark(model, rgb, args.benchmark);
            return 0;
        }

        // Single inference
        auto t0 = std::chrono::high_resolution_clock::now();
        auto depth_raw = model.infer(rgb.data, input_size);
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "Inference: " << elapsed << " ms" << std::endl;

        // Resize depth to original image size
        auto depth = resizeDepth(depth_raw.data(),
                                  model.output_w, model.output_h,
                                  orig_w, orig_h);

        printStats(depth.data(), orig_w, orig_h);

        // Save outputs
        if (!args.output.empty()) {
            saveColormap(depth.data(), orig_w, orig_h, args.output, args.mode);
        }

        if (!args.save_depth.empty()) {
            saveDepthBinary(depth.data(), orig_w * orig_h, args.save_depth);
        }

        if (!args.save_ply.empty()) {
            savePointCloud(depth.data(), orig, orig_w, orig_h,
                           args.save_ply, args.pc_downsample);
        }

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
