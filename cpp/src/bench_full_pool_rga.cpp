#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <mutex>
#include <numeric>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <im2d.h>
#include <rga.h>

#include "rknn.hpp"
#include "postprocess.hpp"
#include "rga_preprocess.hpp"

static void printUsage(const char *prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "  --model <path>    .rknn model path\n"
              << "  --image <path>    Input image\n"
              << "  --frames <N>      Total frames to process (default: 180)\n"
              << "  --workers <N>     Worker threads (default: 6)\n"
              << "  --warmup <N>      Warmup runs per worker (default: 3)\n"
              << "  --rga             Use RGA preprocessing instead of OpenCV\n";
}

struct Args {
    std::string model;
    std::string image;
    int frames = 180;
    int workers = 6;
    int warmup = 3;
    bool use_rga = false;
};

static Args parseArgs(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) args.model = argv[++i];
        else if (std::strcmp(argv[i], "--image") == 0 && i + 1 < argc) args.image = argv[++i];
        else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) args.frames = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--workers") == 0 && i + 1 < argc) args.workers = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) args.warmup = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--rga") == 0) args.use_rga = true;
        else if (std::strcmp(argv[i], "--help") == 0) { printUsage(argv[0]); std::exit(0); }
    }
    if (args.model.empty() || args.image.empty()) {
        printUsage(argv[0]);
        throw std::runtime_error("Missing required arguments");
    }
    return args;
}

static constexpr NpuCore CORES[] = {
    NpuCore::Core0, NpuCore::Core1, NpuCore::Core2
};

int main(int argc, char **argv) {
    try {
        Args args = parseArgs(argc, argv);

        // Probe model for input dimensions
        int input_w, input_h;
        {
            DepthModel probe(args.model, NpuCore::Auto);
            input_w = probe.input_w;
            input_h = probe.input_h;
        }
        std::cout << "Model input: " << input_w << "x" << input_h << std::endl;
        std::cout << "Preprocessing: " << (args.use_rga ? "RGA" : "OpenCV") << std::endl;

        // Load BGR image once
        cv::Mat bgr = cv::imread(args.image, cv::IMREAD_COLOR);
        if (bgr.empty()) throw std::runtime_error("Failed to load image: " + args.image);
        int orig_w = bgr.cols, orig_h = bgr.rows;
        std::cout << "Image: " << orig_w << "x" << orig_h << std::endl;

        // Output depth resize target
        int depth_out_w = orig_w;
        int depth_out_h = orig_h;

        // Workers
        std::vector<std::thread> threads;
        std::vector<double> worker_avgs(args.workers, 0.0);
        std::atomic<int> ready_count{0};
        std::mutex cout_mutex;

        auto wall_t0 = std::chrono::high_resolution_clock::now();

        for (int w = 0; w < args.workers; ++w) {
            threads.emplace_back([&, w]() {
                NpuCore core = CORES[w % 3];
                DepthModel model(args.model, core);

                // Each worker has its own RGA preprocessor (if using RGA)
                std::unique_ptr<RgaPreprocessor> rga_proc;
                if (args.use_rga) {
                    rga_proc = std::make_unique<RgaPreprocessor>(
                        orig_w, orig_h, input_w, input_h);
                }

                int total = args.frames / args.workers +
                            (w < args.frames % args.workers ? 1 : 0);
                std::vector<double> latencies;
                latencies.reserve(total);

                // Warmup
                cv::Mat tmp, tmp_rgb;
                cv::resize(bgr, tmp, cv::Size(input_w, input_h), 0, 0, cv::INTER_LINEAR);
                cv::cvtColor(tmp, tmp_rgb, cv::COLOR_BGR2RGB);
                for (int i = 0; i < args.warmup; ++i) {
                    model.infer(tmp_rgb.data, tmp_rgb.total() * tmp_rgb.elemSize());
                }

                ready_count++;

                // Process frames
                for (int i = 0; i < total; ++i) {
                    auto t0 = std::chrono::high_resolution_clock::now();

                    cv::Mat rgb_input;
                    if (args.use_rga) {
                        // RGA: hardware resize + BGR->RGB
                        rgb_input = rga_proc->process(bgr);
                    } else {
                        // OpenCV: resize + cvtColor
                        cv::Mat resized;
                        cv::resize(bgr, resized, cv::Size(input_w, input_h), 0, 0, cv::INTER_LINEAR);
                        cv::cvtColor(resized, rgb_input, cv::COLOR_BGR2RGB);
                    }

                    // Infer
                    auto depth_raw = model.infer(rgb_input.data,
                                                  rgb_input.total() * rgb_input.elemSize());

                    // Postprocess: resize depth back to original dims
                    auto depth = resizeDepth(depth_raw.data(),
                                              model.output_w, model.output_h,
                                              depth_out_w, depth_out_h);

                    auto t1 = std::chrono::high_resolution_clock::now();
                    (void)depth;
                    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    latencies.push_back(ms);
                }

                double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
                worker_avgs[w] = sum / latencies.size();
            });
        }

        for (auto &t : threads) t.join();
        auto wall_t1 = std::chrono::high_resolution_clock::now();
        double wall_sec = std::chrono::duration<double>(wall_t1 - wall_t0).count();

        double overall_fps = args.frames / wall_sec;

        std::cout << "\nFull pipeline pool (" << args.frames << " frames, "
                  << args.workers << " workers)"
                  << (args.use_rga ? " [RGA]" : " [OpenCV]") << ":\n";
        std::cout << "  Wall time:  " << wall_sec << " s\n";
        std::cout << "  Throughput: " << overall_fps << " FPS\n";
        std::cout << "  Per-worker avg latency:";
        for (int w = 0; w < args.workers; ++w) {
            std::cout << " " << worker_avgs[w] << "ms";
        }
        std::cout << std::endl;

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
