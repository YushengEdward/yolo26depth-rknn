#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

#include "preprocess.hpp"
#include "rknn.hpp"

static void printUsage(const char *prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "  --model <path>    .rknn model path\n"
              << "  --image <path>    Input image\n"
              << "  --frames <N>      Total frames to process (default: 180)\n"
              << "  --workers <N>     Worker threads (default: 6)\n"
              << "  --warmup <N>      Warmup runs per worker (default: 3)\n";
}

struct Args {
    std::string model;
    std::string image;
    int frames = 180;
    int workers = 6;
    int warmup = 3;
};

static Args parseArgs(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            args.model = argv[++i];
        else if (std::strcmp(argv[i], "--image") == 0 && i + 1 < argc)
            args.image = argv[++i];
        else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc)
            args.frames = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--workers") == 0 && i + 1 < argc)
            args.workers = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
            args.warmup = std::stoi(argv[++i]);
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

static constexpr NpuCore CORES[] = {
    NpuCore::Core0, NpuCore::Core1, NpuCore::Core2
};

struct WorkItem {
    std::vector<uint8_t> data;
};

struct WorkerResult {
    NpuCore core;
    int frames_processed;
    std::vector<double> latencies;
};

int main(int argc, char **argv) {
    try {
        Args args = parseArgs(argc, argv);

        // Load model once just to get input dims, then discard
        int input_w, input_h;
        {
            DepthModel probe(args.model, NpuCore::Auto);
            input_w = probe.input_w;
            input_h = probe.input_h;
        }

        // Decode + preprocess ONCE
        cv::Mat rgb = preprocess(args.image, input_w, input_h);
        size_t input_size = static_cast<size_t>(rgb.total()) * rgb.elemSize();
        std::vector<uint8_t> frame_data(rgb.data, rgb.data + input_size);

        // Synchronization
        std::queue<WorkItem> queue;
        std::mutex queue_mutex;
        std::condition_variable cv;
        std::atomic<bool> done{false};

        std::vector<std::thread> threads;
        std::vector<WorkerResult> results(args.workers);
        std::atomic<int> ready_count{0};
        std::mutex ready_mutex;
        std::condition_variable ready_cv;

        // Start workers
        for (int i = 0; i < args.workers; ++i) {
            threads.emplace_back([&, i]() {
                NpuCore core = CORES[i % 3];
                DepthModel model(args.model, core);
                std::vector<double> latencies;

                // Warmup
                for (int w = 0; w < args.warmup; ++w) {
                    model.infer(frame_data.data(), frame_data.size());
                }

                // Signal ready
                {
                    std::lock_guard<std::mutex> lock(ready_mutex);
                    ready_count++;
                }
                ready_cv.notify_one();

                // Process frames
                while (true) {
                    WorkItem item;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        cv.wait(lock, [&]() { return !queue.empty() || done; });
                        if (queue.empty() && done) break;
                        item = std::move(queue.front());
                        queue.pop();
                    }

                    auto t0 = std::chrono::high_resolution_clock::now();
                    model.infer(item.data.data(), item.data.size());
                    auto t1 = std::chrono::high_resolution_clock::now();
                    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    latencies.push_back(ms);
                }

                results[i] = {core, static_cast<int>(latencies.size()), std::move(latencies)};
            });
        }

        // Wait for all workers to finish warmup
        {
            std::unique_lock<std::mutex> lock(ready_mutex);
            ready_cv.wait(lock, [&]() { return ready_count.load() == args.workers; });
        }

        // Dispatch frames
        auto wall_t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < args.frames; ++i) {
            WorkItem item;
            item.data = frame_data; // copy
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                queue.push(std::move(item));
            }
            cv.notify_one();
        }

        // Signal done and wait for workers
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            done = true;
        }
        cv.notify_all();

        for (auto &t : threads) {
            t.join();
        }
        auto wall_t1 = std::chrono::high_resolution_clock::now();
        double wall_sec = std::chrono::duration<double>(wall_t1 - wall_t0).count();

        // Aggregate results
        std::vector<double> all_latencies;
        for (auto &r : results) {
            all_latencies.insert(all_latencies.end(),
                                  r.latencies.begin(), r.latencies.end());
        }
        std::sort(all_latencies.begin(), all_latencies.end());

        double median = all_latencies[all_latencies.size() / 2];
        double p95 = all_latencies[static_cast<size_t>(all_latencies.size() * 0.95)];

        std::cout << "\nQueue benchmark: " << args.frames << " frames, "
                  << args.workers << " workers" << std::endl;
        std::cout << "  Wall time:  " << wall_sec << " s" << std::endl;
        std::cout << "  Throughput: " << (args.frames / wall_sec) << " FPS (aggregate)"
                  << std::endl;
        std::cout << "  Latency:    median " << median << " ms, p95 " << p95 << " ms"
                  << std::endl;

        for (int i = 0; i < args.workers; ++i) {
            std::cout << "  Worker " << i << " (core" << static_cast<int>(results[i].core)
                      << "): " << results[i].frames_processed << " frames" << std::endl;
        }

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
