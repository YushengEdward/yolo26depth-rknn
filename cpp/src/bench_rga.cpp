#include <chrono>
#include <cstring>
#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <im2d.h>
#include <rga.h>

#include "rga_preprocess.hpp"

static void printUsage(const char *prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "  --image <path>    Input image (default: ../assets/bus.jpg)\n"
              << "  --width <N>       Target width (default: 640)\n"
              << "  --height <N>      Target height (default: 480)\n"
              << "  --iters <N>       Iterations per test (default: 100)\n";
}

struct Args {
    std::string image = "../assets/bus.jpg";
    int width = 640;
    int height = 480;
    int iterations = 100;
};

static Args parseArgs(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--image") == 0 && i + 1 < argc)
            args.image = argv[++i];
        else if (std::strcmp(argv[i], "--width") == 0 && i + 1 < argc)
            args.width = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--height") == 0 && i + 1 < argc)
            args.height = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc)
            args.iterations = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            std::exit(0);
        }
    }
    return args;
}

int main(int argc, char **argv) {
    Args args = parseArgs(argc, argv);

    cv::Mat bgr = cv::imread(args.image, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        std::cerr << "Failed to load image: " << args.image << std::endl;
        return 1;
    }

    std::cout << "Image: " << args.image << " (" << bgr.cols << "x" << bgr.rows << ")\n";
    std::cout << "Target: " << args.width << "x" << args.height << "\n";
    std::cout << "Iterations: " << args.iterations << "\n\n";

    auto result = benchRgaVsCv(bgr, args.width, args.height, args.iterations);

    std::cout << "\n--- Summary ---\n";
    std::cout << "  RGA   preprocess: " << result.rga_us << " us\n";
    std::cout << "  OpenCV preprocess: " << result.cv_us << " us\n";
    std::cout << "  Speedup: " << result.speedup << "x\n";

    return 0;
}
