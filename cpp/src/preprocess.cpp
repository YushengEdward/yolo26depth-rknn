#include "preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

static constexpr int STRIDE = 32;

static int align32(float x) {
    return std::max(static_cast<int>(std::round(x / STRIDE) * STRIDE), 32);
}

void computeRectInput(int src_h, int src_w, int model_max_dim,
                      int &out_h, int &out_w) {
    float scale = static_cast<float>(model_max_dim) / std::max(src_h, src_w);
    out_h = align32(src_h * scale);
    out_w = align32(src_w * scale);
}


cv::Mat preprocess(const cv::Mat &bgr, int target_w, int target_h) {
    int src_h = bgr.rows;
    int src_w = bgr.cols;
    int model_max_dim = std::max(target_h, target_w);

    int rect_h, rect_w;
    computeRectInput(src_h, src_w, model_max_dim, rect_h, rect_w);

    cv::Mat resized;
    if (rect_h == target_h && rect_w == target_w) {
        cv::resize(bgr, resized, cv::Size(target_w, target_h), 0, 0, cv::INTER_LINEAR);
    } else {
        std::cerr << "Warning: Image aspect ratio (" << src_w << ":" << src_h
                  << ") does not match model aspect ratio (" << target_w << ":" << target_h
                  << "). Rect input would be " << rect_w << "x" << rect_h
                  << " but model expects " << target_w << "x" << target_h << "."
                  << std::endl;
        cv::resize(bgr, resized, cv::Size(target_w, target_h), 0, 0, cv::INTER_LINEAR);
    }

    // BGR -> RGB, keep NHWC layout
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    std::cout << "Preprocess: " << src_w << "x" << src_h << " -> "
              << rgb.cols << "x" << rgb.rows << " (bilinear)" << std::endl;

    return rgb; // uint8, H x W x 3, NHWC
}

cv::Mat preprocess(const std::string &img_path, int target_w, int target_h) {
    cv::Mat bgr = cv::imread(img_path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        throw std::runtime_error("Failed to load image: " + img_path);
    }
    return preprocess(bgr, target_w, target_h);
}

cv::Mat loadImage(const std::string &img_path) {
    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        throw std::runtime_error("Failed to load image: " + img_path);
    }
    return img;
}
