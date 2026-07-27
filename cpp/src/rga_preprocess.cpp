#include "rga_preprocess.hpp"

#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include <im2d.h>
#include <rga.h>

static uint8_t *alignPage(uint8_t *p) {
    return reinterpret_cast<uint8_t *>(
        (reinterpret_cast<uintptr_t>(p) + 4095) & ~4095);
}

// ---- One-shot convenience ----

cv::Mat rgaPreprocess(const cv::Mat &bgr, int target_w, int target_h) {
    RgaPreprocessor proc(bgr.cols, bgr.rows, target_w, target_h);
    return proc.process(bgr);
}

// ---- Reusable preprocessor ----

RgaPreprocessor::RgaPreprocessor(int src_w, int src_h, int tgt_w, int tgt_h)
    : src_w_(src_w), src_h_(src_h), tgt_w_(tgt_w), tgt_h_(tgt_h)
{
    src_wa_ = (src_w + 15) & ~15;
    tgt_wa_ = (tgt_w + 15) & ~15;

    size_t src_bytes = static_cast<size_t>(src_wa_) * src_h_ * 3 + 4096;
    size_t tgt_bytes = static_cast<size_t>(tgt_wa_) * tgt_h_ * 3 + 4096;

    src_buf_.resize(src_bytes);
    mid_buf_.resize(tgt_bytes);
    rgb_buf_.resize(tgt_bytes);

    src_ptr_ = alignPage(src_buf_.data());
    mid_ptr_ = alignPage(mid_buf_.data());
    rgb_ptr_ = alignPage(rgb_buf_.data());

    h_src_ = importbuffer_virtualaddr(src_ptr_, src_wa_, src_h_, RK_FORMAT_BGR_888);
    if (h_src_ < 0) throw std::runtime_error("import src failed: " + std::to_string(h_src_));

    h_mid_ = importbuffer_virtualaddr(mid_ptr_, tgt_wa_, tgt_h_, RK_FORMAT_BGR_888);
    if (h_mid_ < 0) { releasebuffer_handle(h_src_);
        throw std::runtime_error("import mid failed: " + std::to_string(h_mid_)); }

    h_dst_ = importbuffer_virtualaddr(rgb_ptr_, tgt_wa_, tgt_h_, RK_FORMAT_RGB_888);
    if (h_dst_ < 0) { releasebuffer_handle(h_src_); releasebuffer_handle(h_mid_);
        throw std::runtime_error("import dst failed: " + std::to_string(h_dst_)); }

    rga_src_ = wrapbuffer_handle(h_src_, src_w_, src_h_, RK_FORMAT_BGR_888, src_wa_, src_h_);
    rga_mid_ = wrapbuffer_handle(h_mid_, tgt_w_, tgt_h_, RK_FORMAT_BGR_888, tgt_wa_, tgt_h_);
    rga_dst_ = wrapbuffer_handle(h_dst_, tgt_w_, tgt_h_, RK_FORMAT_RGB_888, tgt_wa_, tgt_h_);
}

RgaPreprocessor::~RgaPreprocessor() {
    if (h_src_) releasebuffer_handle(h_src_);
    if (h_mid_) releasebuffer_handle(h_mid_);
    if (h_dst_) releasebuffer_handle(h_dst_);
}

cv::Mat RgaPreprocessor::process(const cv::Mat &bgr) {
    // Copy source into aligned buffer (row by row)
    size_t row_copy = static_cast<size_t>(src_w_) * 3;
    for (int y = 0; y < src_h_; ++y) {
        std::memcpy(src_ptr_ + y * src_wa_ * 3, bgr.ptr(y), row_copy);
    }

    IM_STATUS ret = imresize(rga_src_, rga_mid_, 0.0, 0.0, IM_INTERP_LINEAR, 1);
    if (ret != IM_STATUS_SUCCESS)
        throw std::runtime_error("RGA imresize: " + std::to_string(ret));

    ret = imcvtcolor(rga_mid_, rga_dst_, RK_FORMAT_BGR_888, RK_FORMAT_RGB_888);
    if (ret != IM_STATUS_SUCCESS)
        throw std::runtime_error("RGA imcvtcolor: " + std::to_string(ret));

    // Copy result to contiguous cv::Mat
    cv::Mat result(tgt_h_, tgt_w_, CV_8UC3);
    size_t row_out = static_cast<size_t>(tgt_w_) * 3;
    for (int y = 0; y < tgt_h_; ++y) {
        std::memcpy(result.ptr(y), rgb_ptr_ + y * tgt_wa_ * 3, row_out);
    }
    return result;
}

auto RgaPreprocessor::benchmark(const cv::Mat &bgr, int iterations) -> BenchResult {
    // Warmup
    for (int i = 0; i < 5; ++i) process(bgr);

    // Timed RGA process() calls
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) process(bgr);
    auto t1 = std::chrono::high_resolution_clock::now();
    double rga_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iterations;

    // OpenCV comparison (includes resize + cvtColor)
    cv::Mat resized, rgb;
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        cv::resize(bgr, resized, cv::Size(tgt_w_, tgt_h_), 0, 0, cv::INTER_LINEAR);
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double cv_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iterations;

    // Verify correctness
    cv::Mat rga_out = process(bgr);
    cv::resize(bgr, resized, cv::Size(tgt_w_, tgt_h_), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    int max_diff = 0, bad = 0;
    for (int y = 0; y < tgt_h_; y += 4) {
        for (int x = 0; x < tgt_w_; x += 4) {
            for (int c = 0; c < 3; ++c) {
                int d = std::abs(rga_out.ptr(y)[x * 3 + c] - rgb.ptr(y)[x * 3 + c]);
                if (d > max_diff) max_diff = d;
                if (d > 2) bad++;
            }
        }
    }

    std::cout << "  [RGA]   " << rga_us << " us (" << (1e6 / rga_us) << " FPS)" << std::endl;
    std::cout << "  [OpenCV] " << cv_us << " us (" << (1e6 / cv_us) << " FPS)" << std::endl;
    std::cout << "  [Speedup] " << (cv_us / rga_us) << "x" << std::endl;
    std::cout << "  [Correctness] max_diff=" << max_diff << " outliers=" << bad
              << " (" << (max_diff <= 3 ? "PASS" : "CHECK") << ")" << std::endl;

    return {rga_us, 1e6 / rga_us, cv_us};
}

// ---- One-shot convenience benchmark ----

RgaBenchResult benchRgaVsCv(const cv::Mat &bgr, int target_w, int target_h,
                             int iterations) {
    RgaPreprocessor proc(bgr.cols, bgr.rows, target_w, target_h);
    auto r = proc.benchmark(bgr, iterations);
    return {r.avg_us, r.cv_us, r.cv_us / r.avg_us};  // RgaBenchResult{rga_us, cv_us, speedup}
}
