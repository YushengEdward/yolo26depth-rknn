#include "postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

// ============================================================================
// Bilinear resize for f32 depth maps
// ============================================================================

std::vector<float> resizeDepth(const float *depth, int model_w, int model_h,
                                int target_w, int target_h) {
    if (model_w == target_w && model_h == target_h) {
        return std::vector<float>(depth, depth + target_w * target_h);
    }


    std::vector<float> dst(target_w * target_h);
    double sx = static_cast<double>(model_w) / target_w;
    double sy = static_cast<double>(model_h) / target_h;

    for (int y = 0; y < target_h; ++y) {
        double sy_f = (y + 0.5) * sy - 0.5;
        int y0 = std::max(0, static_cast<int>(sy_f));
        int y1 = std::min(y0 + 1, model_h - 1);
        double fy = sy_f - y0;

        for (int x = 0; x < target_w; ++x) {
            double sx_f = (x + 0.5) * sx - 0.5;
            int x0 = std::max(0, static_cast<int>(sx_f));
            int x1 = std::min(x0 + 1, model_w - 1);
            double fx = sx_f - x0;

            double v = depth[y0 * model_w + x0] * (1.0 - fx) * (1.0 - fy)
                     + depth[y0 * model_w + x1] * fx * (1.0 - fy)
                     + depth[y1 * model_w + x0] * (1.0 - fx) * fy
                     + depth[y1 * model_w + x1] * fx * fy;
            dst[y * target_w + x] = static_cast<float>(v);
        }
    }
    return dst;
}

// ============================================================================
// Statistics
// ============================================================================

static float percentile(std::vector<float> sorted, float p) {
    if (sorted.empty()) return 0.0f;
    float idx = p / 100.0f * (sorted.size() - 1);
    int lo = static_cast<int>(idx);
    int hi = std::min(lo + 1, static_cast<int>(sorted.size()) - 1);
    float frac = idx - lo;
    return sorted[lo] * (1.0f - frac) + sorted[hi] * frac;
}

void printStats(const float *depth, int w, int h) {
    std::vector<float> valid;
    valid.reserve(w * h);
    for (int i = 0; i < w * h; ++i) {
        if (depth[i] > 0.0f) valid.push_back(depth[i]);
    }
    if (valid.empty()) {
        std::cout << "  No valid depth values!" << std::endl;
        return;
    }
    std::sort(valid.begin(), valid.end());
    float median = valid[valid.size() / 2];
    float p5 = percentile(valid, 5.0f);
    float p95 = percentile(valid, 95.0f);
    std::cout << "  Depth median: " << median << " m" << std::endl;
    std::cout << "  Depth 5%-95%: " << p5 << " ~ " << p95 << " m" << std::endl;
}

// ============================================================================
// JET colormap
// ============================================================================

static void jetColor(float t, uint8_t &r, uint8_t &g, uint8_t &b) {
    auto clamp01 = [](float v) { return std::max(0.0f, std::min(1.0f, v)); };
    r = static_cast<uint8_t>(clamp01(1.5f - std::abs(4.0f * t - 3.0f)) * 255.0f + 0.5f);
    g = static_cast<uint8_t>(clamp01(1.5f - std::abs(4.0f * t - 2.0f)) * 255.0f + 0.5f);
    b = static_cast<uint8_t>(clamp01(1.5f - std::abs(4.0f * t - 1.0f)) * 255.0f + 0.5f);
}

// ============================================================================
// Colormap generation
// ============================================================================

static void disparityToPixels(const float *depth, int n, std::vector<uint8_t> &pixels) {
    std::vector<float> valid_disparity;
    valid_disparity.reserve(n);
    std::vector<float> disparity(n, 0.0f);

    for (int i = 0; i < n; ++i) {
        if (depth[i] > 0.0f) {
            float d = 1.0f / depth[i];
            disparity[i] = d;
            valid_disparity.push_back(d);
        }
    }

    std::sort(valid_disparity.begin(), valid_disparity.end());
    float lo = percentile(valid_disparity, 2.0f);
    float hi = percentile(valid_disparity, 98.0f);
    float range = std::max(hi - lo, 1e-6f);

    pixels.resize(n * 3);
    for (int i = 0; i < n; ++i) {
        if (depth[i] <= 0.0f) {
            pixels[i * 3] = 0;
            pixels[i * 3 + 1] = 0;
            pixels[i * 3 + 2] = 0;
        } else {
            float t = std::max(0.0f, std::min(1.0f, (disparity[i] - lo) / range));
            uint8_t r, g, b;
            jetColor(t, r, g, b);
            pixels[i * 3] = r;
            pixels[i * 3 + 1] = g;
            pixels[i * 3 + 2] = b;
        }
    }
}

static void metricToPixels(const float *depth, int n, std::vector<uint8_t> &pixels) {
    float min_d = std::numeric_limits<float>::max();
    float max_d = std::numeric_limits<float>::lowest();
    for (int i = 0; i < n; ++i) {
        if (depth[i] > 0.0f) {
            min_d = std::min(min_d, depth[i]);
            max_d = std::max(max_d, depth[i]);
        }
    }
    if (min_d > max_d) { min_d = 0.0f; max_d = 1.0f; }
    float range = std::max(max_d - min_d, 1e-8f);

    pixels.resize(n * 3);
    for (int i = 0; i < n; ++i) {
        float t = std::max(0.0f, std::min(1.0f, (depth[i] - min_d) / range));
        jetColor(t, pixels[i * 3], pixels[i * 3 + 1], pixels[i * 3 + 2]);
    }
}

void saveColormap(const float *depth, int w, int h,
                  const std::string &path, const std::string &mode) {
    int n = w * h;
    std::vector<uint8_t> pixels;

    if (mode == "disparity") {
        disparityToPixels(depth, n, pixels);
    } else {
        metricToPixels(depth, n, pixels);
    }

    cv::Mat img(h, w, CV_8UC3, pixels.data());
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    cv::imwrite(path, img);
    std::cout << "Colormap saved: " << path << std::endl;
}
