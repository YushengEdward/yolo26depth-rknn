#pragma once

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

#include <im2d.h>
#include <rga.h>

/// RGA-accelerated BGR resize + BGR2RGB conversion.
/// Takes a BGR cv::Mat, returns RGB cv::Mat at target size.
cv::Mat rgaPreprocess(const cv::Mat &bgr, int target_w, int target_h);

/// RGA preprocessing with reusable buffer pool (zero-alloc path).
/// Allocates and imports buffers once for reuse across many frames.
class RgaPreprocessor {
public:
    RgaPreprocessor(int src_w, int src_h, int target_w, int target_h);
    ~RgaPreprocessor();

    /// Process a BGR frame -> RGB result at target size.
    /// Uses pre-imported RGA buffers (no IOCTL per frame).
    cv::Mat process(const cv::Mat &bgr);

    /// Benchmark: process() throughput.
    /// Excludes construction/destruction overhead.
    struct BenchResult {
        double avg_us;     // avg time per process() call
        double fps;        // frames per second
        double cv_us;       // comparable OpenCV time
    };
    BenchResult benchmark(const cv::Mat &bgr, int iterations = 100);

private:
    int src_w_, src_h_, src_wa_;
    int tgt_w_, tgt_h_, tgt_wa_;

    std::vector<uint8_t> src_buf_;
    std::vector<uint8_t> mid_buf_;
    std::vector<uint8_t> rgb_buf_;

    uint8_t *src_ptr_;
    uint8_t *mid_ptr_;
    uint8_t *rgb_ptr_;

    rga_buffer_handle_t h_src_ = 0;
    rga_buffer_handle_t h_mid_ = 0;
    rga_buffer_handle_t h_dst_ = 0;

    rga_buffer_t rga_src_;
    rga_buffer_t rga_mid_;
    rga_buffer_t rga_dst_;
};

/// Convenience benchmark (one-shot, includes buffer setup).
struct RgaBenchResult {
    double rga_us;
    double cv_us;
    double speedup;
};
RgaBenchResult benchRgaVsCv(const cv::Mat &bgr, int target_w, int target_h,
                             int iterations = 100);
