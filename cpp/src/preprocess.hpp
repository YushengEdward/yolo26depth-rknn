#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

/// Compute rect (aspect-ratio-preserving) input size matching ultralytics.
/// Rounds to nearest stride-32 multiple.
void computeRectInput(int src_h, int src_w, int model_max_dim,
                      int &out_h, int &out_w);


/// Load image and preprocess for RKNN: rect resize -> BGR2RGB -> uint8 NHWC.
/// Returns cv::Mat with data in RGB NHWC layout (1 x H x W x 3).
/// If image aspect ratio doesn't match model, falls back to stretch with warning.
cv::Mat preprocess(const cv::Mat &bgr, int target_w, int target_h);
cv::Mat preprocess(const std::string &img_path, int target_w, int target_h);

/// Load original image (for point cloud color lookup).
cv::Mat loadImage(const std::string &img_path);
