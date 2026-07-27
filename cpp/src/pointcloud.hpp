#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

/// Save colored point cloud as binary PLY.
/// Camera intrinsics: fx=fy=h, cx=w/2, cy=h/2.
/// Points with depth <= 0.1 are filtered out.
void savePointCloud(const float *depth, const cv::Mat &rgb,
                    int w, int h, const std::string &path, int downsample);

/// Save raw depth map as float32 binary.
void saveDepthBinary(const float *depth, int n, const std::string &path);
