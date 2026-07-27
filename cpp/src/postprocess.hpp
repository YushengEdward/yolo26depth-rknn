#pragma once

#include <cstdint>
#include <string>
#include <vector>

/// Resize depth map from model output size to target size (bilinear).
std::vector<float> resizeDepth(const float *depth, int model_w, int model_h,
                                int target_w, int target_h);

/// Print depth statistics (median, 5%-95% range).
void printStats(const float *depth, int w, int h);

/// Save colormap PNG from depth map.
/// mode: "disparity" (1/d + 2%-98% clip) or "metric" (linear).
void saveColormap(const float *depth, int w, int h,
                  const std::string &path, const std::string &mode);
