#include "pointcloud.hpp"

#include <cstring>
#include <fstream>
#include <iostream>

#include <opencv2/imgproc.hpp>

void savePointCloud(const float *depth, const cv::Mat &rgb,
                    int w, int h, const std::string &path, int downsample) {
    float fx = static_cast<float>(h);
    float fy = static_cast<float>(h);
    float cx = static_cast<float>(w) / 2.0f;
    float cy = static_cast<float>(h) / 2.0f;

    cv::Mat rgb_8u;
    cv::cvtColor(rgb, rgb_8u, cv::COLOR_BGR2RGB);

    // Collect vertices as flat buffer: 3xf32 + 3xu8 = 15 bytes per point
    std::vector<uint8_t> buf;
    buf.reserve(w * h * 15 / (downsample * downsample));

    for (int y = 0; y < h; y += downsample) {
        for (int x = 0; x < w; x += downsample) {
            float d = depth[y * w + x];
            if (d > 0.1f) {
                float px = static_cast<float>(x);
                float py = static_cast<float>(y);
                float x3 = (px - cx) * d / fx;
                float y3 = (py - cy) * d / fy;

                auto le = [](float v) {
                    auto bytes = reinterpret_cast<const uint8_t *>(&v);
                    return std::vector<uint8_t>(bytes, bytes + 4);
                };

                auto append = le(x3);
                buf.insert(buf.end(), append.begin(), append.end());
                append = le(y3);
                buf.insert(buf.end(), append.begin(), append.end());
                append = le(d);
                buf.insert(buf.end(), append.begin(), append.end());

                cv::Vec3b pixel = rgb_8u.at<cv::Vec3b>(y, x);
                buf.push_back(pixel[0]);
                buf.push_back(pixel[1]);
                buf.push_back(pixel[2]);
            }
        }
    }

    int n = static_cast<int>(buf.size() / 15);

    std::ofstream f(path, std::ios::binary);
    f << "ply\nformat binary_little_endian 1.0\nelement vertex " << n << "\n"
      << "property float x\nproperty float y\nproperty float z\n"
      << "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
    f.write(reinterpret_cast<const char *>(buf.data()), buf.size());
    f.close();

    std::cout << "Point cloud: " << path << " (" << n << " points)" << std::endl;
}

void saveDepthBinary(const float *depth, int n, const std::string &path) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(depth), n * sizeof(float));
    f.close();
    std::cout << "Depth saved: " << path << std::endl;
}
