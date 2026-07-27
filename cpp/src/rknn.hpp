#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <rknn_api.h>

enum class NpuCore {
    Auto = RKNN_NPU_CORE_AUTO,
    Core0 = RKNN_NPU_CORE_0,
    Core1 = RKNN_NPU_CORE_1,
    Core2 = RKNN_NPU_CORE_2,
    Core012 = RKNN_NPU_CORE_0_1_2,
};

class DepthModel {
public:
    DepthModel(const std::string &model_path, NpuCore core = NpuCore::Auto);
    ~DepthModel();

    DepthModel(const DepthModel &) = delete;
    DepthModel &operator=(const DepthModel &) = delete;

    DepthModel(DepthModel &&other) noexcept;
    DepthModel &operator=(DepthModel &&other) noexcept;

    std::vector<float> infer(const uint8_t *rgb_data, size_t size);

    int input_w = 0;
    int input_h = 0;
    int output_w = 0;
    int output_h = 0;

private:
    rknn_context ctx_ = 0;
};
