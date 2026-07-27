#include "rknn.hpp"

#include <cstring>
#include <iostream>
#include <stdexcept>

static void check(int ret, const char *msg) {
    if (ret != RKNN_SUCC) {
        throw std::runtime_error(std::string(msg) + ": ret=" + std::to_string(ret));
    }
}

static rknn_tensor_attr query_attr(rknn_context ctx, rknn_query_cmd cmd, uint32_t index) {
    rknn_tensor_attr attr;
    std::memset(&attr, 0, sizeof(attr));
    attr.index = index;
    check(rknn_query(ctx, cmd, &attr, sizeof(attr)), "rknn_query");
    return attr;
}

DepthModel::DepthModel(const std::string &model_path, NpuCore core) {
    rknn_context ctx = 0;

    check(rknn_init(&ctx, const_cast<char *>(model_path.c_str()), 0, 0, nullptr),
          "rknn_init");
    ctx_ = ctx;

    if (core != NpuCore::Auto) {
        check(rknn_set_core_mask(ctx, static_cast<rknn_core_mask>(core)),
              "rknn_set_core_mask");
    }

    rknn_input_output_num io_num;
    std::memset(&io_num, 0, sizeof(io_num));
    check(rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)),
          "rknn_query io_num");

    auto in_attr = query_attr(ctx, RKNN_QUERY_INPUT_ATTR, 0);
    input_h = static_cast<int>(in_attr.dims[1]);
    input_w = static_cast<int>(in_attr.dims[2]);

    auto out_attr = query_attr(ctx, RKNN_QUERY_OUTPUT_ATTR, 0);
    output_h = static_cast<int>(out_attr.dims[2]);
    output_w = static_cast<int>(out_attr.dims[3]);

    std::cout << "Model: " << input_w << "x" << input_h << " input, "
              << output_w << "x" << output_h << " output" << std::endl;
}

DepthModel::~DepthModel() {
    if (ctx_ != 0) {
        rknn_destroy(ctx_);
    }
}

DepthModel::DepthModel(DepthModel &&other) noexcept
    : ctx_(other.ctx_),
      input_w(other.input_w), input_h(other.input_h),
      output_w(other.output_w), output_h(other.output_h) {
    other.ctx_ = 0;
}

DepthModel &DepthModel::operator=(DepthModel &&other) noexcept {
    if (this != &other) {
        if (ctx_ != 0) rknn_destroy(ctx_);
        ctx_ = other.ctx_;
        input_w = other.input_w; input_h = other.input_h;
        output_w = other.output_w; output_h = other.output_h;
        other.ctx_ = 0;
    }
    return *this;
}

std::vector<float> DepthModel::infer(const uint8_t *rgb_data, size_t size) {
    rknn_input input;
    std::memset(&input, 0, sizeof(input));
    input.index = 0;
    input.buf = const_cast<uint8_t *>(rgb_data);
    input.size = static_cast<uint32_t>(size);
    input.pass_through = 0;
    input.type = RKNN_TENSOR_UINT8;
    input.fmt = RKNN_TENSOR_NHWC;

    check(rknn_inputs_set(ctx_, 1, &input), "rknn_inputs_set");
    check(rknn_run(ctx_, nullptr), "rknn_run");

    rknn_output output;
    std::memset(&output, 0, sizeof(output));
    output.want_float = 1;
    output.is_prealloc = 0;

    check(rknn_outputs_get(ctx_, 1, &output, nullptr), "rknn_outputs_get");

    size_t elem_count = output.size / sizeof(float);
    std::vector<float> result(elem_count);
    std::memcpy(result.data(), output.buf, output.size);

    rknn_outputs_release(ctx_, 1, &output);

    return result;
}
