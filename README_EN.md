# YOLO26-Depth RKNN

Monocular depth estimation with YOLO26-Depth on RK3588 NPU.

PT -> ONNX -> RKNN -> NPU inference pipeline, supporting n/s/m/l/x model sizes.

## Quick Start

```bash
# 1. PT -> ONNX (run on PC)
python scripts/export.py --model models/yolo26n-depth.pt --imgsz 640

# 2. ONNX -> RKNN (run on PC with rknn-toolkit2)
python scripts/convert.py --model models/yolo26n-depth.onnx

# 3. RKNN inference (run on RK3588)
python scripts/inference_rknn.py --model models/yolo26n-depth-float.rknn --image bus.jpg --save result.png
```

## Features

- **Export**: PT -> ONNX with ultralytics YOLO, supports rect (non-square) models
- **Conversion**: ONNX -> RKNN with rknn-toolkit2, INT8 quantization support
- **Python Inference**: RKNN NPU inference via rknn-toolkit-lite2, multi-core support
- **Onnx CPU Fallback**: ONNX Runtime CPU inference for testing
- **Depth Processing**: Colormap (disparity/metric), point cloud PLY export, depth binary save
- **C++ Port**: Full C++ implementation in cpp/, same pipeline with OpenCV + librknnrt
- **Benchmark**: Multi-threaded pool benchmark across 3 NPU cores

## Project Structure

```
scripts/             Python scripts (export, convert, inference)
models/              Pre-trained model files (.pt/.onnx/.rknn)
examples/            Benchmark examples
cpp/                 C++ port (CMake + OpenCV + librknnrt)
yolo26depth_rknn/    Shared Python utilities
assets/              Demo images and results
```

## Benchmarks

RK3588 NPU, measured with bus.jpg (640 input, with NPU upsample).

### n model (640x640)

| Workers | Python FPS | Python Latency | C++ FPS | C++ Latency |
|---------|-----------|---------------|---------|-------------|
| 1 | 9.1 | 109.8 ms | 10.4 | 96.6 ms |
| 3 | 26.0 | 112.9 ms | 27.2 | 107.7 ms |
| 6 | 31.0 | 188.7 ms | 30.8 | 185.3 ms |

### s model (640x640)

| Workers | Python FPS | Python Latency | C++ FPS | C++ Latency |
|---------|-----------|---------------|---------|-------------|
| 1 | 7.6 | 131.4 ms | 7.7 | 129.6 ms |
| 3 | 18.8 | 154.8 ms | 19.7 | 148.6 ms |
| 6 | 21.4 | 279.1 ms | 21.0 | 269.3 ms |

### Python vs C++ Speedup (n model)

| Workers | Python | C++ | Speedup |
|---------|--------|-----|---------|
| 1 | 109.8 ms | 96.6 ms | 1.14x |
| 3 | 112.9 ms | 107.7 ms | 1.05x |
| 6 | 188.7 ms | 185.3 ms | 1.02x |

C++ is faster for single-core inference. Multi-core throughput is similar since NPU compute is the bottleneck.

## Requirements

- RK3588 board with librknnrt.so and rknn-toolkit-lite2
- Python 3.10+ with requirements.txt
- For ONNX conversion: rknn-toolkit2 on x86 PC
- For C++ build: cmake, opencv, librknnrt
