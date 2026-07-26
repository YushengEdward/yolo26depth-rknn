# YOLO26-Depth RKNN

YOLO26-Depth 单目深度估计模型在 RK3588 NPU 上的完整部署方案。

**PT → ONNX → RKNN → NPU 推理**，支持 n/s/m/l/x 五种模型和 640/768/960/1280 四种分辨率。

## 快速开始

```bash
# 1. 导出 ONNX（x86 开发机，本地无 .pt 时自动从 ultralytics 官方下载）
python export.py --model yolo26n-depth.pt --imgsz 640

# 2. 转换 RKNN（x86 开发机）
python convert.py --model yolo26n-depth.onnx

# 3. 推理（RK3588 板端）
python inference_rknn.py --model yolo26n-depth-float.rknn --image bus.jpg --save result.png
```

## 环境

### x86 开发机（导出 + 转换）

```bash
pip install -r requirements.txt
```

| 依赖 | 版本 |
|------|------|
| rknn-toolkit2 | ≥ 2.3.0 |
| torch | ≥ 2.0 |
| onnx opset | 19 |

### RK3588 板端（推理）

```bash
pip3 install opencv-python numpy onnxruntime
# rknn-toolkit-lite2 通常预装
```

## 模型导出

本地不存在的官方模型名（`yolo26{n,s,m,l,x}-depth.pt`）会由 ultralytics 自动下载。

### 单个模型
```bash
python export.py --model yolo26x-depth.pt --imgsz 768
```

### 全部模型
```bash
python export.py --all --imgsz 640
```

### 多分辨率
```bash
python export.py --model yolo26x-depth.pt --imgsz 640 768 960 1280
```

### 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | `.pt` 模型路径 | 必填（或 `--all`） |
| `--all` | 导出所有模型 | 否 |
| `--imgsz` | 输入分辨率，可指定多个 | 640 |
| `--output` | 输出目录 | 当前目录 |
| `--force` | 强制重新导出 | 否 |

### 输出命名

| 输入 | 输出 |
|------|------|
| `yolo26n-depth.pt` + `--imgsz 640` | `yolo26n-depth.onnx` |
| `yolo26n-depth.pt` + `--imgsz 768` | `yolo26n-depth_768.onnx` |
| `yolo26x-depth.pt` + `--imgsz 1280` | `yolo26x-depth_1280.onnx` |

## RKNN 转换

### 基础转换（FLOAT16）
```bash
python convert.py --model yolo26n-depth.onnx
```

### 批量转换
```bash
python convert.py --model yolo26n-depth.onnx yolo26s-depth.onnx yolo26x-depth.onnx
```

### INT8 量化
```bash
python convert.py --model yolo26n-depth.onnx --quantize --dataset datasets.txt
```

### 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | ONNX 模型路径（可多个） | 必填 |
| `--quantize` | 启用 INT8 量化 | 否 |
| `--dataset` | 量化校准数据集 | `datasets.txt` |
| `--test-image` | 验证图片路径 | 无 |
| `--output-dir` | 输出目录 | 当前目录 |
| `--force` | 强制重新转换 | 否 |

### 注意事项

- **opset**: ONNX 必须是 opset ≤ 19（`export.py` 已自动设置）
- **DISABLE_RULES**: 禁用了 `fuse_exmatmul_add_mul_exsoftmax13_exmatmul_to_sdpa`，否则 build 报 `IndexError`
- **归一化**: RKNN 通过 `std_values=[[255,255,255]]` 内部 `/255`，推理侧保持 `uint8`

## 板端推理

### Python RKNN

```bash
# 单次推理 + 保存叠加图
python inference_rknn.py --model yolo26n-depth-float.rknn \
    --image bus.jpg --save result.png

# 保存热力图 + 原始深度
python inference_rknn.py --model yolo26n-depth-float.rknn \
    --image bus.jpg --save-heat heatmap.png --save-depth depth.npy

# Benchmark
python inference_rknn.py --model yolo26n-depth-float.rknn \
    --image bus.jpg --benchmark 100
```

### Python ONNX（CPU 对比）

```bash
# 自动从 ONNX 元数据检测输入尺寸
python inference_onnx.py --model yolo26n-depth_768.onnx \
    --image bus.jpg --save result.png

# Benchmark
python inference_onnx.py --model yolo26n-depth.onnx \
    --image bus.jpg --benchmark 100
```

### 公共参数

| 参数 | 说明 |
|------|------|
| `--model` | 模型路径 (`.rknn` / `.onnx`) |
| `--image` | 输入图片 |
| `--save` | 保存叠加图（原图 + 热力图混合） |
| `--save-heat` | 保存独立热力图 |
| `--save-depth` | 保存原始深度图 (`.npy`) |
| `--mode` | `disparity`（默认）或 `metric` |
| `--benchmark N` | Benchmark N 次 |
| `--warmup N` | 预热次数（默认 3） |
| `--imgsz` | 输入尺寸（自动检测，可手动覆盖） |

### Rust

需要板端存在 `librknnrt.so`（通常在 `/usr/lib/`，链接由 `#[link(name = "rknnrt")]` 声明）。

```bash
# 编译（RK3588 板端）
cargo build --release

# 推理 + 点云
./target/release/yolo26depth-rknn \
    --model yolo26n-depth-float.rknn \
    --image bus.jpg \
    --output depth_color.png \
    --save-ply pointcloud.ply

# Benchmark
./target/release/yolo26depth-rknn \
    --model yolo26n-depth-float.rknn \
    --image bus.jpg \
    --output depth_color.png \
    --benchmark 100
```

### Rust 参数

| 参数 | 说明 |
|------|------|
| `--model` | `.rknn` 模型路径 |
| `--image` | 输入图片 |
| `--output` | 输出深度热力图 |
| `--mode` | `disparity`（默认）或 `metric` |
| `--benchmark N` | Benchmark N 次 |
| `--save-depth path` | 保存原始深度图（float32 二进制） |
| `--save-ply path` | 保存彩色点云 PLY |
| `--pc-downsample N` | 点云下采样因子（默认 2） |

## 点云生成

```bash
python depth_to_pointcloud.py --depth depth.npy --image bus.jpg \
    --output pointcloud.ply --downsample 2
```

| 参数 | 说明 |
|------|------|
| `--depth` | 深度图 (`.npy` 或 raw float32 二进制) |
| `--image` | 原始图片 |
| `--output` | 输出 `.ply` 文件 |
| `--downsample` | 下采样因子（1=全量，2=半量） |
| `--max-depth` | 最大深度过滤（米） |
| `--min-depth` | 最小深度过滤（米，默认 0.1） |
| `--fx/--fy/--cx/--cy` | 相机内参（默认 fx=fy=h, cx=w/2, cy=h/2） |

## 性能

RK3588 实测（bus.jpg，FLOAT16，warmup 3 + 平均 10~20 次）：

| 模型 | 分辨率 | RKNN NPU | FPS |
|------|--------|----------|-----|
| n | 640 | 224 ms | 4.5 |
| s | 640 | 246 ms | 4.1 |
| m | 640 | 349 ms | 2.9 |
| l | 640 | 402 ms | 2.5 |
| x | 640 | 702 ms | 1.4 |
| n | 768 | 215 ms | 4.7 |
| s | 768 | 302 ms | 3.3 |
| x | 768 | 1079 ms | 0.9 |
| x | 960 | 1722 ms | 0.6 |
| x | 1280 | 3030 ms | 0.3 |

对比 ONNX CPU（RK3588 A76）：x/640 为 2748 ms，NPU 加速约 3.7x。

> 注：板端 librknnrt 为 1.6.0 时会提示与模型 toolkit 2.3.2 版本不匹配（仅警告，可正常运行）。升级 `/usr/lib/librknnrt.so` 至 2.3.x 可能进一步提速。

## 模型管线

```
输入 BGR uint8 (任意尺寸)
  → cv2.resize → imgsz×imgsz
  → BGR → RGB, uint8 NHWC
  → RKNN NPU (内部 /255 归一化)
  → [1,1,imgsz,imgsz] float32 米制深度
  → cv2.resize → 原图尺寸
  → 热力图 / 点云
```

模型内部已烘焙 `exp(clamp(log_depth))` 和 4x 上采样，NPU 输出即为正的米制深度值。

## 文件结构

```
├── export.py                  # PT → ONNX 导出
├── convert.py                 # ONNX → RKNN 转换
├── inference_rknn.py          # Python RKNN 推理
├── inference_onnx.py          # Python ONNX 推理 (CPU)
├── depth_to_pointcloud.py     # 深度图 → 点云 PLY
├── yolo26depth_rknn/          # 共享工具模块
│   ├── __init__.py
│   └── utils.py               # colorize_depth、imgsz 检测等
├── src/                       # Rust RKNN 推理
│   ├── main.rs                # CLI 入口（推理 / benchmark）
│   ├── lib.rs                 # 库导出
│   ├── cli.rs                 # 命令行参数 (clap)
│   ├── error.rs               # 错误类型
│   ├── rknn.rs                # RKNN C 绑定 + 模型封装
│   ├── preprocess.rs          # 图像加载与缩放
│   ├── postprocess.rs         # 深度缩放、colormap、统计
│   └── pointcloud.rs          # 点云 PLY 生成
├── Cargo.toml                 # Rust 依赖
├── requirements.txt           # Python 依赖
├── LICENSE                    # MIT 开源协议
└── README.md
```

## 常见问题

### 深度值异常大（100m+）

ONNX 推理忘记 `/255.0` 归一化。RKNN 不需要（内部处理）。

### 边缘有条带

使用了 letterbox padding。本项目使用直接 `cv2.resize()`，无此问题。

### RKNN build 报 `IndexError`

需要保留 `DISABLE_RULES` 配置（`convert.py` 已内置）。

### opset 版本不匹配

```
ValueError: Unsupport onnx opset 20, need <= 19!
```

确保使用 `export.py`（已设置 `opset=19`）。

### 点云 Y 轴方向

图像坐标系 Y 向下，3D 查看器 Y 向上。在 3D 查看器中旋转 180° 即可。

### n/s 模型输出 160×160

这是旧版导出的问题。用 `export.py` 重新导出，输出即为完整的 640×640。
