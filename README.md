# YOLO26-Depth RKNN

YOLO26-Depth 单目深度估计模型在 RK3588 NPU 上的完整部署方案。

**PT → ONNX → RKNN → NPU 推理**，支持 n/s/m/l/x 五种模型和 640/768/960/1280 四种分辨率。

| 输入 | RKNN NPU 深度输出 (yolo26x, disparity 模式) |
|:---:|:---:|
| <img src="assets/bus.jpg" width="380"> | <img src="assets/depth_bus.png" width="380"> |

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

### Rect（非正方形，与 ultralytics 原版对齐）

ultralytics 原版单图推理用 **rect 模式**：保持宽高比缩放到 stride-32 对齐尺寸（如 810×1080 → 480×640），不填充也不拉伸。本项目推理端（Python/Rust）已实现相同的 rect 预处理——自动根据输入图片的宽高比计算 rect 尺寸，与模型输入尺寸匹配时零差异对齐原版。

导出与相机宽高比匹配的 rect 模型即可完全对齐原版（实测 RKNN 768×576 vs PT 原版 0.02%）。比 640×640 方形模型还快 12%：

```bash
# 4:3 相机（如 1080×810），PT 模型默认 imgsz=768 → rect 768×576
python export.py --model yolo26n-depth.pt --imgsz 768x576
python convert.py --model yolo26n-depth_768x576.onnx

# 16:9 相机（如 1920×1080），imgsz=640 → rect 352×640
python export.py --model yolo26n-depth.pt --imgsz 352x640
```

> **注意**：PT 模型默认 `imgsz=768`。要与原版逐像素一致，rect 导出时 max_dim 也用 768（如 `768x576` 对应 4:3）。用 `640x480` 推理更快但与 768 原版有分辨率差异。

推理端无需额外参数——自动从模型名或 ONNX 元数据读取输入尺寸。若图片宽高比与模型不匹配，会给出警告并回退到拉伸（有精度损失）。

### 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | `.pt` 模型路径 | 必填（或 `--all`） |
| `--all` | 导出所有模型 | 否 |
| `--imgsz` | 输入分辨率，可多个；支持 `HxW` rect（如 `640x480`） | 640 |
| `--output` | 输出目录 | 当前目录 |
| `--force` | 强制重新导出 | 否 |

### 输出命名

| 输入 | 输出 |
|------|------|
| `yolo26n-depth.pt` + `--imgsz 640` | `yolo26n-depth.onnx` |
| `yolo26n-depth.pt` + `--imgsz 768` | `yolo26n-depth_768.onnx` |
| `yolo26n-depth.pt` + `--imgsz 640x480` | `yolo26n-depth_640x480.onnx` |
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

RK3588 实测（bus.jpg，FLOAT16，librknnrt 2.3.2 + performance governor，warmup 3 + 平均 15~20 次）：

| 模型 | 分辨率 | RKNN NPU | FPS |
|------|--------|----------|-----|
| n | 640 | 94 ms | 10.6 |
| s | 640 | 128 ms | 7.8 |
| m | 640 | 238 ms | 4.2 |
| l | 640 | 285 ms | 3.5 |
| x | 640 | 603 ms | 1.7 |

对比 ONNX CPU（RK3588 A76）：x/640 为 2748 ms，NPU 加速约 4.6x。

### 性能调优（重要）

默认板卡配置下 n 模型约 224 ms，两步优化后降至 94 ms（**2.4 倍**）：

1. **升级板端 runtime**（收益最大）。旧版 librknnrt（如 1.6.0）跑 toolkit 2.3.2 编译的模型只有一半速度。从 [rknpu2 仓库](https://github.com/airockchip/rknn-toolkit2/tree/master/rknpu2/runtime/Linux/librknn_api/aarch64) 取与 toolkit 匹配的版本：
   ```bash
   sudo cp librknnrt.so /usr/lib/librknnrt.so
   ```

2. **锁定 NPU/CPU 频率**。默认 ondemand governor 空闲降频，单次推理经常跑不满：
   ```bash
   echo performance | sudo tee /sys/class/devfreq/fdab0000.npu/governor
   for p in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do
       echo performance | sudo tee $p; done
   ```
   （重启后失效，需开机脚本固化）

3. **NPU 多核**：`--core` 参数（Python: `auto/0/1/2/012`，Rust: `auto/core0/core1/core2/core012`）。单帧推理多核收益很小（约 3%）；真正的用法是**多进程/多线程各绑一个核**，吞吐接近 3 倍（实测 t6）：

   | 模型 | 单实例 | 3 实例 × 3 核聚合 | 提升 |
   |------|--------|-------------------|------|
   | n | 10.6 FPS | 27.6 FPS | 2.6x |
   | x | 1.7 FPS | 4.4 FPS | 2.7x |

   **Rust 队列版**（`examples/benchmark_pool.rs`）：帧队列 + N 个 worker 线程，每 worker 独立 RKNN 实例轮询绑核；输入只解码/预处理一次，帧为 buffer clone，测的是纯推理吞吐。n 模型 6 worker（每核 2 实例）可将 NPU 喂满至 **33.3 FPS**：

   | 模型 | workers | 吞吐 | 单帧延迟 (median) |
   |------|---------|------|-------------------|
   | n | 1 | 10.3 FPS | 96 ms |
   | n | 3 | 28.6 FPS | 102 ms |
   | n | 6 | **33.3 FPS** | 172 ms |
   | n | 9 | 31.3 FPS | 257 ms |
   | x | 3 | **4.2 FPS** | 679 ms |
   | x | 6 | 4.0 FPS | 1334 ms |

   小模型每核 2 实例可掩盖调度间隙（+17%）；大模型 NPU 已饱和，3 worker 即最优，加 worker 只涨延迟。

   ```bash
   cargo build --release --example benchmark_pool
   ./target/release/examples/benchmark_pool \
       --model yolo26n-depth-float.rknn --image bus.jpg --frames 180 --workers 6
   ```

   Python 线程池版见 `examples/benchmark_pool.py`：
   ```bash
   # 3 核聚合吞吐
   python3 examples/benchmark_pool.py --model yolo26n-depth-float.rknn \
       --image bus.jpg --frames 90

   # 单核基线对比
   python3 examples/benchmark_pool.py --model yolo26n-depth-float.rknn \
       --image bus.jpg --frames 30 --cores 0
   ```

### 关于 INT8 量化（不推荐）

实测 INT8（w8a8，32 张 COCO 图校准）速度收益明显但**深度精度损失过大**：

| 模型 | FP16 | INT8 | 提速 | INT8 平均相对误差 |
|------|------|------|------|-------------------|
| n | 94 ms | 77 ms | 1.2x | 33% |
| x | 603 ms | 279 ms | 2.2x | 52% |

原因：模型输出端有 `exp()`，log 深度上的微小量化误差会被指数放大；attention 块（逐层分析 cosine 掉到 0.76）也是重灾区。混合量化（敏感层保留 FP16）实测同样无法挽回。rk3588 不支持 w8a16。**建议始终使用 FP16（`-float.rknn`）**，`--quantize` 选项仅供实验。

## 与原版精度对比

bus.jpg（810×1080，4:3），n 模型，以 ultralytics 原版 `YOLO().predict()` 输出为基准：

| 链路 | 平均相对误差 |
|------|--------------|
| PT → ONNX（同预处理 768×576） | 0.02% |
| ONNX → RKNN FP16（同预处理） | 0.16% |
| RKNN 640×640 拉伸 vs 原版 | 11.8% |
| **RKNN 768×576 rect vs 原版** | **0.02%** |

转换链路本身近乎无损；与原版的差异几乎全部来自预处理宽高比（原版 rect vs 拉伸）。rect 导出 + 匹配宽高比的图片 = 与原版逐像素一致。

## 模型管线

```
输入 BGR uint8 (任意尺寸)
  → rect 预处理：根据图片宽高比计算 stride-32 对齐的输入尺寸
  → cv2.resize → 模型输入尺寸（保持宽高比，不填充不拉伸）
  → BGR → RGB, uint8 NHWC (RKNN) / float32 NCHW (ONNX)
  → NPU / CPU 推理
  → [1,1,H,W] float32 米制深度
  → cv2.resize → 原图尺寸
  → 热力图 / 点云
```

如果图片宽高比与模型不匹配（如方形模型处理 4:3 图片），会给出警告并回退到拉伸 resize。

模型内部已烘焙 `exp(clamp(log_depth))` 和 4x 上采样，NPU 输出即为正的米制深度值。

## 文件结构

```
├── export.py                  # PT → ONNX 导出
├── convert.py                 # ONNX → RKNN 转换
├── inference_rknn.py          # Python RKNN 推理
├── inference_onnx.py          # Python ONNX 推理 (CPU)
├── depth_to_pointcloud.py     # 深度图 → 点云 PLY
├── examples/
│   ├── benchmark_pool.rs      # Rust 帧队列多核吞吐 benchmark
│   └── benchmark_pool.py      # Python 线程池版
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

### 边缘有条带 / 深度异常区域

可能是图片宽高比与模型不匹配导致的拉伸。使用 rect 导出（`--imgsz HxW`）匹配相机宽高比即可消除。推理时若不匹配会给出警告。

### 宽高比不匹配警告

```
UserWarning: Image aspect ratio (810:1080) does not match model aspect ratio (640:640).
  Rect input would be 640x480 but model expects 640x640.
```

说明当前图片的宽高比与模型输入尺寸不一致。解决：导出 rect 模型 `--imgsz 640x480`（或根据 PT imgsz 用 `768x576`）。

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
