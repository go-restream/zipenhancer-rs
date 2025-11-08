# ZipEnhancer Rust 版本 - ONNX Runtime 设置指南

## 问题说明

ZipEnhancer Rust 版本依赖 ONNX Runtime 库来进行深度学习模型推理。如果在运行时找不到 ONNX Runtime 库文件，程序会显示错误信息而不会直接崩溃。

## 解决方案

### 方案 1: 使用包装脚本（推荐）

我们提供了一个友好的包装脚本 `zipenhancer.sh`，它会自动处理库路径设置和错误处理。

```bash
# 使用脚本启动（自动查找库）
./zipenhancer.sh --test-only --input dummy.wav

# 指定库路径
./zipenhancer.sh --test-only --input dummy.wav --onnx-lib /path/to/libonnxruntime.dylib

# 查看帮助
./zipenhancer.sh --help
```

### 方案 2: 直接使用二进制文件

如果您更喜欢直接使用二进制文件，需要手动设置环境变量：

```bash
# macOS/Linux
export DYLD_LIBRARY_PATH=/path/to/onnx/lib:$DYLD_LIBRARY_PATH
export ORT_STRATEGY=system
export ORT_LIB_LOCATION=/path/to/onnx/lib

# 运行程序
./target/release/zipenhancer --test-only --input dummy.wav
```

## ONNX Runtime 安装方法

### 方法 1: Homebrew（macOS）

```bash
brew install onnxruntime
```

库文件通常位于：`/opt/homebrew/lib/libonnxruntime.dylib`

### 方法 2: 从官网下载

1. 访问 [ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases)
2. 下载适合您系统的版本
3. 解压并将库文件放到合适的位置

### 方法 3: 使用项目提供的库

项目已经包含了 ONNX Runtime 库文件：

```bash
# 使用项目提供的库
./zipenhancer.sh --test-only --input dummy.wav --onnx-lib ../lib/libonnxruntime.1.24.0.dylib
```

## 常见库文件位置

- macOS (Homebrew): `/opt/homebrew/lib/libonnxruntime.dylib`
- macOS (项目提供): `../lib/libonnxruntime.1.24.0.dylib`
- Linux: `/usr/local/lib/libonnxruntime.so`
- Linux: `/usr/lib/libonnxruntime.so`

## 命令行参数

新增的 `--onnx-lib` 参数用于指定 ONNX Runtime 库文件路径：

```bash
--onnx-lib <PATH>    指定 ONNX Runtime 库文件路径
```

## 错误处理

当找不到 ONNX Runtime 库时，程序会显示：

```
=== ONNX Runtime 库未找到 ===
无法找到 ONNX Runtime 库文件。请确保已安装 ONNX Runtime 或使用 --onnx-lib 参数指定库文件路径。

解决方案:
1. 使用 Homebrew 安装: brew install onnxruntime
2. 从官网下载: https://github.com/microsoft/onnxruntime/releases
3. 使用 --onnx-lib 参数指定库文件路径:
   ./zipenhancer.sh --onnx-lib /path/to/libonnxruntime.dylib [其他参数...]

常见库文件位置:
  - /opt/homebrew/lib/libonnxruntime.dylib
  - /usr/local/lib/libonnxruntime.dylib
  - ../lib/libonnxruntime.1.24.0.dylib
============================
```

## 环境变量说明

- `DYLD_LIBRARY_PATH`: macOS 动态库搜索路径
- `LD_LIBRARY_PATH`: Linux 动态库搜索路径
- `ORT_STRATEGY`: ONNX Runtime 加载策略
- `ORT_LIB_LOCATION`: ONNX Runtime 库文件位置

## 编译要求

编译时需要设置以下环境变量：

```bash
export ORT_STRATEGY=system
export ORT_LIB_LOCATION=/path/to/onnxruntime_sdk
export DYLD_LIBRARY_PATH=/path/to/lib:$DYLD_LIBRARY_PATH

cargo build --release
```

## 故障排除

### 1. 找不到库文件

确保库文件存在且路径正确：

```bash
ls -la /path/to/libonnxruntime.dylib
```

### 2. 库文件版本不兼容

确保使用的是 ONNX Runtime 1.24.0 或更高版本。

### 3. 权限问题

确保库文件具有读取权限：

```bash
chmod 644 /path/to/libonnxruntime.dylib
```

## 示例用法

```bash
# 基本测试
./zipenhancer.sh --test-only --input dummy.wav

# 详细模式
./zipenhancer.sh --test-only --input dummy.wav --verbose

# 处理音频文件
./zipenhancer.sh \
  --input ../examples/noise1.wav \
  --output enhanced.wav \
  --model ./model/ZipEnhancer_ONNX/ZipEnhancer.onnx \
  --verbose

# 指定库路径
./zipenhancer.sh \
  --test-only \
  --input dummy.wav \
  --onnx-lib /opt/homebrew/lib/libonnxruntime.dylib \
  --verbose
```