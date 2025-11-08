# Makefile for zipenhancer-rust with ONNX Runtime integration

# Configuration
CARGO := cargo
PROJECT_NAME := zipenhancer-rust
TARGET_DIR := target/debug
RELEASE_TARGET_DIR := target/release

# ONNX Runtime paths
ORT_SDK_DIR := ./lib/onnxruntime_sdk
ORT_LIB_DIR := ./lib
ORT_DYLIB := libonnxruntime.1.24.0.dylib

# Environment variables for ONNX Runtime
export ORT_STRATEGY := system
export ORT_LIB_LOCATION := $(shell pwd)/$(ORT_SDK_DIR)
export DYLD_LIBRARY_PATH := $(shell pwd)/$(ORT_SDK_DIR)/lib:$(shell pwd)/$(ORT_LIB_DIR):$(DYLD_LIBRARY_PATH)

# Default target
.PHONY: all
all: build

# Build debug version
.PHONY: build
build:
	@echo "=== Building $(PROJECT_NAME) (Debug) ==="
	@echo "ORT_STRATEGY: $(ORT_STRATEGY)"
	@echo "ORT_LIB_LOCATION: $(ORT_LIB_LOCATION)"
	@echo "DYLD_LIBRARY_PATH: $(DYLD_LIBRARY_PATH)"
	$(CARGO) build

# Build debug version
.PHONY: build-lib
build-lib:
	@echo "=== Building $(PROJECT_NAME) (Debug) ==="
	@echo "ORT_STRATEGY: $(ORT_STRATEGY)"
	@echo "ORT_LIB_LOCATION: $(ORT_LIB_LOCATION)"
	@echo "DYLD_LIBRARY_PATH: $(DYLD_LIBRARY_PATH)"
	$(CARGO) build --lib


# Build release version
.PHONY: release
release:
	@echo "=== Building $(PROJECT_NAME) (Release) ==="
	@echo "ORT_STRATEGY: $(ORT_STRATEGY)"
	@echo "ORT_LIB_LOCATION: $(ORT_LIB_LOCATION)"
	@echo "DYLD_LIBRARY_PATH: $(DYLD_LIBRARY_PATH)"
	$(CARGO) build --release

# Clean build artifacts
.PHONY: clean
clean:
	@echo "=== Cleaning build artifacts ==="
	$(CARGO) clean

# Run tests
.PHONY: test
test:
	@echo "=== Running tests ==="
	$(CARGO) test

# Run clippy (linting)
.PHONY: clippy
clippy:
	@echo "=== Running Clippy ==="
	$(CARGO) clippy -- -D warnings

# Run clippy for library only
.PHONY: clippy-lib
clippy-lib:
	@echo "=== Running Clippy (Library only) ==="
	$(CARGO) clippy --lib

# Format code
.PHONY: fmt
fmt:
	@echo "=== Formatting code ==="
	$(CARGO) fmt

# Check code without building
.PHONY: check
check:
	@echo "=== Checking code ==="
	$(CARGO) check

# Run with test-only flag
.PHONY: test-run
test-run:
	@echo "=== Running test-only mode ==="
	$(CARGO) run --bin zipenhancer -- --input dummy.wav --test-only --verbose

# Run with custom parameters (use: make run INPUT=file.wav OUTPUT=out.wav MODEL=model.onnx CONFIG=config.toml)
.PHONY: run
run:
	@echo "=== Running zipenhancer ==="
	@if [ -z "$(INPUT)" ]; then \
		echo "❌ INPUT parameter is required. Usage: make run INPUT=file.wav OUTPUT=out.wav MODEL=model.onnx CONFIG=config.toml"; \
		exit 1; \
	fi; \
	OUTPUT_ARG=""; \
	if [ -n "$(OUTPUT)" ]; then \
		OUTPUT_ARG="--output $(OUTPUT)"; \
	fi; \
	MODEL_ARG=""; \
	if [ -n "$(MODEL)" ]; then \
		MODEL_ARG="--model $(MODEL)"; \
	fi; \
	CONFIG_ARG=""; \
	if [ -n "$(CONFIG)" ]; then \
		CONFIG_ARG="--config $(CONFIG)"; \
	fi; \
	echo "Input: $(INPUT)"; \
	if [ -n "$(OUTPUT)" ]; then echo "Output: $(OUTPUT)"; else echo "Output: (default)"; fi; \
	if [ -n "$(MODEL)" ]; then echo "Model: $(MODEL)"; else echo "Model: (default)"; fi; \
	if [ -n "$(CONFIG)" ]; then echo "Config: $(CONFIG)"; else echo "Config: (default)"; fi; \
	echo "Working directory: $(shell pwd)"; \
	echo "DYLD_LIBRARY_PATH: $(DYLD_LIBRARY_PATH)"; \
	DYLD_LIBRARY_PATH="$(shell pwd)/$(ORT_SDK_DIR)/lib:$(shell pwd)/$(ORT_LIB_DIR):$(DYLD_LIBRARY_PATH)" $(CARGO) run --bin zipenhancer -- --input "$(INPUT)" $$OUTPUT_ARG $$MODEL_ARG $$CONFIG_ARG --verbose

# Run with verbose output
.PHONY: run-verbose
run-verbose:
	@echo "=== Running with verbose output ==="
	$(CARGO) run --bin zipenhancer -- --input audio_samples/speech_with_noise.wav --verbose

# Run with default example
.PHONY: run-example
run-example:
	@echo "=== Running with example audio ==="
	$(CARGO) run --bin zipenhancer -- --input ./audio_samples/noise1.wav --output examples_output/enhanced_speech.wav --model ../model/ZipEnhancer_ONNX/ZipEnhancer.onnx --verbose

# Run with config file
.PHONY: run-with-config
run-with-config:
	@echo "=== Running with config file ==="
	@if [ ! -f "config.toml" ]; then \
		echo "❌ config.toml not found. Creating default config..."; \
		make create-config; \
	fi; \
	$(CARGO) run --bin zipenhancer -- --config config.toml --input ./audio_samples/noise1.wav --output examples_output/enhanced_with_config.wav --verbose

# Create default config file
.PHONY: create-config
create-config:
	@echo "=== Creating default config file ==="
	@if [ -f "config.toml" ]; then \
		echo "⚠️  config.toml already exists. Backup created as config.toml.bak"; \
		cp config.toml config.toml.bak; \
	fi
	@echo "Creating config.toml with default settings..."
	@printf '%s\n' '# ZipEnhancer Rust 版本配置文件' > config.toml
	@printf '%s\n' '# 这个配置文件包含了音频降噪处理器的所有参数设置' >> config.toml
	@printf '%s\n' '' >> config.toml
	@printf '%s\n' '[model]' >> config.toml
	@printf '%s\n' '# ONNX 模型文件路径' >> config.toml
	@printf '%s\n' 'path = "../model/ZipEnhancer_ONNX/ZipEnhancer.onnx"' >> config.toml
	@printf '%s\n' '# ONNX 推理最大重试次数' >> config.toml
	@printf '%s\n' 'max_retries = 3' >> config.toml
	@printf '%s\n' '# ONNX 推理线程数' >> config.toml
	@printf '%s\n' 'inference_threads = 4' >> config.toml
	@printf '%s\n' '' >> config.toml
	@printf '%s\n' '[audio]' >> config.toml
	@printf '%s\n' '# 音频采样率 (Hz)' >> config.toml
	@printf '%s\n' 'sample_rate = 16000' >> config.toml
	@printf '%s\n' '# 段间重叠比例 (0.0 - 1.0)' >> config.toml
	@printf '%s\n' 'overlap_ratio = 0.1' >> config.toml
	@printf '%s\n' '# 音频段大小（样本数）' >> config.toml
	@printf '%s\n' 'segment_size = 16000' >> config.toml
	@printf '%s\n' '' >> config.toml
	@printf '%s\n' '[processing]' >> config.toml
	@printf '%s\n' '# 启用自动增益控制 (AGC)' >> config.toml
	@printf '%s\n' 'enable_agc = true' >> config.toml
	@printf '%s\n' '# 启用性能监控' >> config.toml
	@printf '%s\n' 'enable_performance_monitoring = true' >> config.toml
	@printf '%s\n' '# 详细输出模式' >> config.toml
	@printf '%s\n' 'verbose = true' >> config.toml
	@echo "✅ config.toml created successfully"

# Validate config file
.PHONY: validate-config
validate-config:
	@echo "=== Validating config file ==="
	@if [ ! -f "config.toml" ]; then \
		echo "❌ config.toml not found. Run 'make create-config' first."; \
		exit 1; \
	fi
	@echo "Validating config.toml..."
	$(CARGO) run --bin zipenhancer -- --config config.toml --test-only --verbose

# Install dependencies
.PHONY: deps
deps:
	@echo "=== Installing dependencies ==="
	$(CARGO) fetch

# Update dependencies
.PHONY: update
update:
	@echo "=== Updating dependencies ==="
	$(CARGO) update

# Show build environment
.PHONY: env
env:
	@echo "=== ONNX Runtime Build Environment ==="
	@echo "ORT_STRATEGY: $(ORT_STRATEGY)"
	@echo "ORT_LIB_LOCATION: $(ORT_LIB_LOCATION)"
	@echo "DYLD_LIBRARY_PATH: $(DYLD_LIBRARY_PATH)"
	@echo "=================================="

# Check ONNX Runtime SDK setup
.PHONY: check-sdk
check-sdk:
	@echo "=== Checking ONNX Runtime SDK setup ==="
	@if [ -d "$(ORT_SDK_DIR)" ]; then \
		echo "✅ SDK directory exists: $(ORT_SDK_DIR)"; \
		if [ -f "$(ORT_SDK_DIR)/include/onnxruntime_c_api.h" ]; then \
			echo "✅ Header file found"; \
		else \
			echo "❌ Header file missing"; \
		fi; \
		if [ -f "$(ORT_SDK_DIR)/lib/libonnxruntime.dylib" ]; then \
			echo "✅ Library file found"; \
		else \
			echo "❌ Library file missing"; \
		fi; \
	else \
		echo "❌ SDK directory not found: $(ORT_SDK_DIR)"; \
	fi

# Development workflow: clean, check, build, test
.PHONY: dev
dev: clean check build test

# Full CI workflow: format, clippy, test, build release
.PHONY: ci
ci: fmt clippy test release

# Pack distribution files
.PHONY: pack
pack:
	@echo "=== Packing distribution files ==="
	@echo "Building release version first..."
	$(MAKE) release
	@echo "Creating distribution directory..."
	@mkdir -p dist
	@echo "Copying binary file..."
	@cp $(RELEASE_TARGET_DIR)/zipenhancer dist/
	@echo "Copying configuration file..."
	@cp config.toml dist/
	@if [ -d "audio_samples" ]; then \
		echo "Copying audio samples..."; \
		cp -r audio_samples dist/; \
	fi
	@if [ -d "lib" ]; then \
		echo "Copying library files..."; \
		cp -r lib dist/; \
	fi
	@if [ -d "model" ]; then \
		echo "Copying model files..."; \
		cp -r model dist/; \
	fi
	@echo "Creating distribution script..."
	@cp zipenhancer.sh dist/
	@sed -i.bak 's|ZIPENHANCER_BIN="$$SCRIPT_DIR/target/release/zipenhancer"|ZIPENHANCER_BIN="$$SCRIPT_DIR/./zipenhancer"|' dist/zipenhancer.sh
	@rm -f dist/zipenhancer.sh.bak
	@chmod +x dist/zipenhancer.sh
	@echo "✅ Distribution packed successfully in dist/"
	@echo "Contents:"
	@ls -la dist/

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  build          - Build debug version"
	@echo "  release        - Build release version"
	@echo "  clean          - Clean build artifacts"
	@echo "  test           - Run tests"
	@echo "  clippy         - Run clippy linting"
	@echo "  clippy-lib     - Run clippy on library only"
	@echo "  fmt            - Format code"
	@echo "  check          - Check code without building"
	@echo "  run            - Run application (specify INPUT, optional OUTPUT, MODEL and CONFIG)"
	@echo "                  Usage: make run INPUT=file.wav OUTPUT=out.wav MODEL=model.onnx CONFIG=config.toml"
	@echo "  run-example    - Run with default example audio file"
	@echo "  run-with-config- Run with config file (creates default if not exists)"
	@echo "  create-config  - Create default config.toml file"
	@echo "  validate-config- Validate existing config.toml file"
	@echo "  test-run       - Run application in test-only mode"
	@echo "  run-verbose    - Run application with verbose output"
	@echo "  deps           - Install dependencies"
	@echo "  update         - Update dependencies"
	@echo "  env            - Show build environment variables"
	@echo "  check-sdk      - Check ONNX Runtime SDK setup"
	@echo "  pack           - Pack distribution files to dist/ directory"
	@echo "  dev            - Development workflow (clean, check, build, test)"
	@echo "  ci             - CI workflow (fmt, clippy, test, release)"
	@echo "  help           - Show this help message"