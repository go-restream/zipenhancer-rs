#!/bin/bash

# ZipEnhancer Rust Version Startup Script
#
# This script is responsible for setting up the correct ONNX Runtime library path and launching the program

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZIPENHANCER_BIN="$SCRIPT_DIR/target/release/zipenhancer"

# Common ONNX Runtime library locations
COMMON_LIB_PATHS=(
    "/opt/homebrew/lib/libonnxruntime.dylib"
    "/usr/local/lib/libonnxruntime.dylib"
    "/usr/lib/libonnxruntime.dylib"
    "/lib/libonnxruntime.dylib"
    "lib/libonnxruntime.1.24.0.dylib"
    "lib/libonnxruntime.dylib"
    "lib/onnxruntime_sdk/lib/libonnxruntime.dylib"
)

# Function: Show friendly error messages
show_library_error() {
    echo "=== ONNX Runtime Library Not Found ==="
    echo "Unable to find ONNX Runtime library file. Please ensure ONNX Runtime is installed or use the --onnx-lib parameter to specify the library file path."
    echo
    echo "Solutions:"
    echo "1. Install using Homebrew: brew install onnxruntime"
    echo "2. Download from official website: https://github.com/microsoft/onnxruntime/releases"
    echo "3. Use --onnx-lib parameter to specify library file path:"
    echo "   $0 --onnx-lib /path/to/libonnxruntime.dylib [other parameters...]"
    echo
    echo "Common library file locations:"
    for path in "${COMMON_LIB_PATHS[@]}"; do
        echo "  - $path"
    done
    echo "============================"
    echo
}

# Function: Find ONNX Runtime library
find_onnx_library() {
    local lib_path=""
    local found_lib=false

    # Check if library path is specified in command line arguments
    local i=0
    while [[ $i -lt $# ]]; do
        local arg="${!i}"
        if [[ "$arg" == --onnx-lib ]]; then
            # --onnx-lib /path/to/lib format, need to get the next argument
            local next_index=$((i + 1))
            if [[ $next_index -lt $# ]]; then
                lib_path="${!next_index}"
                found_lib=true
                break
            fi
        elif [[ "$arg" == --onnx-lib=* ]]; then
            # --onnx-lib=/path/to/lib format
            lib_path="${arg#--onnx-lib=}"
            found_lib=true
            break
        fi
        i=$((i + 1))
    done

    if [[ "$found_lib" == true ]]; then
        if [[ -f "$lib_path" ]]; then
            echo "$(dirname "$lib_path")"
            return 0
        else
            echo "Error: Specified ONNX Runtime library file does not exist: $lib_path" >&2
            echo "Please check if the path is correct." >&2
            return 1
        fi
    fi

    # Auto-search library files
    for path in "${COMMON_LIB_PATHS[@]}"; do
        if [[ -f "$path" ]]; then
            echo "$(dirname "$path")"
            return 0
        fi
    done

    return 1
}

# Check if binary file exists
if [[ ! -f "$ZIPENHANCER_BIN" ]]; then
    # Try debug version
    ZIPENHANCER_BIN="$SCRIPT_DIR/target/debug/zipenhancer"
    if [[ ! -f "$ZIPENHANCER_BIN" ]]; then
        echo "Error: ZipEnhancer binary file not found"
        echo "Please run first: cargo build --release"
        exit 1
    fi
fi

# Find ONNX Runtime library
LIB_DIR=$(find_onnx_library "$@")

if [[ $? -ne 0 ]]; then
    show_library_error
    exit 1
fi

# Set library path environment variables
export DYLD_LIBRARY_PATH="$LIB_DIR:$DYLD_LIBRARY_PATH"
export ORT_STRATEGY=system
export ORT_LIB_LOCATION="$LIB_DIR"

# If in verbose mode, display library information
for arg in "$@"; do
    if [[ "$arg" == "-v" || "$arg" == "--verbose" ]]; then
        echo "Using ONNX Runtime library directory: $LIB_DIR"
        break
    fi
done

# Launch program
exec "$ZIPENHANCER_BIN" "$@"