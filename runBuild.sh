#!/bin/bash

# runBuild.sh - Build script for HeCBench CUDA and OpenMP benchmarks
#
# This script configures and builds all CUDA and OpenMP codes from HeCBench
# using clang/clang++ for C/C++ compilation and nvcc for CUDA compilation.
#
# Build outputs:
#   - All compiled executables are placed in the top-level build/ directory
#   - Each executable is named after its project directory (e.g., lulesh-cuda)
#
# Usage:
#   ./runBuild.sh [OPTIONS]
#
# Options:
#   --clean      Remove build directory before building
#   --cuda-arch  CUDA architecture (default: 86 for sm_86)
#   --help       Show this help message

set -e  # Exit on error

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
HECBENCH_DIR="$PROJECT_ROOT/HeCBench"
BUILD_DIR="$PROJECT_ROOT/build"

# Default configuration
CUDA_ARCH="86"
CLEAN_BUILD=false

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help message
show_help() {
    cat << EOF
Usage: ./runBuild.sh [OPTIONS]

Build all CUDA and OpenMP benchmarks from HeCBench using LLVM clang/clang++.

Options:
    --clean          Remove build directory before building
    --cuda-arch ARG  CUDA compute capability (e.g., 60, 70, 75, 80, 86, 89, 90)
                     Default: 86 (for sm_86/compute_86)
    --help           Show this help message

Examples:
    ./runBuild.sh
    ./runBuild.sh --clean
    ./runBuild.sh --cuda-arch 86
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --cuda-arch)
            CUDA_ARCH="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check prerequisites
log_info "Checking prerequisites..."

if [ ! -d "$HECBENCH_DIR" ] || [ -z "$(ls -A "$HECBENCH_DIR")" ]; then
    log_error "HeCBench submodule not initialized. Run: git submodule update --init"
    exit 1
fi

# Check for required tools
command -v clang >/dev/null 2>&1 || { log_error "clang not found. Please install LLVM."; exit 1; }
command -v clang++ >/dev/null 2>&1 || { log_error "clang++ not found. Please install LLVM."; exit 1; }
command -v cmake >/dev/null 2>&1 || { log_error "cmake not found. Please install CMake."; exit 1; }

log_info "Prerequisites satisfied"
log_info "  clang:   $(clang --version | head -n1)"
log_info "  clang++: $(clang++ --version | head -n1)"
log_info "  cmake:   $(cmake --version | head -n1)"

# Clean build directory if requested
if [ "$CLEAN_BUILD" = true ]; then
    log_info "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"

log_info "Build configuration:"
log_info "  Project root: $PROJECT_ROOT"
log_info "  HeCBench dir: $HECBENCH_DIR"
log_info "  Build dir:    $BUILD_DIR"
log_info "  CUDA arch:    $CUDA_ARCH"

# Configure CMake with LLVM toolchain
log_info "Configuring CMake..."

cd "$BUILD_DIR"



# Use clang/clang++ for C/C++/OpenMP and let CMake find nvcc for CUDA
cmake "$HECBENCH_DIR" \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DHECBENCH_ENABLE_CUDA=ON \
    -DHECBENCH_ENABLE_OPENMP=ON \
    -DHECBENCH_ENABLE_HIP=OFF \
    -DHECBENCH_ENABLE_SYCL=OFF \
    -DHECBENCH_CUDA_ARCH="$CUDA_ARCH" \
    -DHECBENCH_BUILD_ALL_BENCHMARKS=ON \
    -DHECBENCH_ENABLE_TESTING=OFF \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_VERBOSE_MAKEFILE=ON \
    -DHECBENCH_OMP_TARGET=-fopenmp \
    -DHECBENCH_OMP_TARGET_BACKEND=--offload-arch=sm_$CUDA_ARCH \
    -DNCCL_DIR=/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/comm_libs/13.0/nccl \
    -DCMAKE_PREFIX_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/comm_libs/13.0/nccl 

if [ $? -ne 0 ]; then
    log_error "CMake configuration failed"
    exit 1
fi

log_info "Configuration complete"

# Build all benchmarks
log_info "Building benchmarks (this may take a while)..."

# we use the -k flag to keep going on errors and log all output
# this way we at least build what we can and not stop on the codes with 
# bad builds

cmake --build . -j 4 -- VERBOSE=1 -k 2>&1 | tee build.log

# Note: We intentionally do NOT check PIPESTATUS here because -k flag
# allows the build to continue on errors. We'll validate success by
# counting the number of successfully built executables instead.

log_info "Build complete!"

# Thresholds for successful build
MIN_CUDA_EXECUTABLES=450
MIN_OMP_EXECUTABLES=300

# Number of sample executables to display
SAMPLE_COUNT=5

# Count built executables by type
CUDA_BIN_DIR="$BUILD_DIR/bin/cuda"
OMP_BIN_DIR="$BUILD_DIR/bin/omp"

if [ -d "$CUDA_BIN_DIR" ]; then
    CUDA_COUNT=$(find "$CUDA_BIN_DIR" -maxdepth 1 -type f 2>/dev/null | wc -l)
else
    CUDA_COUNT=0
fi

if [ -d "$OMP_BIN_DIR" ]; then
    OMP_COUNT=$(find "$OMP_BIN_DIR" -maxdepth 1 -type f 2>/dev/null | wc -l)
else
    OMP_COUNT=0
fi

# Print diagnostics
log_info "===== Build Results ====="
log_info "CUDA executables: $CUDA_COUNT (minimum required: $MIN_CUDA_EXECUTABLES)"
log_info "OpenMP executables: $OMP_COUNT (minimum required: $MIN_OMP_EXECUTABLES)"
log_info "========================="

# Check if thresholds are met
BUILD_SUCCESS=true

if [ $CUDA_COUNT -lt $MIN_CUDA_EXECUTABLES ]; then
    log_error "Insufficient CUDA executables: $CUDA_COUNT < $MIN_CUDA_EXECUTABLES"
    BUILD_SUCCESS=false
fi

if [ $OMP_COUNT -lt $MIN_OMP_EXECUTABLES ]; then
    log_error "Insufficient OpenMP executables: $OMP_COUNT < $MIN_OMP_EXECUTABLES"
    BUILD_SUCCESS=false
fi

if [ "$BUILD_SUCCESS" = false ]; then
    log_error "Build failed to meet executable count thresholds"
    log_error "Check build.log for compilation errors"
    exit 1
fi

# Show sample executables
log_info "Build thresholds met successfully!"

if [ $CUDA_COUNT -gt 0 ]; then
    log_info "Sample CUDA executables:"
    find "$CUDA_BIN_DIR" -maxdepth 1 -type f 2>/dev/null | head -n $SAMPLE_COUNT | while IFS= read -r exe; do
        log_info "  - $(basename "$exe")"
    done
    if [ $CUDA_COUNT -gt $SAMPLE_COUNT ]; then
        log_info "  ... and $((CUDA_COUNT - SAMPLE_COUNT)) more"
    fi
fi

if [ $OMP_COUNT -gt 0 ]; then
    log_info "Sample OpenMP executables:"
    find "$OMP_BIN_DIR" -maxdepth 1 -type f 2>/dev/null | head -n $SAMPLE_COUNT | while IFS= read -r exe; do
        log_info "  - $(basename "$exe")"
    done
    if [ $OMP_COUNT -gt $SAMPLE_COUNT ]; then
        log_info "  ... and $((OMP_COUNT - SAMPLE_COUNT)) more"
    fi
fi

log_info "Build script completed successfully"
log_info "Executables are in: $BUILD_DIR/bin/cuda and $BUILD_DIR/bin/omp"
