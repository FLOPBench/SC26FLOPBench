# Dockerfile for gpuFLOPBench-updated
#
# This container provides a complete environment for building HeCBench benchmarks
# and profiling them with NVIDIA Nsight Compute (ncu).
#
# Base: NVIDIA CUDA development image with Ubuntu
# Includes: CUDA toolkit, Nsight Compute, LLVM/Clang, CMake, Python dependencies
#
# Build:
#   docker build -t gpuflopbench-updated .
#
# Run (requires NVIDIA GPU and nvidia-docker):
#   docker run --gpus all --cap-add=SYS_ADMIN --cap-add=SYS_PTRACE \
#       -v $(pwd):/workspace gpuflopbench-updated bash
#
# Note: SYS_ADMIN and SYS_PTRACE capabilities are required for ncu profiling

FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

LABEL maintainer="gpuFLOPBench-updated"
LABEL description="Build and profiling environment for HeCBench benchmarks"

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    cmake \
    git \
    wget \
    curl \
    # LLVM/Clang toolchain
    clang-15 \
    llvm-15 \
    libomp-15-dev \
    # Additional tools
    python3 \
    python3-pip \
    binutils \
    # Utilities
    vim \
    less \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Set clang as default compiler
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 100 && \
    update-alternatives --install /usr/bin/llvm-objdump llvm-objdump /usr/bin/llvm-objdump-15 100 && \
    update-alternatives --install /usr/bin/llvm-cxxfilt llvm-cxxfilt /usr/bin/llvm-cxxfilt-15 100

# Install Python dependencies for profiling script
RUN pip3 install --no-cache-dir \
    pandas \
    numpy \
    pyyaml \
    tqdm

# Verify CUDA and NCU are available
RUN nvcc --version && \
    ncu --version || echo "NCU not available in this image - may need nsight-compute package"

# Set working directory
WORKDIR /workspace

# Copy repository contents
COPY . /workspace/

# Initialize HeCBench submodule if not already done
RUN if [ -d "HeCBench" ] && [ ! "$(ls -A HeCBench)" ]; then \
        git submodule update --init --depth 1 HeCBench; \
    fi

# Set executable permissions
RUN chmod +x runBuild.sh

# Environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Default command
CMD ["/bin/bash"]

# Usage instructions
RUN echo "===================================================" && \
    echo "gpuFLOPBench-updated Docker Container" && \
    echo "===================================================" && \
    echo "To build HeCBench benchmarks:" && \
    echo "  ./runBuild.sh" && \
    echo "" && \
    echo "To profile benchmarks (requires GPU):" && \
    echo "  python3 cuda-profiling/gatherData.py" && \
    echo "" && \
    echo "Output locations:" && \
    echo "  - Executables: build/" && \
    echo "  - Profiling data: cuda-profiling/gpuData.csv" && \
    echo "===================================================" && \
    echo ""
