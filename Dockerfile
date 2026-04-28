# Dockerfile for gpuFLOPBench-updated
#
# This container provides a complete environment for building HeCBench benchmarks
# and profiling them with NVIDIA Nsight Compute (ncu).
#
# Base: NVIDIA HPC SDK development image with CUDA 13.0
# Includes: CUDA 13.0 toolkit, NVHPC compilers, CMake, Python via Miniconda
#
# Build:
#   docker build -t gpuflopbench-updated .
#
# Run (requires NVIDIA GPU and nvidia-docker):
#   docker run --gpus all --cap-add=SYS_ADMIN --cap-add=SYS_PTRACE \
#       -v $(pwd):/workspace gpuflopbench-updated bash
#
# Note: SYS_ADMIN and SYS_PTRACE capabilities are required for ncu profiling

FROM nvcr.io/nvidia/nvhpc:25.11-devel-cuda13.0-ubuntu24.04

LABEL maintainer="gpuFLOPBench-updated"
LABEL description="Build and profiling environment for HeCBench benchmarks"

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Change default shell to bash
SHELL ["/bin/bash", "-c"]

# Install system dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    # Build essentials
    build-essential \
    g++ gcc libstdc++-14-dev libboost-all-dev libgsl-dev \
    cmake \
    git git-lfs \
    wget \
    curl \
    unzip \
    # LLVM/Clang 20 toolchain
    lsb-release \
    software-properties-common \
    gnupg \
    # Additional tools
    binutils \
    # MPI for distributed benchmarks
    libopenmpi-dev \
    openmpi-bin \
    # Optional libraries for some benchmarks
    libboost-all-dev \
    libgsl-dev \
    # Utilities
    vim \
    less \
    htop \
    postgresql \
    && rm -rf /var/lib/apt/lists/*

# Install LLVM/Clang 21
RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 21 all && \
    rm llvm.sh && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# for offloading support, we need to install these packages
# they fix the following OpenMP compilation error
# clang++: error: no library 'libomptarget-nvptx.bc' found in the default clang lib directory
RUN apt-get update && \
    apt-get install liboffload-21-dev libomp-21-dev

# Set clang-21 as default compiler
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-21 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-21 100 && \
    update-alternatives --install /usr/bin/llvm-objdump llvm-objdump /usr/bin/llvm-objdump-21 100 && \
    update-alternatives --install /usr/bin/llvm-cxxfilt llvm-cxxfilt /usr/bin/llvm-cxxfilt-21 100
# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash ./Miniconda3-latest-Linux-x86_64.sh -b -p ~/anaconda3 && \
    rm ./Miniconda3-latest-Linux-x86_64.sh

# Initialize conda and accept terms of service
RUN source ~/anaconda3/bin/activate && \
    conda init --all && \
    conda config --set channel_priority strict

# Accept Anaconda Terms of Service for non-interactive builds
# This is required for CI/CD pipelines and automated Docker builds
# where user interaction is not possible. Without this, conda will
# fail with CondaToSNonInteractiveError when trying to use default channels.
RUN source ~/anaconda3/bin/activate && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create conda environment for gpuFLOPBench-updated
RUN source ~/anaconda3/bin/activate && \
    conda create --name gpuflopbench-updated python=3.11 -y

# Copy requirements files first so pip install is cached independently of other source changes
COPY requirements.txt /gpuFLOPBench-updated/requirements.txt

# Install Python dependencies in conda environment from requirements files
RUN source ~/anaconda3/bin/activate && \
    conda activate gpuflopbench-updated && \
    pip install --no-cache-dir -r /gpuFLOPBench-updated/requirements.txt

# Set working directory
WORKDIR /gpuFLOPBench-updated

# Copy repository contents
COPY . /gpuFLOPBench-updated/

# Set executable permissions
RUN chmod +x runBuild.sh runTests.sh

# Environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Activate conda environment on container startup
RUN echo 'source ~/anaconda3/bin/activate' >> ~/.bashrc && \
    echo 'conda activate gpuflopbench-updated' >> ~/.bashrc

# Set environment variable for convenience
ENV GPUFLOPBENCH_ROOT=/gpuFLOPBench-updated

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
    echo "  python cuda-profiling/gatherData.py" && \
    echo "" && \
    echo "Output locations:" && \
    echo "  - Executables: build/" && \
    echo "  - Profiling data: cuda-profiling/gpuData.csv" && \
    echo "===================================================" && \
    echo ""
