# gpuFLOPBench-updated

Build and profiling infrastructure for [HeCBench](https://github.com/zjin-lcf/HeCBench) GPU benchmarks using LLVM clang/clang++ and NVIDIA Nsight Compute.

## Quick Start

### Prerequisites

- **LLVM/Clang** 21+ (clang, clang++)
- **CMake** 3.21+
- **CUDA Toolkit** v13.0 (for CUDA support)
- **Python 3.8+** with packages: pandas, numpy, pyyaml, tqdm
- **NVIDIA GPU** (for profiling)
- **Nsight Compute (ncu)** (for profiling)

### Clone with Submodule

```bash
git clone --recurse-submodules https://github.com/gregbolet/gpuFLOPBench-updated.git
cd gpuFLOPBench-updated
```

Or if already cloned:
```bash
git submodule update --init
```

### Build Benchmarks

```bash
./runBuild.sh
```

Executables will be placed in `build/bin/cuda/` (CUDA implementations) and `build/bin/omp/` (OpenMP implementations).

The build succeeds if it produces at least 450 CUDA executables and 300 OpenMP executables.

### Run Tests

```bash
./runTests.sh           # Run all tests
./runTests.sh --noGPU   # Run tests excluding GPU-dependent tests
```

### Profile Benchmarks

```bash
cd cuda-profiling
python3 gatherData.py
```

Results will be saved to `cuda-profiling/gpuData.csv`.

## Docker Usage

⚠️ **Storage Requirements**: The Docker container requires approximately 15 GB for the base image, expanding to 40 GB when built, and up to 50 GB when building codes and gathering profiling data. Ensure sufficient disk space before proceeding.

### Build Container

```bash
docker build -t gpuflopbench-updated .
```

This takes approximately 5-15 minutes depending on your system.

### Platform-Specific Setup Instructions

#### Linux with NVIDIA GPU

For Ubuntu systems with NVIDIA GPUs and nvidia-docker runtime:

```bash
# update the modprobe to elevate privileges
echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" | sudo tee /etc/modprobe.d/nvidia-elevate-privs.conf > /dev/null

# Build the container
docker build --progress=plain -t gpuflopbench-updated .

# Run with GPU access (ensure Docker Desktop has 'Enable Host Networking' enabled)
docker run -ti --network=host --gpus all --name gpuflopbench-updated-container -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all gpuflopbench-updated

# Access the container shell
docker exec -it gpuflopbench-updated-container /bin/bash
```

**Capabilities explained**:
- `--gpus all`: Provides GPU access
- `--network=host`: Enables host networking (useful for Jupyter notebooks)

#### macOS (Apple Silicon M1/M2/M3/M4) - No NVIDIA GPU

For macOS systems without NVIDIA GPU (useful for dataset analysis and LLM querying):

```bash
# Build for x86_64 architecture (takes ~10 minutes on Apple Silicon)
docker build --platform=linux/amd64 --progress=plain -t gpuflopbench-updated .

# Run container (ensure Docker Desktop has 'Enable Host Networking' enabled)
docker run -ti --network=host \
    --name gpuflopbench-updated-container \
    --platform=linux/amd64 \
    -v $(pwd):/workspace \
    gpuflopbench-updated

# Access the container shell
docker exec -it gpuflopbench-updated-container /bin/bash
```

**Note**: Without GPU, you can still access the codebase, run unit tests (excluding GPU-dependent tests), and work with pre-collected profiling data.

#### Windows with NVIDIA GPU

For Windows systems with Docker Desktop and NVIDIA GPU:

**Prerequisites**: Enable GPU performance counters in NVIDIA Control Panel:
1. Open **NVIDIA Control Panel**
2. Navigate to **Desktop** tab → Enable **Developer Settings**
3. Navigate to **Select a Task...** → **Developer** → **Manage GPU Performance Counters**
4. Select **Allow access to the GPU performance counters to all users**
5. Restart Docker Desktop

**Run container**:
```powershell
# Build the container
docker build --progress=plain -t gpuflopbench-updated .

# Run with GPU access
docker run -ti --network=host --gpus all --name gpuflopbench-updated-container -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all gpuflopbench-updated

# Access the container shell
docker exec -it gpuflopbench-updated-container /bin/bash
```

### Container Management

Start and stop the container as needed:
```bash
# Start container
docker start gpuflopbench-updated-container

# Stop container
docker stop gpuflopbench-updated-container

# Remove container (preserves image)
docker rm gpuflopbench-updated-container
```

File changes in the container persist unless you delete the container.

### Inside the Container

Once inside the container shell:

```bash
# Activate conda environment (should auto-activate)
conda activate gpuflopbench-updated

# Build HeCBench benchmarks
./runBuild.sh

# Profile benchmarks (requires GPU)
cd cuda-profiling
python gatherData.py
```

## Build Notes
Of the version of HeCBench that we use as a submodule, we manage to build most of the codes.
- 488 CUDA codes building
- 318 OMP codes building

There are some errors that we run into during build for some codes, but we ignore these programs as we didn't want to spend the extra effort to get their builds to work.

## Running Jupyter Notebooks
We have a couple Jupyter Notebooks to visualize collected data so we can manually inspect it's correctness.
To start the server, use the following command:

```
jupyter notebook --allow-root --no-browser --ip=0.0.0.0 --port=8888 --NotebookApp.token=''
```

## Documentation

- **[AGENTS.md](AGENTS.md)**: Comprehensive documentation of the infrastructure
- **[unit-tests/AGENTS.md](unit-tests/AGENTS.md)**: Testing documentation

## Testing

```bash
./runTests.sh           # Run all tests
./runTests.sh --noGPU   # Run tests excluding GPU-dependent tests
```

Or manually with pytest:
```bash
cd unit-tests
pip install -r requirements.txt
pytest -v
```

## Project Structure

```
├── HeCBench/              # Submodule: benchmark suite
├── runBuild.sh            # Build script
├── runTests.sh            # Test script
├── cuda-profiling/
│   ├── gatherData.py      # Profiling script
│   └── gpuData.csv        # Output (generated)
├── build/                 # Build artifacts (generated)
│   └── bin/
│       ├── cuda/          # CUDA executables (450+ expected)
│       └── omp/           # OpenMP executables (300+ expected)
├── unit-tests/            # Test suite
├── Dockerfile             # Container definition
└── AGENTS.md              # Full documentation
```

## Key Features

- **LLVM Toolchain**: Uses clang/clang++ for all compilation (including CUDA)
- **Roofline Profiling**: Gathers FLOP/s, arithmetic intensity, memory traffic
- **Kernel Discovery**: Automatic extraction and demangling of kernel names
- **Input Handling**: Auto-downloads required benchmark input files
- **Comprehensive Tests**: Validates build artifacts, kernel extraction, demangling

## References

- **HeCBench**: https://github.com/zjin-lcf/HeCBench
- **Original gpuFLOPBench**: https://github.com/Scientific-Computing-Lab/gpuFLOPBench
- **Nsight Compute**: https://developer.nvidia.com/nsight-compute

## License

See [LICENSE](LICENSE) file.
