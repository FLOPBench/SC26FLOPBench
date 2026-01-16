# gpuFLOPBench-updated

Build and profiling infrastructure for [HeCBench](https://github.com/zjin-lcf/HeCBench) GPU benchmarks using LLVM clang/clang++ and NVIDIA Nsight Compute.

## Quick Start

### Prerequisites

- **LLVM/Clang** 14+ (clang, clang++)
- **CMake** 3.21+
- **CUDA Toolkit** (for CUDA support)
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

Executables will be placed in `build/`.

### Profile Benchmarks

```bash
cd cuda-profiling
python3 gatherData.py
```

Results will be saved to `cuda-profiling/gpuData.csv`.

## Docker Usage

### Build Container

```bash
docker build -t gpuflopbench-updated .
```

### Run with GPU

```bash
docker run --gpus all \
    --cap-add=SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    -v $(pwd):/workspace \
    gpuflopbench-updated bash
```

Inside container:
```bash
./runBuild.sh
python3 cuda-profiling/gatherData.py
```

## Documentation

- **[AGENTS.md](AGENTS.md)**: Comprehensive documentation of the infrastructure
- **[unit-tests/AGENTS.md](unit-tests/AGENTS.md)**: Testing documentation

## Testing

```bash
cd unit-tests
pip install -r requirements.txt
pytest -v
```

## Project Structure

```
├── HeCBench/              # Submodule: benchmark suite
├── runBuild.sh            # Build script
├── cuda-profiling/
│   ├── gatherData.py      # Profiling script
│   └── gpuData.csv        # Output (generated)
├── build/                 # Executables (generated)
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