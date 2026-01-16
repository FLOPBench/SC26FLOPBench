# gpuFLOPBench-updated: Build and Profiling Infrastructure

## Overview

`gpuFLOPBench-updated` provides infrastructure for building and profiling GPU benchmarks from the HeCBench suite. It extends the approach from [Scientific-Computing-Lab/gpuFLOPBench](https://github.com/Scientific-Computing-Lab/gpuFLOPBench) to work with HeCBench using LLVM clang/clang++ toolchain.

## Project Structure

```
gpuFLOPBench-updated/
├── HeCBench/                  # Git submodule containing benchmark suite
│   ├── benchmarks.yaml        # Benchmark metadata and execution arguments
│   ├── CMakeLists.txt         # Top-level build configuration
│   └── src/                   # Individual benchmark source directories
│       ├── accuracy-cuda/
│       ├── accuracy-omp/
│       └── ...
├── runBuild.sh                # Build script for all benchmarks
├── cuda-profiling/
│   ├── gatherData.py          # Profiling script using ncu
│   ├── gpuData.csv            # Output: profiling results (generated)
│   └── downloads/             # Downloaded input files (generated)
├── build/                     # Built executables (generated)
├── unit-tests/                # Test suite
│   ├── AGENTS.md              # Test documentation
│   └── test_*.py              # Test files
├── Dockerfile                 # Container for building and profiling
└── AGENTS.md                  # This file

```

## HeCBench Submodule

HeCBench is a comprehensive heterogeneous computing benchmark suite from [zjin-lcf/HeCBench](https://github.com/zjin-lcf/HeCBench) containing implementations of benchmarks in multiple programming models:
- **CUDA**: Native NVIDIA GPU programming
- **OpenMP**: OpenMP target offloading
- **HIP**: AMD GPU programming (not built by default)
- **SYCL**: Cross-platform heterogeneous programming (not built by default)

### Benchmark Organization

Each benchmark has separate source directories for each model:
```
HeCBench/src/
├── accuracy-cuda/       # CUDA implementation
├── accuracy-omp/        # OpenMP implementation
├── accuracy-sycl/       # SYCL implementation
└── ...
```

### benchmarks.yaml

The `benchmarks.yaml` file in the HeCBench root contains metadata for each benchmark:
```yaml
accuracy:
  categories: [algorithms]
  models: [cuda, hip, omp, sycl]
  test:
    regex: '(?:Average execution time...)...'
    args: ["8192", "10000", "10", "100"]
    timeout: 300
```

This metadata is used by `gatherData.py` to:
- Determine which benchmarks are available
- Extract execution arguments for profiling
- Understand expected output patterns

## Building Benchmarks

### Prerequisites

- **LLVM/Clang**: clang and clang++ (version 14+)
- **CMake**: 3.21+
- **CUDA Toolkit**: For CUDA support (cuobjdump, cuda headers)
- **OpenMP**: Typically included with clang

### Build Script: `runBuild.sh`

The build script configures and builds all CUDA and OpenMP benchmarks using the LLVM toolchain.

#### Usage

```bash
# Standard build
./runBuild.sh

# Clean build (remove existing build directory)
./runBuild.sh --clean

# Specify CUDA architecture
./runBuild.sh --cuda-arch sm_86
```

#### Build Process

1. **Submodule Check**: Verifies HeCBench submodule is initialized
2. **Tool Verification**: Checks for clang, clang++, cmake
3. **CMake Configuration**: 
   - Sets `CMAKE_C_COMPILER=clang`
   - Sets `CMAKE_CXX_COMPILER=clang++`
   - Sets `CMAKE_CUDA_COMPILER=clang++` (uses clang for CUDA!)
   - Enables CUDA and OpenMP models
   - Disables HIP and SYCL models
4. **Parallel Build**: Builds using all available cores
5. **Output**: Executables placed in `build/` directory

#### Build Outputs

All executables are placed in the `build/` directory with names matching their source directory:

```
build/
├── accuracy-cuda          # From HeCBench/src/accuracy-cuda/
├── accuracy-omp           # From HeCBench/src/accuracy-omp/
├── lulesh-cuda            # From HeCBench/src/lulesh-cuda/
└── ...
```

#### Troubleshooting

- **Missing submodule**: Run `git submodule update --init`
- **Clang not found**: Install LLVM/Clang 14+
- **CUDA compilation fails**: Ensure CUDA toolkit is installed
- **Some benchmarks fail**: Check `build/build.log` for specific errors

## Profiling Benchmarks

### Prerequisites

- **NVIDIA GPU**: For executing benchmarks
- **Nsight Compute (ncu)**: NVIDIA's profiling tool
- **Python 3.11+**: With packages: pandas, numpy, pyyaml, tqdm (installed via Miniconda in Docker)
- **Built benchmarks**: Run `./runBuild.sh` first

### Profiling Script: `cuda-profiling/gatherData.py`

The profiling script executes benchmarks with NVIDIA Nsight Compute to gather roofline performance data.

#### Usage

```bash
cd cuda-profiling

# Standard profiling
python3 gatherData.py

# Custom paths
python3 gatherData.py --buildDir ../build --srcDir ../HeCBench/src

# Custom output file
python3 gatherData.py --outfile ./my_results.csv

# Skip execution (only parse existing ncu-rep files)
python3 gatherData.py --skipRuns
```

#### Profiling Workflow

1. **Parse benchmarks.yaml**: Load metadata and execution arguments
2. **Discover executables**: Find all binaries in `build/`
3. **Extract kernel names**: 
   - For CUDA: Use `cuobjdump --list-text` or `llvm-objdump -t`
   - For OpenMP: Use `objdump -t --section=omp_offloading_entries`
4. **Demangle kernel names**:
   - Try `cu++filt`, `c++filt`, `llvm-cxxfilt` in order
   - Extract simple name for `-k` option (handle templates, namespaces)
5. **Execute with ncu**:
   ```bash
   ncu -f -o report --set roofline \
       --metrics smsp__sass_thread_inst_executed_op_integer_pred_on \
       -c 2 -k "kernel_name" ./executable args
   ```
6. **Parse ncu-rep files**: Extract performance counters to CSV
7. **Calculate metrics**: Compute roofline data (AI, performance, traffic)
8. **Save results**: Write to `gpuData.csv`

#### Kernel Name Extraction and Demangling

**Problem**: CUDA kernel names in binaries are mangled. The `ncu -k` option needs correct kernel names to profile specific kernels.

**Solution**: Three-step process:

1. **Extract mangled names**:
   ```bash
   # CUDA binaries
   cuobjdump --list-text binary | grep "\.text\."
   
   # OpenMP binaries
   objdump -t --section=omp_offloading_entries binary
   ```

2. **Demangle names**:
   ```bash
   # Try in order of preference
   echo "mangled_name" | cu++filt
   echo "mangled_name" | c++filt
   echo "mangled_name" | llvm-cxxfilt
   ```

3. **Extract simple name** for `ncu -k`:
   - Remove return type: `void kernel<int>` → `kernel<int>`
   - Remove templates: `kernel<int>` → `kernel`
   - Remove namespaces: `my::ns::kernel` → `kernel`
   - Filter library kernels: Skip `cub::` and `thrust::`

**Bug Fix**: The upstream gpuFLOPBench script had convoluted fallback logic for demangling. Our implementation:
- Tries demangling tools in correct order (`cu++filt` first for CUDA)
- Returns first successful demangle result
- Avoids unnecessary retries with broken logic

#### Roofline Metrics

The script calculates standard roofline metrics:

**Double Precision:**
- Performance (FLOP/s): `(DP_ADD + DP_MUL + DP_FMA*2) * cycles_per_sec`
- Arithmetic Intensity: `DP_Performance / DRAM_bytes_per_sec`

**Single Precision:**
- Performance (FLOP/s): `(SP_ADD + SP_MUL + SP_FMA*2) * cycles_per_sec`
- Arithmetic Intensity: `SP_Performance / DRAM_bytes_per_sec`

**Integer:**
- Performance (OP/s): `int_ops / execution_time`
- Arithmetic Intensity: `INT_Performance / DRAM_bytes_per_sec`

**Memory Traffic:**
- DRAM bytes/sec

#### Output: gpuData.csv

The output CSV contains one row per kernel invocation:

| Column | Description |
|--------|-------------|
| targetName | Executable basename (e.g., "accuracy-cuda") |
| kernelName | Kernel name used for profiling |
| exeArgs | Command-line arguments used |
| Kernel Name | Full kernel name from ncu |
| traffic | DRAM traffic (bytes/sec) |
| dpPerf | Double-precision performance (FLOP/s) |
| spPerf | Single-precision performance (FLOP/s) |
| intPerf | Integer performance (OP/s) |
| dpAI | Double-precision arithmetic intensity |
| spAI | Single-precision arithmetic intensity |
| intAI | Integer arithmetic intensity |
| xtime | Kernel execution time (ns) |
| Block Size | CUDA block dimensions |
| Grid Size | CUDA grid dimensions |
| device | GPU device name |
| intops | Total integer operations |

#### Input Data Handling

Some benchmarks require input files. The script:
- Automatically extracts `.tar.gz`, `.tgz`, `.zip` archives in source directories
- Downloads remote files when specified in benchmark metadata
- Reuses downloaded files in `cuda-profiling/downloads/`

Example benchmarks needing data:
- `rodinia-*`: Uses Rodinia dataset (auto-downloaded)
- `mriQ-cuda`: Downloads MRI-Q dataset
- `word2vec-cuda`: Downloads text8 corpus

## Docker Usage

### Building the Container

```bash
docker build -t gpuflopbench-updated .
```

### Running the Container

**Requirements**:
- NVIDIA Docker runtime (`nvidia-docker` or `docker --gpus`)
- GPU access for profiling

**Build only** (no GPU needed):
```bash
docker run --rm -v $(pwd):/workspace gpuflopbench-updated ./runBuild.sh
```

**Build and profile** (GPU required):
```bash
docker run --gpus all \
    --cap-add=SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    -v $(pwd):/workspace \
    gpuflopbench-updated bash
    
# Inside container:
./runBuild.sh
python3 cuda-profiling/gatherData.py
```

**Capabilities explained**:
- `--gpus all`: Provides GPU access
- `--cap-add=SYS_ADMIN`: Required for ncu GPU profiling
- `--cap-add=SYS_PTRACE`: Required for process tracing

### Container Contents

- **Base**: `nvidia/cuda:12.2.0-devel-ubuntu22.04`
- **Compilers**: clang-20, clang++-20 (LLVM 20)
- **Tools**: cmake, git, cuobjdump, ncu, objdump
- **Python**: Python 3.11 via Miniconda with pandas, numpy, pyyaml, tqdm

## Output Locations

| Output | Location | Description |
|--------|----------|-------------|
| Executables | `build/` | Compiled benchmarks |
| Build logs | `build/build.log` | CMake and compiler output |
| Profiling data | `cuda-profiling/gpuData.csv` | Performance metrics |
| NCU reports | `HeCBench/src/*/*.ncu-rep` | Raw ncu reports (large!) |
| Downloaded files | `cuda-profiling/downloads/` | Input datasets |

## Workflow Example

### Complete Build and Profile Workflow

```bash
# 1. Clone repository with submodule
git clone --recurse-submodules https://github.com/gregbolet/gpuFLOPBench-updated.git
cd gpuFLOPBench-updated

# 2. Build benchmarks
./runBuild.sh

# 3. Run profiling (requires GPU)
cd cuda-profiling
python3 gatherData.py

# 4. Analyze results
# gpuData.csv now contains performance data for all kernels
```

### Docker Workflow

```bash
# 1. Build Docker image
docker build -t gpuflopbench-updated .

# 2. Run container with GPU
docker run --gpus all \
    --cap-add=SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    -v $(pwd)/results:/workspace/cuda-profiling \
    gpuflopbench-updated bash -c \
    "./runBuild.sh && python3 cuda-profiling/gatherData.py"

# 3. Results available in ./results/gpuData.csv
```

## Testing

Unit tests verify the correctness of the build and profiling infrastructure. See [`unit-tests/AGENTS.md`](unit-tests/AGENTS.md) for details.

```bash
# Run all tests
cd unit-tests
pytest -v

# Run specific test
pytest test_build_artifacts.py -v
```

## Troubleshooting

### Build Issues

**CMake can't find CUDA**:
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
```

**Clang not found**:
```bash
# Install LLVM 20
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
./llvm.sh 20 all
```

### Profiling Issues

**ncu not found**:
```bash
# Install NVIDIA Nsight Compute
# Usually included in CUDA Toolkit or available separately
export PATH=/usr/local/cuda/bin:$PATH
```

**Permission denied for ncu**:
```bash
# Add user to video group or run with appropriate permissions
sudo usermod -a -G video $USER
# Or use docker with --cap-add=SYS_ADMIN
```

**No kernels profiled**:
- Check kernel name extraction: `cuobjdump --list-text build/benchmark`
- Verify benchmark executes correctly: `build/benchmark args`
- Check ncu-rep file was created

**Out of memory during profiling**:
- Reduce number of benchmarks profiled
- Increase timeout in script
- Profile one benchmark at a time

## References

- **HeCBench**: https://github.com/zjin-lcf/HeCBench
- **gpuFLOPBench**: https://github.com/Scientific-Computing-Lab/gpuFLOPBench
- **Nsight Compute**: https://developer.nvidia.com/nsight-compute
- **LLVM**: https://llvm.org/

## Contributing

When modifying this infrastructure:

1. **runBuild.sh**: Keep LLVM toolchain configuration, ensure all models build
2. **gatherData.py**: Maintain demangling correctness, handle new input file patterns
3. **Dockerfile**: Keep base image updated, ensure all tools available
4. **Tests**: Add tests for new functionality
5. **Documentation**: Update this file for any workflow changes

## License

See repository LICENSE file for details.
