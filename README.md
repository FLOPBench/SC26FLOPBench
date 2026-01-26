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

## Using a Lambda AI instance
Here are some commands for getting set up in a lambda.ai GPU cloud instance.
```
sudo apt-get update && sudo apt-get -y full-upgrade 

echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" | sudo tee /etc/modprobe.d/nvidia-elevate-privs.conf > /dev/null

sudo reboot

# 1) Remove the kernel that triggers the DKMS failures
sudo apt-get purge -y \
  linux-image-6.14.0-1015-nvidia \
  linux-headers-6.14.0-1015-nvidia \
  linux-nvidia-hwe-24.04 \
  linux-headers-nvidia-hwe-24.04

# 2) Finish configuring what’s already installed / fix broken state
sudo dpkg --configure -a
sudo apt-get -f install

# 3) Force DKMS to build/install for the CURRENT running kernel only
sudo apt-get install -y linux-headers-$(uname -r) build-essential dkms
sudo dkms autoinstall -k "$(uname -r)"

# 4) Re-run the driver install to ensure everything is configured
sudo apt-get install -y nvidia-driver-580-open

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda-repo-ubuntu2404-13-0-local_13.0.0-580.65.06-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-13-0-local_13.0.0-580.65.06-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-0

sudo apt-get remove --purge -y nvidia-cuda-toolkit

sudo reboot

sudo update-alternatives --install /usr/bin/nvcc nvcc /usr/local/cuda-13.0/bin/nvcc 1300
sudo update-alternatives --set nvcc /usr/local/cuda-13.0/bin/nvcc

curl https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg
echo 'deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' | sudo tee /etc/apt/sources.list.d/nvhpc.list
sudo apt-get update -y
sudo apt-get install -y nvhpc-25-11


wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 21 all
rm llvm.sh
sudo apt-get clean
sudo apt-get install liboffload-21-dev libomp-21-dev

sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-21 100
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-21 100
sudo update-alternatives --install /usr/bin/llvm-objdump llvm-objdump /usr/bin/llvm-objdump-21 100
sudo update-alternatives --install /usr/bin/llvm-cxxfilt llvm-cxxfilt /usr/bin/llvm-cxxfilt-21 100

sudo apt autoremove

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -p ~/anaconda3
rm ./Miniconda3-latest-Linux-x86_64.sh
source ~/anaconda3/bin/activate
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main &&     conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create --name gpuflopbench-updated python=3.11 -y
conda activate gpuflopbench-updated 

ssh-keygen -t ed25519 -C "gbolet@vt.edu"
cat ~/.ssh/id_ed25519.pub
pip install -r requirements.txt

sudo apt-get install -y g++ gcc libstdc++-14-dev libboost-all-dev

export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/comm_libs/nvshmem/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/comm_libs/nccl/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/math_libs/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/extras/qd/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/cuda/lib64:/usr/local/cuda/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/comm_libs/nvshmem/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/comm_libs/nccl/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/math_libs/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/extras/qd/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/cuda/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/cuda/13.0/bin:/usr/lib/llvm-21/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/extras/qd/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/comm_libs/mpi/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/cuda/bin:/usr/local/cuda/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/comm_libs/nvshmem/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/comm_libs/nccl/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/profilers/bin:/home/gbolet/.vscode-server/data/User/globalStorage/github.copilot-chat/debugCommand:/home/gbolet/.vscode-server/data/User/globalStorage/github.copilot-chat/copilotCli:/home/gbolet/.vscode-server/cli/servers/Stable-94e8ae2b28cb5cc932b86e1070569c4463565c37/server/bin/remote-cli:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/cuda/13.0/bin:/usr/lib/llvm-21/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/extras/qd/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/comm_libs/mpi/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/cuda/bin:/usr/local/cuda/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/comm_libs/nvshmem/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/comm_libs/nccl/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/profilers/bin:/home/gbolet/anaconda3/envs/gpuflopbench-updated/bin:/home/gbolet/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/home/gbolet/.vscode-server/extensions/ms-python.debugpy-2025.18.0-linux-x64/bundled/scripts/noConfigScripts:$PATH

export CUDA_HOME=/usr/local/cuda-13.0

# need to add these lines to you cmake invocation in ./runBuild.sh

    -DCUDAToolkit_ROOT=$CUDA_HOME \
    -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \

./runBuild.sh
```

## GPU Clean Troubleshooting
We ideally want an `nvidia-smi` command output to show that `0MiB` are being used on the GPU memory.
We found that on our test system, we were reading an idle mem use of `1MiB`.
In order to fix this, we use the following commands:

```
sudo modprobe -r nvidia_drm
sudo nvidia-smi -pm 1
sudo modprobe -r nvidia_drm
nvidia-smi
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
