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

Results are saved to gpuData.csv (prefixed with the GPU name when available) alongside:
- profiling-log-*.json (per-run stdout/stderr and status)
- profiling-results-*.zip (CSV, log, build log, compile_commands.json, and NCU reports)
- ncu-rep-results/*.ncu-rep (raw Nsight Compute reports)

Common options:
- --cudaOnly / --ompOnly: limit to CUDA or OpenMP targets
- --samples N: repeat profiling per target (default: 3)
- --timeout SEC: per-run timeout (default: 120)
- --rerunTimeouts: re-run targets that previously timed out
- --skipConfirm: skip the interactive confirmation prompt
- --zipOnly: skip profiling and only create the results archive

---

## End-to-End Workflow

The complete pipeline runs in five phases: building and profiling HeCBench benchmarks, constructing the structured dataset, running LLM experiments, generating paper figures and listings, and validating results with the test suite.

### Prerequisites

**Build & Profile:**
- LLVM/Clang 21+, CMake 3.21+, CUDA Toolkit v13.0, NVIDIA GPU, Nsight Compute (`ncu`)

**LLM Experiments:**
- PostgreSQL (scripts auto-start the local cluster via `pg_ctlcluster` when needed)
- `OPENROUTER_API_KEY` environment variable set to a valid [OpenRouter](https://openrouter.ai/) API key:

```bash
export OPENROUTER_API_KEY=your_key_here
```

Pre-collected database dumps (`gpuflops_db.dump`, `code_features_db.dump`) are committed in the repo and can be restored with `--importDBDumpFile` to reproduce paper results without re-running LLM queries.

---

### Phase 1 — Build, Profile & Collect Data

#### Step 1.1 — Build all benchmarks

```bash
./runBuild.sh
```

Compiles all CUDA and OpenMP benchmarks using the LLVM toolchain. Executables land in `build/bin/cuda/` (488 expected) and `build/bin/omp/` (318 expected).

#### Step 1.2 — Profile with Nsight Compute

```bash
./runProfiling.sh
```

Runs `cuda-profiling/gatherData.py` against the installed GPU. This step must be run separately on each target GPU (3080, A10, A100, H100). Produces:
- `cuda-profiling/gpuData.csv` — per-kernel roofline metrics
- `cuda-profiling/ncu-rep-results/*.ncu-rep` — raw Nsight Compute reports
- `cuda-profiling/profiling-results-*.zip` — archive containing the CSV, logs, and NCU reports

Common options passed through to `gatherData.py`:
- `--cudaOnly` / `--ompOnly`: limit to CUDA or OpenMP targets
- `--samples N`: repeat profiling per target (default: 3)
- `--timeout SEC`: per-run timeout (default: 120)
- `--rerunTimeouts`: re-run targets that previously timed out
- `--skipConfirm`: skip the interactive confirmation prompt
- `--zipOnly`: skip profiling and only create the results archive

#### Step 1.3 — Extract SASS from built executables

```bash
cd cuda-profiling/collected-data
python extact_sass_from_built_executables.py
```

Disassembles executables in `build/bin/` using `cuobjdump` and `llvm-objdump`. Produces:
- `cuda-profiling/collected-data/scraped-sass/*.sass` — SASS disassembly per benchmark
- `cuda-profiling/collected-data/scraped-sass/sass_files.zip` — zipped SASS archive

> **Using pre-collected data**: if you want to skip this step, a committed `sass_files.zip` is already in `scraped-sass/`. Run `python unzip_collected_data.py --extract` (see Step 1.4) to restore all `.sass` files from it.

#### Step 1.4 — Collect profiling results from all GPUs

After profiling on each GPU system, collect the results into `cuda-profiling/collected-data/`. The helper script `cuda-profiling/collected-data/unzip_collected_data.py` handles this automatically.

```bash
cd cuda-profiling/collected-data

# Preview what would be extracted (safe, no files written):
python unzip_collected_data.py

# Extract when ready:
python unzip_collected_data.py --extract
```

The script unpacks each `NVIDIA_*_profiling-results-*.zip` into the matching GPU subdirectory (`3080/`, `A10/`, `A100/`, `H100/`) and, if present, restores SASS files from `scraped-sass/sass_files.zip` back into `scraped-sass/`.  Pre-collected archives for all four GPUs are committed in `cuda-profiling/collected-data/NVIDIA*.zip` and can be extracted this way without re-running profiling.

Files that already exist at the destination are skipped by default; pass `--overwrite` to replace them.

Each GPU subdirectory must contain the `*.ncu-rep` report files before the next step.

#### Step 1.5 — Condense per-GPU NCU reports into a single CSV

```bash
cd cuda-profiling/collected-data
python condense_perf_counter_data.py
```

Reads `*.ncu-rep` files from the `3080/`, `A10/`, `A100/`, and `H100/` subdirectories and produces:
- `cuda-profiling/collected-data/all-NCU-GPU-Data.csv`

This CSV can also be visualized interactively with `cuda-profiling/collected-data/compare_gpus.ipynb`.
If you have another GPU, you'll need to manually add it to the script to parse the manually-collected `*.ncu-rep` files correctly.

#### Step 1.6 — Scrape benchmark source files

```bash
python dataset-creation/scrape-sources.py
```

Parses compiler-generated `.d` dependency files in `build/src/` to map each benchmark to the source and header files it compiled. Produces:
- `dataset-creation/scraped_sources.json`

---

### Phase 2 — Build the gpuFLOPBench Dataset

```bash
python dataset-creation/make-gpuFLOPBench-dataset.py
```

Merges profiling metrics, SASS disassembly, and scraped source code into a single structured JSON. Depends on:
- `cuda-profiling/collected-data/all-NCU-GPU-Data.csv`
- `dataset-creation/scraped_sources.json`
- `cuda-profiling/collected-data/scraped-sass/`

Produces:
- `dataset-creation/gpuFLOPBench.json` — the dataset read by all LLM experiment scripts

---

### Phase 3 — LLM Experiments

Both experiments use [OpenRouter](https://openrouter.ai/) to query LLMs and persist results in local PostgreSQL databases via LangGraph checkpoints. Scripts auto-start PostgreSQL if the local cluster is offline.

Run a **dry run** first to verify API connectivity before committing to a full batch:

```bash
# Feature-voting dry run
cd experiments/feature-voting
python run_voting_queries.py --singleDryRun --modelNames "openai/gpt-5.1-codex-mini"

# Direct-prompting dry run
cd experiments/direct-prompting
python run_queries.py --singleDryRun --modelName "openai/gpt-5.1-codex-mini"
```

To reproduce paper results without re-running LLM queries, restore the committed database dumps:

```bash
# Restore feature-voting database
cd experiments/feature-voting
python run_voting_queries.py --importDBDumpFile code_features_db.dump --exportDBOnly

# Restore direct-prompting database
cd experiments/direct-prompting
python run_queries.py --importDBDumpFile gpuflops_db.dump --exportDBOnly
```

#### Step 3.1 — Feature voting: classify kernel code features

```bash
cd experiments/feature-voting
python run_voting_queries.py \
    --trials 3 \
    --modelNames "openai/gpt-5.1-codex-mini" \
    --queryBatchSize 4
```

Asks LLMs to statically classify 12 boolean code-feature flags for each benchmark kernel using source code only (no GPU or hardware data). Stores results in the `code_features_db` PostgreSQL database. See `experiments/feature-voting/runExperiments.sh` for the multi-model invocations used in the paper.

Key options:
- `--modelNames`: comma-separated OpenRouter model identifiers (multiple models per run supported)
- `--trials N`: repeat trials per kernel per model (default: 1)
- `--queryBatchSize N`: parallel queries per batch (default: 4)
- `--maxSpend USD`: hard spend cap for this run
- `--dumpDBOnFinish`: export `code_features_db.dump` when the run completes

#### Step 3.2 — Direct prompting: predict arithmetic intensity and DRAM traffic

```bash
cd experiments/direct-prompting
python run_queries.py \
    --trials 1 \
    --modelName "openai/gpt-5.1-codex-mini" \
    --queryBatchSize 4
```

Queries LLMs to predict per-kernel arithmetic intensity (AI) and DRAM traffic from benchmark source code. Optionally augmented with SASS disassembly (`--useSASS`) or static instruction-mix data (`--useIMIX`). Stores results in the `gpuflops_db` PostgreSQL database. See `experiments/direct-prompting/runExperiments.sh` for the multi-configuration invocations used in the paper.

Key options:
- `--modelName`: OpenRouter model identifier (default: `openai/gpt-5.1-codex-mini`)
- `--useSASS`: include SASS disassembly in the prompt
- `--useIMIX`: include static instruction-mix data in the prompt
- `--trials N`: repeat trials per query (default: 1)
- `--queryBatchSize N`: parallel queries per batch (default: 1)
- `--maxSpend USD`: hard spend cap for this run
- `--dumpDBOnFinish`: export `gpuflops_db.dump` when the run completes

---

### Phase 4 — Paper Figures, Tables & Listings

All figure-generation scripts read directly from the PostgreSQL databases populated in Phase 3. The exact commands below are those verified by `unit-tests/test_artifact_evaluation.py`.

#### Step 4.1 — Export paper prompt listings

```bash
cd experiments/direct-prompting
python print_prompt_for_paper_listing_1.py \
    --listing1Path listing1.txt \
    --listing2Path listing2.txt
```

Produces:
- `experiments/direct-prompting/listing1.txt` → Listing 1
- `experiments/direct-prompting/listing2.txt` → Listing 3

#### Step 4.2 — Direct-prompting paper figures

```bash
cd experiments/direct-prompting
python make_plots_for_paper.py \
    --onlySharedSamples \
    --outputDir paper-figure-output
```

Produces:
- `paper-figure-output/figure6_expected_rai_distribution_by_gpu_precision.png` → Figure 2
- `paper-figure-output/figure11_ai_percent_difference_boxplots.png` → Figure 6
- `paper-figure-output/figure2_5_ai_bound_confusion_heatmaps_with_zero.png` → Figure 7
- `paper-figure-output/table_figure12_8_threshold_coverage.tex` → Table 3

#### Step 4.3 — OpenRouter request-metadata figures

```bash
cd experiments/direct-prompting
python fetch_openrouter_request_metadata.py \
    --makePlotsForPaper \
    --onlySharedSamples \
    --plotOutputDir paper-figure-output/request-metadata
```

Fetches per-request timing and cost metadata from the OpenRouter API (requires `OPENROUTER_API_KEY`) and produces:
- `paper-figure-output/request-metadata/plot3_cost_distribution.png` → Figure 9
- `paper-figure-output/request-metadata/plot2_query_time_distribution.png` → Figure 10

#### Step 4.4 — Error-analysis feature-association figure

```bash
cd experiments/error-analysis
python make_plots_for_paper.py --outputDir paper-figure-output
```

Joins `gpuflops_db` and `code_features_db`, computes Cliff's delta associations between code features and AI prediction error, and produces:
- `paper-figure-output/figure1_model_feature_association_heatmap.png` → Figure 8

---

### Phase 5 — Run Tests

```bash
./runTests.sh --noGPU    # skip GPU-dependent and artifact-evaluation tests
./runTests.sh            # full suite including artifact evaluation
```

The artifact evaluation tests (`unit-tests/test_artifact_evaluation.py`) re-run each figure-generation script into a temporary directory and verify SHA-256 hashes against the committed reference outputs in `paper-figure-output/`.

---

## Paper Artifacts

| Paper artifact | Generated file |
|---|---|
| Figure 2 | `experiments/direct-prompting/paper-figure-output/figure6_expected_rai_distribution_by_gpu_precision.png` |
| Figure 6 | `experiments/direct-prompting/paper-figure-output/figure11_ai_percent_difference_boxplots.png` |
| Figure 7 | `experiments/direct-prompting/paper-figure-output/figure2_5_ai_bound_confusion_heatmaps_with_zero.png` |
| Table 3 | `experiments/direct-prompting/paper-figure-output/table_figure12_8_threshold_coverage.tex` |
| Figure 8 | `experiments/error-analysis/paper-figure-output/figure1_model_feature_association_heatmap.png` |
| Figure 9 | `experiments/direct-prompting/paper-figure-output/request-metadata/plot3_cost_distribution.png` |
| Figure 10 | `experiments/direct-prompting/paper-figure-output/request-metadata/plot2_query_time_distribution.png` |
| Listing 1 | `experiments/direct-prompting/listing1.txt` |
| Listing 3 | `experiments/direct-prompting/listing2.txt` |

---

## Experiments

### Feature Voting (`experiments/feature-voting/`)

The feature-voting experiment asks one or more LLMs to inspect benchmark source code and return a structured boolean checklist of code features for the first execution path of the target kernel. This is a data-collection experiment only — it does not use GPU metrics, SASS, or IMIX evidence. Each query covers a single (program, kernel, model, trial) combination and is GPU-agnostic.

The 12 boolean feature flags include:
- **Execution-path flags**: `has_branching`, `has_data_dependent_branching`, `has_flop_division`, `uses_preprocessor_defines`, `has_common_float_subexpr`, `has_loop_invariant_flops`, `has_special_math_functions`, `calls_device_function`
- **Host-side setup flags**: `has_rng_input_data`, `reads_input_values_from_file`, `has_constant_propagatable_gridsz`, `has_constant_propagatable_blocksz`

Key files:
- `run_voting_queries.py` — top-level runner; reads `gpuFLOPBench.json`, manages resume/retry, writes to `code_features_db`
- `graph.py` — LangGraph `StateGraph` for the single-query pipeline
- `prompts.py` — `CodeFeatureFlags` Pydantic model and XML prompt generator
- `db_manager.py` — PostgreSQL lifecycle, dump/restore, checkpoint parsing
- `code_features_db.dump` — pre-collected results dump (restore with `--importDBDumpFile`)

### Direct Prompting (`experiments/direct-prompting/`)

The direct-prompting experiment queries LLMs to predict per-kernel arithmetic intensity (AI) and DRAM traffic from benchmark source code, optionally augmented with SASS disassembly or static instruction-mix (IMIX) data. Predictions are compared against NCU ground-truth measurements across four GPUs (3080, A10, A100, H100).

Key files:
- `run_queries.py` — top-level runner; reads `gpuFLOPBench.json`, manages resume/retry, writes to `gpuflops_db`
- `graph.py` — LangGraph `StateGraph` for the single-query pipeline
- `prompts.py` — structured output models and XML prompt generator
- `db_manager.py` — PostgreSQL lifecycle, dump/restore, checkpoint parsing
- `result_viz_helper.py` — shared library for extracting data from PostgreSQL checkpoints and plot utilities (no standalone CLI)
- `make_plots_for_paper.py` — generates Figures 2, 6, 7 and Table 3
- `fetch_openrouter_request_metadata.py` — fetches per-request cost/timing metadata from OpenRouter; generates Figures 9 and 10
- `print_prompt_for_paper_listing_1.py` — exports Listings 1 and 3
- `gpuflops_db.dump` — pre-collected results dump (restore with `--importDBDumpFile`)

### Error Analysis (`experiments/error-analysis/`)

The error-analysis experiment joins results from `gpuflops_db` and `code_features_db` to study which kernel code features are associated with higher AI prediction error. It uses Cliff's delta to measure present-vs.-absent effect sizes across all (GPU, runtime, precision, model, prompt-type) combinations, rendering feature-association heatmaps for the paper.

Key files:
- `db_reader.py` — loads and merges both databases; produces `sample_with_features_df` for plotting
- `make_plots_for_paper.py` — computes Cliff's delta associations and renders heatmaps; produces Figure 8

---

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
ssh-keygen -t ed25519 -C "gbolet@vt.edu"
cat ~/.ssh/id_ed25519.pub

# add the above key to your github ssh authentication

git clone --recurse-submodules git@github.com:gregbolet/gpuFLOPBench-updated.git

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

sudo apt-get install -y g++ gcc libstdc++-14-dev libboost-all-dev libgsl-dev

wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 21 all
rm llvm.sh
sudo apt-get clean

sudo apt-get install -y liboffload-21-dev libomp-21-dev

sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-21 100
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-21 100
sudo update-alternatives --install /usr/bin/llvm-objdump llvm-objdump /usr/bin/llvm-objdump-21 100
sudo update-alternatives --install /usr/bin/llvm-cxxfilt llvm-cxxfilt /usr/bin/llvm-cxxfilt-21 100

sudo apt -y autoremove

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -p ~/anaconda3
rm ./Miniconda3-latest-Linux-x86_64.sh
source ~/anaconda3/bin/activate
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main &&     conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create --name gpuflopbench-updated python=3.11 -y
conda activate gpuflopbench-updated 

cd ~/gpuFLOPBench-updated/
pip install -r requirements.txt

echo 'source ~/anaconda3/bin/activate' >> ~/.bashrc
echo 'conda activate gpuflopbench-updated' >> ~/.bashrc

echo 'export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/comm_libs/nvshmem/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/comm_libs/nccl/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/math_libs/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/extras/qd/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/cuda/lib64:/usr/local/cuda/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/comm_libs/nvshmem/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/comm_libs/nccl/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/math_libs/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/extras/qd/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/cuda/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

echo 'export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/cuda/13.0/bin:/usr/lib/llvm-21/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/extras/qd/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/comm_libs/mpi/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/cuda/bin:/usr/local/cuda/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/comm_libs/nvshmem/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/comm_libs/nccl/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/profilers/bin:/home/gbolet/.vscode-server/data/User/globalStorage/github.copilot-chat/debugCommand:/home/gbolet/.vscode-server/data/User/globalStorage/github.copilot-chat/copilotCli:/home/gbolet/.vscode-server/cli/servers/Stable-94e8ae2b28cb5cc932b86e1070569c4463565c37/server/bin/remote-cli:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/cuda/13.0/bin:/usr/lib/llvm-21/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/extras/qd/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/comm_libs/mpi/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/cuda/bin:/usr/local/cuda/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/comm_libs/nvshmem/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/comm_libs/nccl/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/profilers/bin:/home/gbolet/anaconda3/envs/gpuflopbench-updated/bin:/home/gbolet/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/home/gbolet/.vscode-server/extensions/ms-python.debugpy-2025.18.0-linux-x64/bundled/scripts/noConfigScripts:$PATH' >> ~/.bashrc

echo 'export CUDA_HOME=/usr/local/cuda-13.0' >> ~/.bashrc

echo 'cd ~/gpuFLOPBench-updated/' >> ~/.bashrc

source ~/.bashrc

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
├── HeCBench/                             # Git submodule: HeCBench benchmark suite
├── runBuild.sh                           # Build all CUDA/OMP benchmarks with LLVM
├── runProfiling.sh                       # Profile built executables with Nsight Compute
├── runTests.sh                           # Test runner
├── Dockerfile                            # Container definition
├── requirements.txt                      # Python package dependencies
├── build/                                # Build artifacts (generated)
│   └── bin/
│       ├── cuda/                         # CUDA executables (488 expected)
│       └── omp/                          # OMP executables (318 expected)
├── cuda-profiling/
│   ├── gatherData.py                     # Profiles built code; writes gpuData.csv and *.ncu-rep
│   ├── utils.py                          # Demangling and kernel discovery helpers
│   └── collected-data/
│       ├── extact_sass_from_built_executables.py  # Extracts SASS from build/bin/ executables
│       ├── condense_perf_counter_data.py          # Merges per-GPU *.ncu-rep → all-NCU-GPU-Data.csv
│       ├── compare_gpus.ipynb                     # Interactive visualization of collected data
│       ├── all-NCU-GPU-Data.csv                   # Merged profiling data across all GPUs (generated)
│       ├── scraped-sass/                          # Per-benchmark SASS disassembly (generated)
│       ├── NVIDIA*.zip                            # Pre-collected per-GPU profiling archives
│       └── {3080,A10,A100,H100}/                  # Per-GPU unpacked NCU reports
├── dataset-creation/
│   ├── scrape-sources.py                 # Parses .d deps → scraped_sources.json
│   ├── make-gpuFLOPBench-dataset.py      # Merges profiling + SASS + sources → gpuFLOPBench.json
│   ├── sass_helper.py                    # SASS parsing helpers
│   ├── sass_objs.py                      # SASS object model
│   ├── scraped_sources.json              # Source files per benchmark (generated)
│   └── gpuFLOPBench.json                 # Structured LLM experiment dataset (generated)
├── experiments/
│   ├── llm_models.py                     # Shared LLM factory helpers (OpenRouter/Azure)
│   ├── feature-voting/
│   │   ├── run_voting_queries.py         # Runner: gpuFLOPBench.json → code_features_db
│   │   ├── graph.py                      # LangGraph pipeline for feature classification
│   │   ├── prompts.py                    # CodeFeatureFlags Pydantic model and prompt generator
│   │   ├── db_manager.py                 # PostgreSQL lifecycle, dump/restore, checkpoint parsing
│   │   └── code_features_db.dump         # Pre-collected results dump
│   ├── direct-prompting/
│   │   ├── run_queries.py                # Runner: gpuFLOPBench.json → gpuflops_db
│   │   ├── graph.py                      # LangGraph pipeline for AI/DRAM prediction
│   │   ├── prompts.py                    # Structured output models and XML prompt generator
│   │   ├── db_manager.py                 # PostgreSQL lifecycle, dump/restore, checkpoint parsing
│   │   ├── result_viz_helper.py          # Shared DB-extraction and plot utilities (no CLI)
│   │   ├── make_plots_for_paper.py       # Figures 2, 6, 7 and Table 3
│   │   ├── fetch_openrouter_request_metadata.py   # Figures 9 and 10
│   │   ├── print_prompt_for_paper_listing_1.py    # Listings 1 and 3
│   │   ├── gpuflops_db.dump              # Pre-collected results dump
│   │   └── paper-figure-output/          # Generated paper figures
│   └── error-analysis/
│       ├── db_reader.py                  # Loads and merges gpuflops_db + code_features_db
│       ├── make_plots_for_paper.py       # Figure 8: feature-association heatmaps
│       └── paper-figure-output/          # Generated paper figures
├── unit-tests/
│   ├── conftest.py
│   ├── test_artifact_evaluation.py       # SHA-256 checks against committed reference figures
│   ├── test_build_artifacts.py
│   ├── test_demangling.py
│   ├── test_kernel_extraction.py
│   └── ...
└── AGENTS.md                             # Full documentation
```

## Key Features

- **LLVM Toolchain**: Uses clang/clang++ for all compilation (including CUDA)
- **Roofline Profiling**: Gathers FLOP/s, arithmetic intensity, memory traffic and per-sample metrics
- **Kernel Discovery**: Automatic extraction, demangling, and library-kernel filtering
- **Input Handling**: Uses benchmarks.yaml args and Makefile run targets (with path resolution)
- **Comprehensive Tests**: Validates build artifacts, kernel extraction, demangling

## References

- **HeCBench**: https://github.com/zjin-lcf/HeCBench
- **Original gpuFLOPBench**: https://github.com/Scientific-Computing-Lab/gpuFLOPBench
- **Nsight Compute**: https://developer.nvidia.com/nsight-compute

## License

See [LICENSE](LICENSE) file.
