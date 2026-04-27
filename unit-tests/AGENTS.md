# Unit Tests for gpuFLOPBench-updated

## Overview

This directory contains unit tests that validate the build and profiling infrastructure, dataset creation utilities, and experiment scripts. Tests ensure that:
1. All expected executables are built
2. Executables contain GPU kernels
3. Kernel names demangle correctly for ncu profiling
4. Dataset-creation helpers produce correct outputs
5. Direct-prompting evidence configurations, DB parsing, and visualization logic are correct
6. Error-analysis paper-plot helpers compute correct statistics
7. Executable argument extraction from Makefiles works correctly
8. Feature-voting query scheduling works correctly
9. OMP kernel name demangling is correct
10. Source-scraping functions extract the right files
11. Result tabulation and shared-sample filtering logic is correct

## Test Framework

**pytest** - Python testing framework

### Installation

```bash
pip install pytest pytest-xdist
```

### Running Tests

```bash
# Run all tests
pytest -v

# Run specific test file
pytest test_build_artifacts.py -v

# Run specific test
pytest test_build_artifacts.py::test_expected_executables_exist -v

# Run with parallel execution
pytest -n auto

# Run with detailed output
pytest -vv -s
```

## Test Files

### test_build_artifacts.py

Verifies that the build process produces expected executables.

**Tests**:
- `test_build_directory_exists()`: Checks build directory exists
- `test_build_has_executables()`: Verifies at least some executables were built
- `test_executables_are_valid()`: Checks files are actually executable
- `test_cuda_executables_exist()`: Verifies CUDA executables are present in `build/bin/cuda/`
- `test_omp_executables_exist()`: Verifies OpenMP executables are present in `build/bin/omp/`
- `test_cuda_executable_threshold()`: Ensures at least 450 CUDA executables were built
- `test_omp_executable_threshold()`: Ensures at least 300 OpenMP executables were built
- `test_no_object_files_in_build_root()`: Ensures no stray `.o`/`.so` files in build root
- `test_executables_match_benchmark_names()`: Cross-checks built executables against benchmarks.yaml names
- `test_all_yaml_benchmarks_attempted()`: Warns about benchmarks listed in yaml but not built

**Requirements**:
- Build directory must exist (`../build`)
- HeCBench submodule must be initialized
- Benchmarks.yaml must be parseable

**Expected behavior**:
- Generates expected executable list from benchmarks.yaml
- Checks each expected executable exists and is executable
- Reports missing executables as test failures
- Allows some missing if build partially failed (warning, not failure)

### test_kernel_extraction.py

Verifies that GPU kernels can be extracted from binaries.

**Tests**:
- `test_cuobjdump_available()`: Verifies cuobjdump is on PATH
- `test_objdump_available()`: Verifies objdump is on PATH
- `test_llvm_objdump_available_or_skip()`: Checks for llvm-objdump (skips if absent)
- `test_cuda_kernel_extraction_sample()`: Extracts kernels from a sample of CUDA binaries
- `test_omp_kernel_extraction_sample()`: Extracts kernels from a sample of OpenMP binaries
- `test_cuda_kernel_names_not_empty()`: Asserts extracted CUDA kernel name lists are non-empty
- `test_omp_kernel_names_valid()`: Asserts extracted OpenMP offload entry names look valid
- `test_all_cuda_executables_have_kernels()` *(slow)*: Every CUDA binary has at least one kernel
- `test_all_omp_executables_have_kernels()` *(slow)*: Every OpenMP binary has at least one offload entry

**Requirements**:
- At least some built executables
- Tools: cuobjdump (or llvm-objdump), objdump

**Expected behavior**:
- For CUDA: Uses cuobjdump/llvm-objdump to find kernel symbols
- For OpenMP: Uses objdump with `omp_offloading_entries`/`llvm_offload_entries` section
- Reports if no kernels found (expected for some benchmarks)

### test_demangling.py

Verifies kernel name demangling works correctly.

**Tests**:
- `test_demangling_tools_available()`: Check that at least one of cu++filt, c++filt, llvm-cxxfilt is present
- `test_cxxfilt_basic_demangling()`: Demangle a known mangled name with c++filt
- `test_demangling_with_preferred_tool()`: Demangle using the tool selected by `utils.py`
- `test_demangling_first_try_succeeds()`: Ensures demangling succeeds on the first attempt (regression test)
- `test_extract_simple_name_removes_templates()`: Template parameters stripped correctly
- `test_extract_simple_name_removes_namespace()`: Namespace prefixes stripped correctly
- `test_extract_simple_name_removes_return_type()`: Return type prefixes stripped correctly
- `test_extract_simple_name_complex()`: Complex mangled names simplify to bare function name
- `test_demangle_preserves_template_parameters()`: Full demangled form retains template info
- `test_extract_simple_name_already_simple()`: Already-simple names pass through unchanged
- `test_omp_kernel_names_dont_need_demangling()`: OMP offload entry names are left as-is
- `test_filter_library_kernels()`: cub:: and thrust:: kernels are filtered out
- `test_demangling_integration()`: End-to-end demangling from binary to readable name
- `test_demangling_multiple_tools()`: All available tools produce consistent results
- `test_empty_name_handling()`: Empty/None input is handled gracefully
- `test_parse_omp_offload_entries()`: objdump output parsed into entry list correctly
- `test_demangle_omp_offload_name_with_line()`: OMP names with line numbers demangle correctly
- `test_ncu_name_extraction_all_executables()` *(slow)*: NCU-compatible kernel names extracted from all built binaries
- `test_profiliable_kernel_counts()` *(slow)*: Count of profiliable kernels per binary matches expectations
- `test_cuda_non_profiliable_targets_have_no_source_kernels()` *(slow)*: Binaries with no profiliable kernels have no user-defined kernels

**Requirements**:
- At least one demangling tool (cu++filt, c++filt, or llvm-cxxfilt)
- Sample mangled kernel names

**Expected behavior**:
- Demangling should work on first try (no fallback needed)
- Simple names should be extractable for ncu -k option
- Template and namespace components should be removed correctly

### test_dataset_merger.py

Verifies helper functions in `dataset-creation/make-gpuFLOPBench-dataset.py` used to merge and normalize profiling results into the final dataset.

**Tests**:
- `test_fix_omp_kernel_name()`: OMP mangled names (`__omp_offloading_...`) are shortened to `func_lN` form; non-OMP names are returned unchanged
- `test_rename_devices()`: GPU device strings are mapped to short canonical names (e.g. `"NVIDIA A100-SXM4-40GB"` → `"A100"`); unknown devices raise `ValueError`
- `test_get_program_name()`: CUDA processes produce `name-cuda` and OMP processes produce `name-omp` program names
- `test_extract_source_mapping()`: Source files relevant to a kernel are identified from the scraped sources dict

**Requirements**:
- `dataset-creation/make-gpuFLOPBench-dataset.py` must be importable

### test_direct_prompting_evidence_config.py

Verifies the direct-prompting experiment modules: evidence configurations, DB checkpoint parsing, shared-sample filtering, prompt generation, and visualization/plot helpers.

Covers `experiments/direct-prompting/` modules: `prompts.py`, `run_queries.py`, `result_viz_helper.py`, `make_plots_for_paper.py`, `db_manager.py`.

**Key test areas**:
- `test_evidence_thread_part_encodes_all_configurations()`: Thread ID parts correctly encode `use_sass`/`use_imix` flags
- `test_thread_metadata_parses_new_and_legacy_configurations()`: Both new and legacy thread-ID formats parse correctly
- `test_display_plot_model_name_shortens_supported_models()`: Model names are shortened for display
- `test_direct_prompt_generator_includes_optional_sections_independently()`: SASS and IMIX sections are included/excluded independently
- `test_system_prompt_distinguishes_sass_from_imix_roles()`: System prompt role differs for SASS vs IMIX
- `test_generate_system_prompt_omits/includes_imix_guidance_when_imix_is_absent/present()`: IMIX guidance section gated correctly
- `test_run_queries_parser_*()`: CLI argument parser defaults and flags
- `test_fetch_tail_checkpoint_for_thread_remains_strict_for_disconnected_history()`: Strict mode rejects broken checkpoint chains
- `test_fetch_tail_checkpoints_by_thread_skips_invalid_threads_in_tolerant_mode()`: Tolerant mode skips broken threads
- `test_database_dataframe_uses_valid_tails_and_ignores_skipped_invalid_threads()`: DataFrame built from valid tails only
- `test_filter_only_shared_samples_*()`: Multiple tests verifying that only kernel/GPU/model/prompt-type combinations present for all models/GPUs are retained
- `test_summarize_expected_rai_distribution_*()`: RAI distribution summaries count unique kernels per GPU/precision
- `test_classify_ai_with_zero_*()`: Zero bandwidth, zero compute, and NaN are classified correctly
- `test_figure*()`: Plot-saving helpers write PNG output files
- `test_percent_diff_axis_config_*()`, `test_ape_axis_config_*()`: Axis configuration helpers return correct parameters
- `test_summarize_pct_error_thresholds_*()`: Threshold tables count correctly
- `test_write_paper_summary_tables_*()`: Paper summary table output is correct

**Requirements**:
- `experiments/direct-prompting/` modules must be importable (DB connection is stubbed)

### test_error_analysis_paper_plots.py

Verifies statistical helper functions and plot-saving utilities in `experiments/error-analysis/make_plots_for_paper.py`.

**Tests**:
- `test_cliffs_delta_extremes()`: Cliff's delta returns +1 / -1 for fully separated distributions
- `test_build_association_dataframe_computes_feature_effects()`: Feature-effect DataFrame has correct columns and values
- `test_clean_sample_dataframe_uses_prompt_type_and_drops_imix_rows()`: IMIX rows are removed and prompt_type is set correctly
- `test_format_model_label_normalizes_claude_opus_variant()`: Claude Opus variant names are normalized
- `test_filter_only_shared_samples_keeps_model_matched_rows()`: Shared-sample filtering keeps only identities present across all models
- `test_build_association_dataframe_supports_collapsed_variants()`: Collapsed model variants are handled
- `test_build_runtime_feature_summary_dataframe_collapses_all_but_runtime()`: Summary DF groups correctly by runtime
- `test_build_gpu_feature_summary_dataframe_collapses_all_but_gpu()`: Summary DF groups correctly by GPU
- `test_save_gpu_feature_summary_heatmap_writes_output()`: Heatmap PNG is written to disk
- `test_build_model_feature_summary_dataframe_collapses_all_but_model()`: Summary DF groups correctly by model
- `test_build_model_prompt_type_feature_summary_dataframe_collapses_runtime()`: Summary DF groups by model × prompt type
- `test_save_model_feature_summary_heatmap_writes_output()`: Model heatmap PNG is written to disk
- `test_save_model_prompt_type_feature_summary_heatmap_writes_output()`: Model × prompt-type heatmap PNG is written
- `test_runtime_feature_order_sorts_by_signed_error_association()`: Feature ordering follows signed error association

**Requirements**:
- `experiments/error-analysis/make_plots_for_paper.py` must be importable

### test_exe_args.py

Verifies Makefile-based executable argument extraction via `cuda-profiling/utils.py`.

**Tests**:
- `test_makefile_args_for_built_targets()`: For each built CUDA/OpenMP executable, the Makefile `run` target args are parseable and consistent with benchmarks.yaml

**Requirements**:
- Built executables in `build/bin/`
- `HeCBench/src/` with Makefiles
- `HeCBench/benchmarks.yaml`

### test_feature_voting_scheduler.py

Verifies the async query-scheduling logic in `experiments/feature-voting/`.

**Tests**:
- `test_run_queries_refills_open_slots_without_waiting_for_full_batch()`: The scheduler refills open query slots immediately when capacity is available, without waiting for a full batch to drain

**Requirements**:
- `experiments/feature-voting/` modules must be importable (LLM and DB calls are monkeypatched)

### test_omp_demangle_sass_imix.py

Verifies OMP kernel name demangling functions in `dataset-creation/make-gpuFLOPBench-dataset.py`.

**Tests**:
- `test_fix_omp_kernel_name()`: `__omp_offloading_<hash>_<hash>_<func>_l<N>` patterns are stripped to `<func>_l<N>`; names with embedded mangled C++ symbols are handled correctly
- `test_get_demangled_omp_name()`: `get_demangled_omp_name()` returns a human-readable `func:lN` string

**Requirements**:
- `dataset-creation/make-gpuFLOPBench-dataset.py` must be importable
- At least one demangling tool (c++filt / llvm-cxxfilt) for the C++ embedded-name case

### test_source_scraping_functions.py

Verifies source-file scraping functions in `dataset-creation/scrape-sources.py`.

**Tests**:
- `test_lulesh_cuda_extraction()`: `get_benchmark_files()` returns the expected dependency files for `lulesh-cuda`, including `lulesh.cu`
- `test_scraped_json_structure_and_content()`: The pre-built `scraped_sources.json` has the correct top-level structure and contains expected benchmark entries
- `test_no_system_headers_included()`: Scraped source dicts do not include system header paths

**Requirements**:
- `dataset-creation/scrape-sources.py` must be importable
- `dataset-creation/scraped_sources.json` must exist for the JSON structure test
- Built executables in `build/bin/` for the extraction test

## Test Configuration

### pytest.ini

```ini
[pytest]
testpaths = .
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short --strict-markers
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    gpu: marks tests requiring GPU (deselect with '-m "not gpu"')
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
```

### Markers

Tests are marked with categories:

- **@pytest.mark.slow**: Tests that take significant time
- **@pytest.mark.gpu**: Tests requiring actual GPU (usually none for symbol inspection)

Example:
```python
@pytest.mark.slow
def test_all_executables_have_kernels():
    # Test every single executable
    pass

@pytest.mark.gpu
def test_kernel_executes():
    # Actually run on GPU
    pass
```

Run without slow tests:
```bash
pytest -m "not slow"
```

## Test Data

### Fixtures

**conftest.py** defines shared fixtures:

- `repo_root`: Path to repository root
- `build_dir`: `repo_root / "build"`
- `src_dir`: `repo_root / "HeCBench" / "src"`
- `hecbench_root`: `repo_root / "HeCBench"`
- `benchmarks_yaml`: Parsed `HeCBench/benchmarks.yaml`; skips if file not found
- `built_executables`: All executable files under `build/bin/cuda/` and `build/bin/omp/`; skips if build directory missing
- `sample_executables`: First 10 entries from `built_executables`
- `cuda_executables`: Executables from `build/bin/cuda/` only
- `omp_executables`: Executables from `build/bin/omp/` only

## Test Execution Flow

### Continuous Integration

Tests are designed to run in CI without GPU:

1. **Build phase**: Run `./runBuild.sh`
2. **Test phase**: Run `pytest -m "not gpu"`
3. **Report**: Generate test report and coverage

### Local Development

With GPU available:

1. **Build**: `./runBuild.sh`
2. **Test**: `pytest -v`
3. **Profile**: Tests validate, then run `gatherData.py`

## Writing New Tests

### Guidelines

1. **Deterministic**: Tests should be repeatable
2. **Independent**: Each test should be self-contained
3. **Fast by default**: Mark slow tests with @pytest.mark.slow
4. **Clear assertions**: Use descriptive assertion messages
5. **Skip when appropriate**: Use @pytest.mark.skipif for unavailable tools

### Example Test

```python
import pytest
import subprocess
from pathlib import Path

def test_cuobjdump_extracts_kernels(sample_executables):
    """Verify cuobjdump can extract kernel names from CUDA binaries"""
    
    # Skip if cuobjdump not available
    try:
        subprocess.run(['cuobjdump', '--version'], 
                      capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("cuobjdump not available")
    
    # Test on CUDA executables
    cuda_exes = [e for e in sample_executables if '-cuda' in e.name]
    
    if not cuda_exes:
        pytest.skip("No CUDA executables found")
    
    for exe in cuda_exes:
        result = subprocess.run(
            ['cuobjdump', '--list-text', str(exe)],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, \
            f"cuobjdump failed on {exe.name}"
        
        # Check for kernel indicators
        assert '.text.' in result.stdout, \
            f"No kernels found in {exe.name}"
```

## Troubleshooting Tests

### Common Issues

**Import errors**:
```bash
# Ensure pytest is installed
pip install pytest pyyaml

# Add parent directory to PYTHONPATH
export PYTHONPATH=$PWD/..:$PYTHONPATH
```

**Tool not found**:
```bash
# Tests should skip gracefully
# If test fails instead of skipping, check skipif logic
pytest -v --tb=long
```

**Build directory empty**:
```bash
# Run build first
cd ..
./runBuild.sh
cd unit-tests
pytest
```

**YAML parse errors**:
```bash
# Check HeCBench submodule
git submodule update --init
```

### Debugging Tests

```bash
# Run with print statements visible
pytest -s

# Run with full traceback
pytest --tb=long

# Run single test with debugging
pytest test_file.py::test_name -vv -s

# Drop into debugger on failure
pytest --pdb
```

## Test Coverage

### Running with Coverage

```bash
# Install coverage
pip install pytest-cov

# Run with coverage report
pytest --cov=../cuda-profiling --cov-report=html

# View report
open htmlcov/index.html
```

### Expected Coverage

- **gatherData.py**: ~70% (excluding GPU execution paths)
- **Test files**: 100%
- **Overall**: >60%

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
      with:
        submodules: true
    
    - name: Install dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y clang-15 cmake
    
    - name: Build benchmarks
      run: ./runBuild.sh
    
    - name: Run tests
      run: |
        pip install pytest pyyaml
        cd unit-tests
        pytest -v -m "not gpu"
```

## Maintenance

### Updating Tests

When modifying infrastructure:

1. **New benchmark models**: Update `test_build_artifacts.py` to recognize new suffixes
2. **New kernel extraction tools**: Add tests in `test_kernel_extraction.py`
3. **Changed demangling logic**: Update `test_demangling.py` and `test_omp_demangle_sass_imix.py`
4. **New YAML fields**: Update fixtures in `conftest.py`
5. **Dataset-creation helpers**: Update `test_dataset_merger.py`
6. **Direct-prompting experiment changes**: Update `test_direct_prompting_evidence_config.py`
7. **Error-analysis plot changes**: Update `test_error_analysis_paper_plots.py`
8. **Source scraping**: Update `test_source_scraping_functions.py`
9. **Feature-voting scheduler**: Update `test_feature_voting_scheduler.py`

### Test Data

Tests use real data from:
- Built executables in `../build/`
- `benchmarks.yaml` from HeCBench
- Actual binary inspection with standard tools

No mock data or test doubles needed for current tests.

## References

- **pytest documentation**: https://docs.pytest.org/
- **pytest fixtures**: https://docs.pytest.org/en/latest/fixture.html
- **pytest markers**: https://docs.pytest.org/en/latest/mark.html

## Contributing

When adding tests:

1. Follow existing naming conventions (`test_*.py`, `test_*()`)
2. Use fixtures for shared setup
3. Add markers for slow/gpu tests
4. Document test purpose in docstring
5. Update this AGENTS.md with new test descriptions
