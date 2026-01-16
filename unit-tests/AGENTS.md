# Unit Tests for gpuFLOPBench-updated

## Overview

This directory contains unit tests that validate the build and profiling infrastructure. Tests ensure that:
1. All expected executables are built
2. Executables contain GPU kernels
3. Kernel names demangle correctly for ncu profiling

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
- `test_expected_executables_exist()`: Verifies executables from benchmarks.yaml are built
- `test_executables_are_valid()`: Checks files are actually executable
- `test_no_unexpected_files()`: Ensures only executables (no .o, .so files) in build root

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
- `test_cuda_kernel_extraction()`: Extract kernels from CUDA binaries
- `test_omp_kernel_extraction()`: Extract kernels from OpenMP binaries
- `test_kernel_tools_available()`: Verify cuobjdump, objdump exist
- `test_sample_binaries_have_kernels()`: Random sample has kernels

**Requirements**:
- At least some built executables
- Tools: cuobjdump (or llvm-objdump), objdump

**Expected behavior**:
- For CUDA: Uses cuobjdump/llvm-objdump to find kernel symbols
- For OpenMP: Uses objdump with omp_offloading_entries section
- Reports if no kernels found (expected for some benchmarks)

### test_demangling.py

Verifies kernel name demangling works correctly.

**Tests**:
- `test_demangle_tools_available()`: Check cu++filt, c++filt, llvm-cxxfilt
- `test_demangle_cuda_names()`: Demangle known CUDA mangled names
- `test_demangle_omp_names()`: Verify OMP names don't need demangling
- `test_extract_simple_name()`: Test template/namespace removal
- `test_demangling_first_try()`: Ensure demangling succeeds on first attempt (bug fix test)

**Requirements**:
- At least one demangling tool (cu++filt, c++filt, or llvm-cxxfilt)
- Sample mangled kernel names

**Expected behavior**:
- Demangling should work on first try (no fallback needed)
- Simple names should be extractable for ncu -k option
- Template and namespace components should be removed correctly

## Test Configuration

### pytest.ini

```ini
[pytest]
testpaths = .
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    gpu: marks tests requiring GPU (deselect with '-m "not gpu"')
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

```python
@pytest.fixture
def build_dir():
    """Path to build directory"""
    return Path(__file__).parent.parent / "build"

@pytest.fixture
def src_dir():
    """Path to HeCBench source directory"""
    return Path(__file__).parent.parent / "HeCBench" / "src"

@pytest.fixture
def benchmarks_yaml():
    """Parsed benchmarks.yaml"""
    yaml_path = Path(__file__).parent.parent / "HeCBench" / "benchmarks.yaml"
    with open(yaml_path) as f:
        return yaml.safe_load(f)

@pytest.fixture
def sample_executables(build_dir):
    """Sample of built executables for testing"""
    exes = list(build_dir.glob("*"))
    exes = [e for e in exes if e.is_file() and os.access(e, os.X_OK)]
    return exes[:10]  # Sample of 10
```

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
3. **Changed demangling logic**: Update `test_demangling.py`
4. **New YAML fields**: Update fixtures in `conftest.py`

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
