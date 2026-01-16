"""
Shared pytest fixtures for gpuFLOPBench-updated tests
"""

import pytest
import yaml
import os
from pathlib import Path


@pytest.fixture
def repo_root():
    """Path to repository root directory"""
    return Path(__file__).parent.parent


@pytest.fixture
def build_dir(repo_root):
    """Path to build directory"""
    return repo_root / "build"


@pytest.fixture
def src_dir(repo_root):
    """Path to HeCBench source directory"""
    return repo_root / "HeCBench" / "src"


@pytest.fixture
def hecbench_root(repo_root):
    """Path to HeCBench root directory"""
    return repo_root / "HeCBench"


@pytest.fixture
def benchmarks_yaml(hecbench_root):
    """Parsed benchmarks.yaml from HeCBench"""
    yaml_path = hecbench_root / "benchmarks.yaml"
    
    if not yaml_path.exists():
        pytest.skip(f"benchmarks.yaml not found at {yaml_path}")
    
    with open(yaml_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def built_executables(build_dir):
    """List of all built executables"""
    if not build_dir.exists():
        pytest.skip(f"Build directory not found: {build_dir}")
    
    exes = []
    for entry in build_dir.iterdir():
        if entry.is_file() and os.access(entry, os.X_OK):
            # Skip non-executable files
            if not any(ext in entry.name for ext in ['.cpp', '.c', '.o', '.so', '.log']):
                exes.append(entry)
    
    return exes


@pytest.fixture
def sample_executables(built_executables):
    """Sample of built executables for faster testing"""
    # Return up to 10 executables as a sample
    return built_executables[:10]


@pytest.fixture
def cuda_executables(built_executables):
    """CUDA executables only"""
    return [exe for exe in built_executables if '-cuda' in exe.name]


@pytest.fixture
def omp_executables(built_executables):
    """OpenMP executables only"""
    return [exe for exe in built_executables if '-omp' in exe.name]
