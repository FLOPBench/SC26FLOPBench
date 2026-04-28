"""
Test suite for verifying build artifacts

Tests that expected executables are built from HeCBench benchmarks.
"""

import pytest
from pathlib import Path
import os


def test_build_directory_exists(build_dir):
    """Verify build directory exists"""
    assert build_dir.exists(), f"Build directory not found: {build_dir}"
    assert build_dir.is_dir(), f"Build path exists but is not a directory: {build_dir}"


def test_build_has_executables(built_executables):
    """Verify at least some executables were built"""
    assert len(built_executables) > 0, \
        "No executables found in build directory. Did the build succeed?"


def test_executables_are_valid(built_executables):
    """Verify all found executables are actually executable"""
    for exe in built_executables:
        assert exe.is_file(), f"{exe.name} is not a regular file"
        assert os.access(exe, os.X_OK), f"{exe.name} is not executable"


def test_cuda_executables_exist(cuda_executables):
    """Verify at least some CUDA executables were built"""
    assert len(cuda_executables) > 0, \
        "No CUDA executables found. Check if CUDA compilation succeeded."


def test_omp_executables_exist(omp_executables):
    """Verify at least some OpenMP executables were built"""
    assert len(omp_executables) > 0, \
        "No OpenMP executables found. Check if OpenMP compilation succeeded."


def test_cuda_executable_threshold(cuda_executables):
    """Verify CUDA executables meet minimum threshold (400)"""
    MIN_CUDA_EXECUTABLES = 400
    cuda_count = len(cuda_executables)
    
    assert cuda_count >= MIN_CUDA_EXECUTABLES, \
        f"CUDA executable count {cuda_count} below threshold {MIN_CUDA_EXECUTABLES}"
    
    print(f"\nCUDA executables: {cuda_count} (threshold: {MIN_CUDA_EXECUTABLES})")


def test_omp_executable_threshold(omp_executables):
    """Verify OpenMP executables meet minimum threshold (300)"""
    MIN_OMP_EXECUTABLES = 300
    omp_count = len(omp_executables)
    
    assert omp_count >= MIN_OMP_EXECUTABLES, \
        f"OpenMP executable count {omp_count} below threshold {MIN_OMP_EXECUTABLES}"
    
    print(f"\nOpenMP executables: {omp_count} (threshold: {MIN_OMP_EXECUTABLES})")


def test_no_object_files_in_build_root(build_dir):
    """Verify no .o or .so files in build root (should be in subdirs)"""
    for entry in build_dir.iterdir():
        if entry.is_file():
            assert not entry.name.endswith('.o'), \
                f"Object file in build root: {entry.name}"
            # .so files might be intentional libraries, so just warn
            if entry.name.endswith('.so'):
                print(f"WARNING: Shared library in build root: {entry.name}")


def test_executables_match_benchmark_names(built_executables, benchmarks_yaml):
    """Verify built executables correspond to benchmarks in YAML"""
    
    # Get benchmark names from YAML
    benchmark_names = set(benchmarks_yaml.keys())
    
    # Extract benchmark names from executable names
    # Current build outputs have no model suffix, but keep backward compatibility.
    exe_benchmarks = set()
    for exe in built_executables:
        name = exe.name
        stripped = name
        for suffix in ['-cuda', '-omp', '-hip', '-sycl']:
            if suffix in name:
                stripped = name.replace(suffix, '')
                break
        exe_benchmarks.add(stripped)
    
    # Check that some executables match YAML benchmarks
    matches = exe_benchmarks & benchmark_names
    
    assert len(matches) > 0, \
        f"No executables match benchmarks in YAML. Found: {exe_benchmarks}, Expected: {benchmark_names}"
    
    # Report coverage
    coverage = len(matches) / len(benchmark_names) * 100
    print(f"\nBuilt {len(exe_benchmarks)} unique benchmarks")
    print(f"Matched {len(matches)} benchmarks from YAML ({coverage:.1f}% coverage)")


@pytest.mark.slow
def test_all_yaml_benchmarks_attempted(built_executables, benchmarks_yaml):
    """
    Check if all benchmarks from YAML have at least one model built.
    
    Note: This is marked slow and may report many failures if build was incomplete.
    """
    
    # Get all expected benchmarks from YAML
    expected = set()
    for bench_name, bench_data in benchmarks_yaml.items():
        if 'models' in bench_data:
            for model in bench_data['models']:
                if model in ['cuda', 'omp']:  # Only check models we build
                    expected.add(bench_name)
                    break
    
    # Get actual executables (strip any model suffix for backward compatibility)
    actual = set()
    for exe in built_executables:
        name = exe.name
        stripped = name
        for suffix in ['-cuda', '-omp', '-hip', '-sycl']:
            if suffix in name:
                stripped = name.replace(suffix, '')
                break
        actual.add(stripped)
    
    # Find missing
    missing = expected - actual
    
    # Report
    print(f"\nExpected: {len(expected)} benchmarks")
    print(f"Built: {len(actual)} benchmarks")
    print(f"Missing: {len(missing)} benchmarks")
    
    if missing:
        print(f"\nSample missing executables (first 10):")
        for name in sorted(missing)[:10]:
            print(f"  - {name}")
    
    # Allow some failures (build may not complete 100%)
    success_rate = len(actual) / len(expected) * 100
    assert success_rate > 10, \
        f"Too few executables built: {success_rate:.1f}% success rate"
    
    print(f"\nBuild success rate: {success_rate:.1f}%")
