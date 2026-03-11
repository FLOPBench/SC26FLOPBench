import os
import json
import pytest
from pathlib import Path
import sys
import importlib

# Add the dataset-creation directory to path to import scrape-sources.py
sys.path.append(str(Path(__file__).parent.parent / "dataset-creation"))
scrape_sources_mod = importlib.import_module("scrape-sources")

get_benchmark_files = scrape_sources_mod.get_benchmark_files
scrape_sources = scrape_sources_mod.scrape_sources

_test_dir = Path(__file__).resolve().parent
_root_dir = _test_dir.parent
BUILD_DIR = str(_root_dir / "build")

def test_lulesh_cuda_extraction():
    """Test that lulesh-cuda dependency files are correctly parsed."""
    benchmark = "lulesh-cuda"
    paths = get_benchmark_files(benchmark, BUILD_DIR)
    
    # We should have found some files
    assert len(paths) > 0, f"No files found for {benchmark}"
    
    # Check that lulesh.cu is among them
    found_lulesh_cu = any("lulesh.cu" in p for p in paths)
    assert found_lulesh_cu, "lulesh.cu not found in the scraped sources"

def check_entrypoint(benchmark_dict):
    """Check if any file in the benchmark has a main function."""
    for filepath, content in benchmark_dict.items():
        if "int main(" in content or "main(" in content:
            return True
    return False

def check_cuda_keywords(benchmark_dict):
    """Check if any file has __global__ or __device__."""
    for filepath, content in benchmark_dict.items():
        if "__global__" in content or "__device__" in content:
            return True
    return False

def check_omp_keywords(benchmark_dict):
    """Check if any file has #pragma omp parallel or target."""
    for filepath, content in benchmark_dict.items():
        if "#pragma omp target" in content or "#pragma omp parallel" in content:
            return True
    return False

def test_scraped_json_structure_and_content():
    """Test the structure and contents of the generated JSON file."""
    # We will test on specific benchmarks to ensure they pass
    test_benchmarks = ["accuracy-cuda", "accuracy-omp"]
    
    data = {}
    for benchmark in test_benchmarks:
        paths = get_benchmark_files(benchmark, BUILD_DIR)
        data[benchmark] = {}
        for p in paths:
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    data[benchmark][p] = f.read()
            except Exception:
                pass
                
    # Verify accurate mapping
    assert "accuracy-cuda" in data
    assert "accuracy-omp" in data
    
    # verify nested structure: dict within dict
    assert isinstance(data["accuracy-cuda"], dict)
    
    # Entrypoint check
    assert check_entrypoint(data["accuracy-cuda"]), "No main() entrypoint found in accuracy-cuda"
    assert check_entrypoint(data["accuracy-omp"]), "No main() entrypoint found in accuracy-omp"
    
    # CUDA keywords check
    assert check_cuda_keywords(data["accuracy-cuda"]), "No __global__ or __device__ found in accuracy-cuda"
    
    # OpenMP keywords check
    assert check_omp_keywords(data["accuracy-omp"]), "No #pragma omp ... found in accuracy-omp"

def test_no_system_headers_included():
    """Test that all scraped paths belong strictly to HeCBench/src and no system headers leak."""
    test_benchmarks = ["accuracy-cuda", "accuracy-omp", "lulesh-cuda"]
    
    for benchmark in test_benchmarks:
        paths = get_benchmark_files(benchmark, BUILD_DIR)
        
        # Ensure we actually found paths to test against
        assert len(paths) > 0, f"No files found for {benchmark}"
        
        for p in paths:
            # We strictly enforce that the path descends into HeCBench/src to avoid pulling 
            # from places like /usr/include/ or /opt/nvidia/
            assert "/HeCBench/src/" in str(p), f"External or system header leaked into scrape: {p}"


