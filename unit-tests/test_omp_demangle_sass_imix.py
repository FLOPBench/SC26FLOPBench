import pytest
import sys
import re
from pathlib import Path
import importlib.util

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "cuda-profiling"))

# Import make_gpuFLOPBench_dataset
spec = importlib.util.spec_from_file_location("make_gpuFLOPBench_dataset", str(ROOT_DIR / "dataset-creation" / "make-gpuFLOPBench-dataset.py"))
dataset_module = importlib.util.module_from_spec(spec)
sys.modules["make_gpuFLOPBench_dataset"] = dataset_module
spec.loader.exec_module(dataset_module)

fix_omp = dataset_module.fix_omp_kernel_name
demangle_omp = dataset_module.get_demangled_omp_name

def test_fix_omp_kernel_name():
    # Example 1: accuracy-omp variations
    k1 = "__omp_offloading_10305_2b800b0_main_l57"
    k2 = "__omp_offloading_fd01_1801e6_main_l57"
    k3 = "__omp_offloading_fd01_182746_main_l1310"
    k4 = "__omp_offloading_fd01_180667_binomialOptionsGPU_l100"
    
    assert fix_omp(k1) == "main_l57"
    assert fix_omp(k2) == "main_l57"
    assert fix_omp(k3) == "main_l1310"
    assert fix_omp(k4) == "binomialOptionsGPU_l100"

    # Example 2: complex mangled names
    k5 = "__omp_offloading_fd01_1802f2__Z7runTestI13uint2_alignedEiPhS1_ii_l196"
    assert fix_omp(k5) == "_Z7runTestI13uint2_alignedEiPhS1_ii_l196"

def test_get_demangled_omp_name():
    k1 = "__omp_offloading_10305_2b800b0_main_l57"
    assert demangle_omp(k1) == "main:l57"
    
    k3 = "__omp_offloading_fd01_1802f2__Z7runTestI13uint2_alignedEiPhS1_ii_l196"
    assert "int runTest<uint2_aligned>(unsigned char *, unsigned char *, int, int):l196" in demangle_omp(k3)

