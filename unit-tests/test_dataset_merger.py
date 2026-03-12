import pytest
import pandas as pd
import sys
import importlib.util
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
dataset_dir = ROOT_DIR / "dataset-creation"

# dynamically load the module with hyphens
spec = importlib.util.spec_from_file_location("make_dataset", dataset_dir / "make-gpuFLOPBench-dataset.py")
make_dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_dataset)

def test_fix_omp_kernel_name():
    assert make_dataset.fix_omp_kernel_name("_Z9my_kernelvi") == "_Z9my_kernelvi"
    omp_mangled = "__omp_offloading_10305_2b800c7_main_l81"
    assert make_dataset.fix_omp_kernel_name(omp_mangled) == "main_l81"
    
def test_rename_devices():
    assert make_dataset.rename_devices("NVIDIA GeForce RTX 3080") == "3080"
    assert make_dataset.rename_devices("NVIDIA A100-SXM4-40GB") == "A100"
    assert make_dataset.rename_devices("NVIDIA A10") == "A10"
    assert make_dataset.rename_devices("NVIDIA H100 PCIe") == "H100"
    with pytest.raises(ValueError):
        make_dataset.rename_devices("Unknown Device")

def test_get_program_name():
    row_cuda = pd.Series({'Process Name': 'accuracy', 'Kernel Name': '_Z11cuda_kernel'})
    assert make_dataset.get_program_name(row_cuda) == "accuracy-cuda"
    
    row_omp = pd.Series({'Process Name': 'accuracy', 'Kernel Name': '__omp_offloading_123_456_main_l12'})
    assert make_dataset.get_program_name(row_omp) == "accuracy-omp"

def test_extract_source_mapping():
    sources_dict = {
        "src/main.cu": "__global__ void my_kernel(int x) {}\n",
        "src/helper.h": "void helper_func() {}\n"
    }
    mapped = make_dataset.extract_source_mapping("accuracy-cuda", "_Z9my_kernelvi", "my_kernel(int)", sources_dict)
    assert "src/main.cu" in mapped
    
    sources_dict_omp = {
        "src/main.cpp": "int main() {\n  #pragma omp target\n  {}\n}\n",
        "src/other.cpp": "void other() {}\n"
    }
    mapped_omp = make_dataset.extract_source_mapping("accuracy-omp", "__omp_offloading_1_2_main_l2", "main:l2", sources_dict_omp)
    assert "src/main.cpp" in mapped_omp
