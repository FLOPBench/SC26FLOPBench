EXPECTED_TREE = (
    "/\n"
    "  kernels.h\n"
    "  Makefile\n"
    "  reduction.cu"
)

EXPECTED_MAIN_FILES = ["reduction.cu"]

EXPECTED_INCLUDE_TREES = {
    "reduction.cu": """reduction.cu
  #include <cstdio> (DNE)
  #include <cstdlib> (DNE)
  #include <iostream> (DNE)
  #include <chrono> (DNE)
  #include <cuda.h> (DNE)
  #include "kernels.h"

""",
}

EXPECTED_KERNELS = [
    {"file": "kernels.h", "kernel": "atomic_reduction", "line": 1},
    {"file": "kernels.h", "kernel": "atomic_reduction_v2", "line": 10},
    {"file": "kernels.h", "kernel": "atomic_reduction_v4", "line": 19},
    {"file": "kernels.h", "kernel": "atomic_reduction_v8", "line": 28},
    {"file": "kernels.h", "kernel": "atomic_reduction_v16", "line": 37},
]

EXPECTED_FUNCTION_DEFINITIONS = {
    "kernels.h": """__global__ void atomic_reduction(int *in, int* out, int arrayLength) (defnt)
__global__ void atomic_reduction_v2(int *in, int* out, int arrayLength) (defnt)
__global__ void atomic_reduction_v4(int *in, int* out, int arrayLength) (defnt)
__global__ void atomic_reduction_v8(int *in, int* out, int arrayLength) (defnt)
__global__ void atomic_reduction_v16(int *in, int* out, int arrayLength) (defnt)""",
    "reduction.cu": """int main(int argc, char** argv) (defnt)""",
}

EXPECTED_FUNCTION_DECLARATIONS = {}

