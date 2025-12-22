EXPECTED_TREE = (
    "atomicReduction-cuda/\n"
    "  kernels.h\n"
    "  Makefile\n"
    "  reduction.cu"
)

EXPECTED_MAIN_FILES = ["reduction.cu"]

EXPECTED_KERNELS = [
    {"file": "kernels.h", "kernel": "atomic_reduction", "line": 1},
    {"file": "kernels.h", "kernel": "atomic_reduction_v2", "line": 10},
    {"file": "kernels.h", "kernel": "atomic_reduction_v4", "line": 19},
    {"file": "kernels.h", "kernel": "atomic_reduction_v8", "line": 28},
    {"file": "kernels.h", "kernel": "atomic_reduction_v16", "line": 37},
]

EXPECTED_FUNCTION_DEFINITIONS = {
    "kernels.h": """__global__ void atomic_reduction (defnt)
__global__ void atomic_reduction_v2 (defnt)
__global__ void atomic_reduction_v4 (defnt)
__global__ void atomic_reduction_v8 (defnt)
__global__ void atomic_reduction_v16 (defnt)""",
    "reduction.cu": "int main (defnt)",
}

EXPECTED_FUNCTION_DECLARATIONS = {}
