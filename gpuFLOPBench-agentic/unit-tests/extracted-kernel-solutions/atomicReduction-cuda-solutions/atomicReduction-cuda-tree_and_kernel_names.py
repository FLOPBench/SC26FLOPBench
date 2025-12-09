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
