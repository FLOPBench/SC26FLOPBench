EXPECTED_TREE = (
    "particlefilter-cuda/\n"
    "  kernel_find_index.h\n"
    "  kernel_likelihood.h\n"
    "  kernel_normalize_weights.h\n"
    "  kernel_sum.h\n"
    "  main.cu\n"
    "  Makefile"
)

EXPECTED_MAIN_FILES = ["main.cu"]

EXPECTED_KERNELS = [
    {"file": "kernel_find_index.h", "kernel": "kernel_find_index", "line": 1},
    {"file": "kernel_likelihood.h", "kernel": "kernel_likelihood", "line": 1},
    {"file": "kernel_normalize_weights.h", "kernel": "kernel_normalize_weights", "line": 1},
    {"file": "kernel_sum.h", "kernel": "kernel_sum", "line": 1},
]
