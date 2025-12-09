EXPECTED_TREE = (
    "ert-cuda/\n"
    "  Copyright\n"
    "  kernel.h\n"
    "  License\n"
    "  main.cu\n"
    "  Makefile\n"
    "  rep.h"
)

EXPECTED_KERNELS = [
    {"file": "kernel.h", "kernel": "block_stride", "line": 39},
    {"file": "kernel.h", "kernel": "block_stride", "line": 85},
]
