EXPECTED_TREE = (
    "addBiasResidualLayerNorm-cuda/\n"
    "  kernels.h\n"
    "  main.cu\n"
    "  Makefile"
)

EXPECTED_MAIN_FILES = ["main.cu"]

EXPECTED_KERNELS = [
    {"file": "kernels.h", "kernel": "addBiasResidualPostLayerNormV2", "line": 202},
    {"file": "kernels.h", "kernel": "addBiasResidualPostLayerNorm", "line": 275},
    {
        "file": "kernels.h",
        "kernel": "generalAddBiasResidualPostLayerNorm",
        "line": 327,
    },
]
