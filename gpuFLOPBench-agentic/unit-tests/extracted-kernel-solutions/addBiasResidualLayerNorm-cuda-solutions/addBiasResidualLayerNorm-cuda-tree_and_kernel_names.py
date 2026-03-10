EXPECTED_TREE = (
    "/\n"
    "  kernels.h\n"
    "  main.cu\n"
    "  Makefile"
)

EXPECTED_MAIN_FILES = ["main.cu"]

EXPECTED_INCLUDE_TREES = {
    "main.cu": """main.cu
  #include <algorithm> (DNE)
  #include <chrono> (DNE)
  #include <random> (DNE)
  #include <cuda.h> (DNE)
  #include <cuda_fp16.h> (DNE)
  #include <cuda_bf16.h> (DNE)
  #include "kernels.h"

""",
}

EXPECTED_KERNELS = [
    {"file": "kernels.h", "kernel": "addBiasResidualPostLayerNormV2", "line": 202},
    {"file": "kernels.h", "kernel": "addBiasResidualPostLayerNorm", "line": 275},
    {
        "file": "kernels.h",
        "kernel": "generalAddBiasResidualPostLayerNorm",
        "line": 327,
    },
]

