EXPECTED_TREE = (
    "/\n"
    "  Copyright\n"
    "  kernel.h\n"
    "  License\n"
    "  main.cu\n"
    "  Makefile\n"
    "  rep.h"
)

EXPECTED_MAIN_FILES = ["main.cu"]

EXPECTED_INCLUDE_TREES = {
    "main.cu": """main.cu
  #include <stdio.h> (DNE)
  #include <stdlib.h> (DNE)
  #include <stdint.h> (DNE)
  #include <sys/time.h> (DNE)
  #include "kernel.h"
    #include <inttypes.h> (DNE)
    #include <type_traits> (DNE)
    #include <typeinfo> (DNE)
    #include <cuda_fp16.h> (DNE)
    #include "rep.h"

""",
}

EXPECTED_KERNELS = [
    {"file": "kernel.h", "kernel": "block_stride", "line": 39},
    {"file": "kernel.h", "kernel": "block_stride", "line": 85},
]

