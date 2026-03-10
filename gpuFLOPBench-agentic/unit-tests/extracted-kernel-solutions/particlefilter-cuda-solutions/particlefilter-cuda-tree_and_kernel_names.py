EXPECTED_TREE = (
    "/\n"
    "  kernel_find_index.h\n"
    "  kernel_likelihood.h\n"
    "  kernel_normalize_weights.h\n"
    "  kernel_sum.h\n"
    "  main.cu\n"
    "  Makefile"
)

EXPECTED_MAIN_FILES = ["main.cu"]

EXPECTED_INCLUDE_TREES = {
    "main.cu": """main.cu
  #include <stdio.h> (DNE)
  #include <stdlib.h> (DNE)
  #include <string.h> (DNE)
  #include <limits.h> (DNE)
  #include <math.h> (DNE)
  #include <unistd.h> (DNE)
  #include <fcntl.h> (DNE)
  #include <float.h> (DNE)
  #include <time.h> (DNE)
  #include <sys/time.h> (DNE)
  #include <iostream> (DNE)
  #include <cuda.h> (DNE)
  #include "kernel_find_index.h"
  #include "kernel_likelihood.h"
  #include "kernel_normalize_weights.h"
  #include "kernel_sum.h"

""",
}

EXPECTED_KERNELS = [
    {"file": "kernel_find_index.h", "kernel": "kernel_find_index", "line": 1},
    {"file": "kernel_likelihood.h", "kernel": "kernel_likelihood", "line": 1},
    {"file": "kernel_normalize_weights.h", "kernel": "kernel_normalize_weights", "line": 1},
    {"file": "kernel_sum.h", "kernel": "kernel_sum", "line": 1},
]

