EXPECTED_TREE = "all-pairs-distance-cuda/\n  main.cu\n  Makefile"

EXPECTED_MAIN_FILES = ["main.cu"]

EXPECTED_INCLUDE_TREES = {
    "main.cu": """main.cu
  #include <cstdio> (DNE)
  #include <cstdlib> (DNE)
  #include <cstring> (DNE)
  #include <cuda.h> (DNE)
  #include <cub/cub.cuh> (DNE)
  #include <sys/time.h> (DNE)

""",
}

EXPECTED_KERNELS = [
    {"file": "main.cu", "kernel": "k1", "line": 37},
    {"file": "main.cu", "kernel": "k2", "line": 66},
    {"file": "main.cu", "kernel": "k3", "line": 131},
]


EXPECTED_FUNCTION_DEFINITIONS = {
    "main.cu": """void CPU(int * data, int * distance) (defnt)
__global__ void k1 (const char *data, int *distance) (defnt)
__global__ void k2 (const char *data, int *distance) (defnt)
__global__ void k3 (const char *data, int *distance) (defnt)
int main(int argc, char **argv) (defnt)""",
}

EXPECTED_FUNCTION_DECLARATIONS = {}

