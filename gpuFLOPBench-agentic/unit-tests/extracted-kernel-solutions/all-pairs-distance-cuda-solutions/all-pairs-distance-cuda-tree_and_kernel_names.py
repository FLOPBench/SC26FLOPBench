EXPECTED_TREE = "all-pairs-distance-cuda/\n  main.cu\n  Makefile"

EXPECTED_MAIN_FILES = ["main.cu"]

EXPECTED_KERNELS = [
    {"file": "main.cu", "kernel": "k1", "line": 37},
    {"file": "main.cu", "kernel": "k2", "line": 66},
    {"file": "main.cu", "kernel": "k3", "line": 131},
]


EXPECTED_FUNCTION_DEFINITIONS = {
    "main.cu": """void CPU (defnt)
__global__ void k1 (defnt)
__global__ void k2 (defnt)
__global__ void k3 (defnt)
int main (defnt)""",
}

EXPECTED_FUNCTION_DECLARATIONS = {}
