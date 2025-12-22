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

EXPECTED_FUNCTION_DEFINITIONS = {
    "kernel_find_index.h": "__global__ void kernel_find_index (defnt)",
    "kernel_likelihood.h": "__global__ void kernel_likelihood (defnt)",
    "kernel_normalize_weights.h": "__global__ void kernel_normalize_weights (defnt)",
    "kernel_sum.h": "__global__ void kernel_sum (defnt)",
    "main.cu": """long long get_time (defnt)
float elapsed_time (defnt)
float randu (defnt)
float randn (defnt)
float roundFloat (defnt)
void setIf (defnt)
void addNoise (defnt)
void strelDisk (defnt)
void dilate_matrix (defnt)
void imdilate_disk (defnt)
void getneighbors (defnt)
void videoSequence (defnt)
int findIndex (defnt)
int particleFilter (defnt)
int main (defnt)""",
}

EXPECTED_FUNCTION_DECLARATIONS = {}
