EXPECTED_TREE = (
    "ert-cuda/\n"
    "  Copyright\n"
    "  kernel.h\n"
    "  License\n"
    "  main.cu\n"
    "  Makefile\n"
    "  rep.h"
)

EXPECTED_MAIN_FILES = ["main.cu"]

EXPECTED_KERNELS = [
    {"file": "kernel.h", "kernel": "block_stride", "line": 39},
    {"file": "kernel.h", "kernel": "block_stride", "line": 85},
]

EXPECTED_FUNCTION_DEFINITIONS = {
    "kernel.h": """template <typename T, typename std::enable_if<std::is_same<T, half2>::value, int>::type = 0> void initialize (defnt)
template <typename T, typename std::enable_if<!std::is_same<T, half2>::value, int>::type = 0> void initialize (defnt)
template <typename T, typename std::enable_if<std::is_same<T, half2>::value, int>::type = 0> __global__ void block_stride (defnt)
template <typename T, typename std::enable_if<!std::is_same<T, half2>::value, int>::type = 0> __global__ void block_stride (defnt)
template <typename T> void gpuKernel (defnt)""",
    "main.cu": """double getTime (defnt)
template <typename T> void launchKernel (defnt)
template <typename T> void run (defnt)
int main (defnt)""",
}

EXPECTED_FUNCTION_DECLARATIONS = {}
