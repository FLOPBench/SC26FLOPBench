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
    "kernel.h": """template <typename T, typename std::enable_if<std::is_same<T, half2>::value, int>::type = 0> void initialize(uint64_t nsize, T *__restrict__ A, float value) (defnt)
template <typename T, typename std::enable_if<!std::is_same<T, half2>::value, int>::type = 0> void initialize(uint64_t nsize, T *__restrict__ A, float value) (defnt)
template <typename T, typename std::enable_if<std::is_same<T, half2>::value, int>::type = 0> __global__ void block_stride(uint32_t ntrials, uint32_t nsize, T *__restrict__ A) (defnt)
template <typename T, typename std::enable_if<!std::is_same<T, half2>::value, int>::type = 0> __global__ void block_stride(uint32_t ntrials, uint32_t nsize, T *__restrict__ A) (defnt)
template <typename T> void gpuKernel(uint32_t nsize, uint32_t ntrials, T *__restrict__ A, int *bytes_per_elem, int *mem_accesses_per_elem) (defnt)""",
    "main.cu": """double getTime() (defnt)
template <typename T> inline void launchKernel(uint64_t n, uint64_t t, T *buf, T *d_buf, int *bytes_per_elem_ptr, int *mem_accesses_per_elem_ptr) (defnt)
template <typename T> void run(uint64_t PSIZE, T *buf) (defnt)
int main(int argc, char *argv[]) (defnt)""",
}

EXPECTED_FUNCTION_DECLARATIONS = {}

