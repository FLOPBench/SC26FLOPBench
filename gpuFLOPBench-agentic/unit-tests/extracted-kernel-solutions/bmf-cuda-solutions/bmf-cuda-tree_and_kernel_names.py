EXPECTED_TREE = (
    "bmf-cuda/\n"
    "  src/\n"
    "    helper/\n"
    "      args_parser.h\n"
    "      clipp.h\n"
    "      confusion.h\n"
    "      cuda_helpers.cuh\n"
    "      rngpu.hpp\n"
    "    bit_vector_functions.h\n"
    "    bit_vector_kernels.cuh\n"
    "    config.h\n"
    "    cuBool_cpu.h\n"
    "    cuBool_gpu.cuh\n"
    "    float_kernels.cuh\n"
    "    io_and_allocation.hpp\n"
    "    main.cu\n"
    "    updates_and_measures.cuh\n"
    "  LICENSE\n"
    "  Makefile\n"
    "  README.md"
)

EXPECTED_MAIN_FILES = ["src/main.cu"]

EXPECTED_KERNELS = [
    {"file": "src/bit_vector_kernels.cuh", "kernel": "initFactor", "line": 31},
    {"file": "src/bit_vector_kernels.cuh", "kernel": "computeDistanceRows", "line": 59},
    {"file": "src/bit_vector_kernels.cuh", "kernel": "computeDistanceRowsShared", "line": 97},
    {"file": "src/bit_vector_kernels.cuh", "kernel": "vectorMatrixMultCompareRowWarpShared", "line": 158},
    {"file": "src/bit_vector_kernels.cuh", "kernel": "vectorMatrixMultCompareColWarpShared", "line": 236},
    {"file": "src/float_kernels.cuh", "kernel": "initFactor", "line": 5},
    {"file": "src/float_kernels.cuh", "kernel": "computeDistanceRowsShared", "line": 28},
]

EXPECTED_FUNCTION_DEFINITIONS = {
    "src/bit_vector_kernels.cuh": """template <typename T> __device__ T warpReduceSum (defnt)
template <typename T> __device__ T blockReduceSum (defnt)
template <typename bit_vector_t, typename index_t> __global__ void initFactor (defnt)
template <typename bit_factor_t, typename bit_matrix_t, typename index_t, typename error_t> __global__ void computeDistanceRows (defnt)
template <typename bit_factor_t, typename bit_matrix_t, typename index_t, typename error_t> __global__ void computeDistanceRowsShared (defnt)
template <typename bit_factor_t, typename bit_matrix_t, typename index_t, typename error_t> __global__ void vectorMatrixMultCompareRowWarpShared (defnt)
template <typename bit_factor_t, typename bit_matrix_t, typename index_t, typename error_t> __global__ void vectorMatrixMultCompareColWarpShared (defnt)""",
    "src/float_kernels.cuh": """__global__ void initFactor (defnt)
__global__ void computeDistanceRowsShared (defnt)""",
    "src/main.cu": "int main (defnt)",
}

EXPECTED_FUNCTION_DECLARATIONS = {}
