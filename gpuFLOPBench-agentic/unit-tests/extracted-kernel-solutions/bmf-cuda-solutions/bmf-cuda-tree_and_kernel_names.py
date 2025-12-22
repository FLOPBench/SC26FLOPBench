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
    "src/bit_vector_kernels.cuh": """template <typename T> __device__ T warpReduceSum(T val, const unsigned width = warpSize) (defnt)
template <typename T> __device__ T blockReduceSum(T val, T* reductionArray) (defnt)
template <typename bit_vector_t, typename index_t> __global__ void initFactor( bit_vector_t * Ab, const index_t height, const uint8_t factorDim, const uint32_t seed, const float threshold) (defnt)
template <typename bit_factor_t, typename bit_matrix_t, typename index_t, typename error_t> __global__ void computeDistanceRows( const bit_factor_t * __restrict__ Ab, const bit_factor_t * __restrict__ Bb, const bit_matrix_t * __restrict__ Cb, const index_t height, const index_t width, const index_t padded_width, const uint8_t factorDim, const int weight, error_t *global_error) (defnt)
template <typename bit_factor_t, typename bit_matrix_t, typename index_t, typename error_t> __global__ void computeDistanceRowsShared( const bit_factor_t * __restrict__ Ab, const bit_factor_t * __restrict__ Bb, const bit_matrix_t * __restrict__ Cb, const index_t height, const index_t width, const index_t padded_width, const uint8_t factorDim, const error_t weight, error_t *global_error) (defnt)
template <typename bit_factor_t, typename bit_matrix_t, typename index_t, typename error_t> __global__ void vectorMatrixMultCompareRowWarpShared( bit_factor_t * __restrict__ A, const bit_factor_t * __restrict__ B, const bit_matrix_t * __restrict__ C, const index_t height, const index_t width, const index_t padded_width, const uint8_t factorDim, const index_t startrow, error_t *global_error, const uint32_t seed, const float temperature, const float flipManyChance, const uint32_t flipManyDepth, const error_t weight) (defnt)
template <typename bit_factor_t, typename bit_matrix_t, typename index_t, typename error_t> __global__ void vectorMatrixMultCompareColWarpShared( const bit_factor_t * __restrict__ A, bit_factor_t * __restrict__ B, const bit_matrix_t * __restrict__ C, const index_t height, const index_t width, const index_t padded_width, const uint8_t factorDim, const index_t startcol, error_t *global_error, const uint32_t seed, const float temperature, const float flipManyChance, const uint32_t flipManyDepth, const error_t weight) (defnt)""",
    "src/float_kernels.cuh": """__global__ void initFactor( float * A, const int height, const uint8_t factorDim, const uint32_t seed, const float threshold) (defnt)
__global__ void computeDistanceRowsShared( const float * __restrict__ A, const float * __restrict__ B, const uint32_t * __restrict__ Cb, const int height, const int width, const int padded_width, const uint8_t factorDim, const int inverse_density, int *__restrict__ global_error) (defnt)""",
    "src/main.cu": """int main(int argc, char **argv) (defnt)""",
}

EXPECTED_FUNCTION_DECLARATIONS = {}

