EXPECTED_TREE = (
    "addBiasResidualLayerNorm-cuda/\n"
    "  kernels.h\n"
    "  main.cu\n"
    "  Makefile"
)

EXPECTED_MAIN_FILES = ["main.cu"]

EXPECTED_KERNELS = [
    {"file": "kernels.h", "kernel": "addBiasResidualPostLayerNormV2", "line": 202},
    {"file": "kernels.h", "kernel": "addBiasResidualPostLayerNorm", "line": 275},
    {
        "file": "kernels.h",
        "kernel": "generalAddBiasResidualPostLayerNorm",
        "line": 327,
    },
]

EXPECTED_FUNCTION_DEFINITIONS = {
    "kernels.h": """template <> __device__ float typeToFloat (defnt)
template <> __device__ float typeToFloat (defnt)
template <> __device__ half floatToType (defnt)
template <> __device__ __nv_bfloat16 floatToType (defnt)
template <> __device__ half2 floatToType2 (defnt)
template <> __device__ __nv_bfloat162 floatToType2 (defnt)
template <typename T> __device__ T float2ToType2 (defnt)
template <> __device__ half2 float2ToType2 (defnt)
template <> __device__ __nv_bfloat162 float2ToType2 (defnt)
template <typename T> __device__ float2 type2ToFloat2 (defnt)
template <> __device__ float2 type2ToFloat2 (defnt)
template <> __device__ float2 type2ToFloat2 (defnt)
template <typename T> __device__ T add (defnt)
template <> __device__ half add (defnt)
template <> __device__ half2 add (defnt)
template <> __device__ __nv_bfloat16 add (defnt)
template <> __device__ __nv_bfloat162 add (defnt)
template <typename T> __device__ T add (defnt)
template <> __device__ __nv_bfloat162 add (defnt)
template <> __device__ __nv_bfloat16 add (defnt)
template <typename T> __device__ T sub (defnt)
template <> __device__ half2 sub (defnt)
template <> __device__ __nv_bfloat162 sub (defnt)
template <typename T> __device__ T fma (defnt)
template <> __device__ half2 fma (defnt)
template <> __device__ __nv_bfloat162 fma (defnt)
template <typename T> __device__ T warpReduceSum (defnt)
template <typename T> __device__ T blockReduceSum (defnt)
template <typename T> __global__ void addBiasResidualPostLayerNormV2 (defnt)
template <typename T, int N> __global__ void addBiasResidualPostLayerNorm (defnt)
template <typename T> __global__ void generalAddBiasResidualPostLayerNorm (defnt)""",
    "main.cu": """template <typename T, int V> void invokeAddBiasResidualLayerNorm (defnt)
template <typename T, int V> void layer (defnt)
int main (defnt)""",
}

EXPECTED_FUNCTION_DECLARATIONS = {}
