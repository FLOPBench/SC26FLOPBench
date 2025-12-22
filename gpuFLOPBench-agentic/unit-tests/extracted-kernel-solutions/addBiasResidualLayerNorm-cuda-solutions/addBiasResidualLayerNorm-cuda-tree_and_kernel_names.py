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
    "kernels.h": """template <> inline __host__ __device__ float typeToFloat(__nv_bfloat16 a) (defnt)
template <> inline __host__ __device__ float typeToFloat(half a) (defnt)
template <> inline __host__ __device__ half floatToType(float a) (defnt)
template <> inline __host__ __device__ __nv_bfloat16 floatToType(float a) (defnt)
template <> inline __device__ half2 floatToType2(float a) (defnt)
template <> inline __device__ __nv_bfloat162 floatToType2(float a) (defnt)
template <typename T> inline __device__ T float2ToType2(float2 a) (defnt)
template <> inline __device__ half2 float2ToType2(float2 a) (defnt)
template <> inline __device__ __nv_bfloat162 float2ToType2(float2 a) (defnt)
template <typename T> inline __device__ float2 type2ToFloat2(T a) (defnt)
template <> inline __device__ float2 type2ToFloat2(half2 a) (defnt)
template <> inline __device__ float2 type2ToFloat2(__nv_bfloat162 a) (defnt)
template <typename T> inline __device__ T add(T a, T b) (defnt)
template <> inline __device__ half add(half a, half b) (defnt)
template <> inline __device__ half2 add(half2 a, half2 b) (defnt)
template <> inline __device__ __nv_bfloat16 add(__nv_bfloat16 a, __nv_bfloat16 b) (defnt)
template <> inline __device__ __nv_bfloat162 add(__nv_bfloat162 a, __nv_bfloat162 b) (defnt)
template <typename T> inline __device__ T add(T a, T b, T c) (defnt)
template <> inline __device__ __nv_bfloat162 add(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c) (defnt)
template <> inline __device__ __nv_bfloat16 add(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) (defnt)
template <typename T> inline __device__ T sub(T a, T b) (defnt)
template <> inline __device__ half2 sub(half2 a, half2 b) (defnt)
template <> inline __device__ __nv_bfloat162 sub(__nv_bfloat162 a, __nv_bfloat162 b) (defnt)
template <typename T> inline __device__ T fma(T a, T b, T c, T d) (defnt)
template <> inline __device__ half2 fma(half2 a, half2 b, half2 c, half2 d) (defnt)
template <> inline __device__ __nv_bfloat162 fma(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c, __nv_bfloat162 d) (defnt)
template <typename T> __device__ T warpReduceSum(T val) (defnt)
template <typename T> __device__ T blockReduceSum(T val) (defnt)
template <typename T> __global__ void addBiasResidualPostLayerNormV2( T* out, const T* __restrict__ input, const T* __restrict__ bias, const T* __restrict__ gamma, const T* __restrict__ beta, const float layernorm_eps, const int n) (defnt)
template <typename T, int N> __global__ void addBiasResidualPostLayerNorm( T* out, const T* __restrict__ input, const T* __restrict__ bias, const T* __restrict__ gamma, const T* __restrict__ beta, const float layernorm_eps, const int n) (defnt)
template <typename T> __global__ void generalAddBiasResidualPostLayerNorm( T* out, const T* __restrict__ input, const T* __restrict__ bias, const T* __restrict__ gamma, const T* __restrict__ beta, const float layernorm_eps, const int n) (defnt)""",
    "main.cu": """template <typename T, int V> void invokeAddBiasResidualLayerNorm( T* out, const T* input, const T* bias, const T* gamma, const T* beta, const float layernorm_eps, int m, int n) (defnt)
template <typename T, int V> void layer(int repeat) (defnt)
int main(int argc, char* argv[]) (defnt)""",
}

EXPECTED_FUNCTION_DECLARATIONS = {}

