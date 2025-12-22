EXPECTED_TREE = (
    "gmm-cuda/\n"
    "  cluster.cu\n"
    "  data.tar.gz\n"
    "  gaussian.h\n"
    "  gaussian_kernel.cu\n"
    "  LICENSE\n"
    "  main.cu\n"
    "  Makefile\n"
    "  readData.cu\n"
    "  README.txt"
)

EXPECTED_MAIN_FILES = ["main.cu"]

EXPECTED_KERNELS = [
    {"file": "gaussian_kernel.cu", "kernel": "constants_kernel", "line": 210},
    {"file": "gaussian_kernel.cu", "kernel": "seed_clusters_kernel", "line": 294},
    {"file": "gaussian_kernel.cu", "kernel": "estep1", "line": 420},
    {"file": "gaussian_kernel.cu", "kernel": "estep2", "line": 483},
    {"file": "gaussian_kernel.cu", "kernel": "mstep_means", "line": 559},
    {"file": "gaussian_kernel.cu", "kernel": "mstep_N", "line": 596},
    {"file": "gaussian_kernel.cu", "kernel": "mstep_covariance1", "line": 664},
    {"file": "gaussian_kernel.cu", "kernel": "mstep_covariance2", "line": 728},
]

EXPECTED_FUNCTION_DEFINITIONS = {
    "gaussian_kernel.cu": """__device__ void mvtmeans (defnt)
__device__ void averageVariance (defnt)
__device__ void invert (defnt)
__device__ void compute_pi (defnt)
__device__ void compute_constants (defnt)
__global__ void constants_kernel (defnt)
__global__ void seed_clusters_kernel (defnt)
__device__ float parallelSum (defnt)
__device__ void compute_indices (defnt)
__global__ void estep1 (defnt)
__global__ void estep2 (defnt)
__global__ void mstep_means (defnt)
__global__ void mstep_N (defnt)
__device__ void compute_row_col (defnt)
__global__ void mstep_covariance1 (defnt)
__global__ void mstep_covariance2 (defnt)""",
    "main.cu": "int main (defnt)",
}

EXPECTED_FUNCTION_DECLARATIONS = {}
