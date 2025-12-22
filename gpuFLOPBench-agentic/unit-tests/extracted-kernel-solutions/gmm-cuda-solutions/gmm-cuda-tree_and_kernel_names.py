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
    "gaussian_kernel.cu": """__device__ void mvtmeans(float* fcs_data, int num_dimensions, int num_events, float* means) (defnt)
__device__ void averageVariance(float* fcs_data, float* means, int num_dimensions, int num_events, float* avgvar) (defnt)
__device__ void invert(float* data, int actualsize, float* log_determinant) (defnt)
__device__ void compute_pi(clusters_t* clusters, int num_clusters) (defnt)
__device__ void compute_constants(clusters_t* clusters, int num_clusters, int num_dimensions) (defnt)
__global__ void constants_kernel(clusters_t* clusters, int num_clusters, int num_dimensions) (defnt)
__global__ void seed_clusters_kernel( const float* fcs_data, clusters_t* clusters, const int num_dimensions, const int num_clusters, const int num_events) (defnt)
__device__ float parallelSum(float* data, const unsigned int ndata) (defnt)
__device__ void compute_indices(int num_events, int* start, int* stop) (defnt)
__global__ void estep1(float* data, clusters_t* clusters, int num_dimensions, int num_events) (defnt)
__global__ void estep2(float* fcs_data, clusters_t* clusters, int num_dimensions, int num_clusters, int num_events, float* likelihood) (defnt)
__global__ void mstep_means(float* fcs_data, clusters_t* clusters, int num_dimensions, int num_clusters, int num_events) (defnt)
__global__ void mstep_N(clusters_t* clusters, int num_dimensions, int num_clusters, int num_events) (defnt)
__device__ void compute_row_col(int n, int* row, int* col) (defnt)
__global__ void mstep_covariance1(float* fcs_data, clusters_t* clusters, int num_dimensions, int num_clusters, int num_events) (defnt)
__global__ void mstep_covariance2(float* fcs_data, clusters_t* clusters, int num_dimensions, int num_clusters, int num_events) (defnt)""",
    "main.cu": """int main( int argc, char** argv) (defnt)""",
}

EXPECTED_FUNCTION_DECLARATIONS = {}

