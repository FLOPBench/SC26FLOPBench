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

EXPECTED_INCLUDE_TREES = {
    "main.cu": """main.cu
  #include <stdio.h> (DNE)
  #include <stdlib.h> (DNE)
  #include <string.h> (DNE)
  #include <limits.h> (DNE)
  #include <math.h> (DNE)
  #include <unistd.h> (DNE)
  #include <fcntl.h> (DNE)
  #include <float.h> (DNE)
  #include <time.h> (DNE)
  #include <sys/time.h> (DNE)
  #include <iostream> (DNE)
  #include <cuda.h> (DNE)
  #include "kernel_find_index.h"
  #include "kernel_likelihood.h"
  #include "kernel_normalize_weights.h"
  #include "kernel_sum.h"

""",
}

EXPECTED_KERNELS = [
    {"file": "kernel_find_index.h", "kernel": "kernel_find_index", "line": 1},
    {"file": "kernel_likelihood.h", "kernel": "kernel_likelihood", "line": 1},
    {"file": "kernel_normalize_weights.h", "kernel": "kernel_normalize_weights", "line": 1},
    {"file": "kernel_sum.h", "kernel": "kernel_sum", "line": 1},
]

EXPECTED_FUNCTION_DEFINITIONS = {
    "kernel_find_index.h": """__global__ void kernel_find_index ( const float*__restrict__ arrayX, const float*__restrict__ arrayY, const float*__restrict__ CDF, const float*__restrict__ u, float*__restrict__ xj, float*__restrict__ yj, const int Nparticles) (defnt)""",
    "kernel_likelihood.h": """__global__ void kernel_likelihood ( float*__restrict__ arrayX, float*__restrict__ arrayY, const float*__restrict__ xj, const float*__restrict__ yj, int*__restrict__ ind, const int*__restrict__ objxy, float*__restrict__ likelihood, const unsigned char*__restrict__ I, float*__restrict__ weights, int*__restrict__ seed, float*__restrict__ partial_sums, const int Nparticles, const int countOnes, const int IszY, const int Nfr, const int k, const int max_size) (defnt)""",
    "kernel_normalize_weights.h": """__global__ void kernel_normalize_weights ( float* __restrict__ weights, const float* __restrict__ partial_sums, float* __restrict__ CDF, float* __restrict__ u, int* __restrict__ seed, const int Nparticles ) (defnt)""",
    "kernel_sum.h": """__global__ void kernel_sum (float* partial_sums, const int Nparticles) (defnt)""",
    "main.cu": """long long get_time() (defnt)
float elapsed_time(long long start_time, long long end_time) (defnt)
float randu(int * seed, int index) (defnt)
float randn(int * seed, int index) (defnt)
float roundFloat(float value) (defnt)
void setIf(int testValue, int newValue, unsigned char * array3D, int * dimX, int * dimY, int * dimZ) (defnt)
void addNoise(unsigned char * array3D, int * dimX, int * dimY, int * dimZ, int * seed) (defnt)
void strelDisk(int * disk, int radius) (defnt)
void dilate_matrix(unsigned char * matrix, int posX, int posY, int posZ, int dimX, int dimY, int dimZ, int error) (defnt)
void imdilate_disk(unsigned char * matrix, int dimX, int dimY, int dimZ, int error, unsigned char * newMatrix) (defnt)
void getneighbors(int * se, int numOnes, int * neighbors, int radius) (defnt)
void videoSequence(unsigned char * I, int IszX, int IszY, int Nfr, int * seed) (defnt)
int findIndex(float * CDF, int lengthCDF, float value) (defnt)
int particleFilter(unsigned char * I, int IszX, int IszY, int Nfr, int * seed, int Nparticles) (defnt)
int main(int argc, char * argv[]) (defnt)""",
}

EXPECTED_FUNCTION_DECLARATIONS = {}

