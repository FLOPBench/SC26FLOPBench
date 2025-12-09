solution = [
 r"""template<typename BinaryFunction, typename T>
__global__ void tsne::utils::BroadcastColumnVector(
          T* __restrict__ d_matrix,     // 4 x 780 x (780 / 2 + 1) = 4 x 304980 = 1219920
    const T* __restrict__ d_vector,     // 780 x 780 = 608400
    const int N,                        // 780 x (780 / 2 + 1) = 304980
    const int M,                        // 4
    BinaryFunction binary_operation,
    const T alpha)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int i = tid % N;
    const int j = tid / N;

    if (j < M) {    // condition makes sure tid < size of d_matrix
        d_matrix[j * N + i] = binary_operation(d_matrix[j * N + i], alpha * d_vector[i]);
    }
}
""".strip()
]
