solution = [
 r"""template<typename BinaryFunction, typename T>
__global__ void tsne::utils::BroadcastRowVector(
          T* __restrict__ d_matrix,
    const T* __restrict__ d_vector,
    const int N,
    const int M,
    BinaryFunction binary_operation,
    const T alpha)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int i = tid % N;
    const int j = tid / N;
    if (j < M) {
        d_matrix[j * N + i] = binary_operation(d_matrix[j * N + i], alpha * d_vector[j]);
    }
}
""".strip()
]
