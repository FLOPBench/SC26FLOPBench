solution = [
 r"""__global__
void RowSumKernel(
    volatile float* __restrict__ row_sum,
    const    float* __restrict__ pij,
    const unsigned int num_points,
    const unsigned int num_neighbors)
{
    int tid;
    tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_points) {
        return;
    }

    float temp_sum = 0.0f;
    for (int j = 0; j < num_neighbors; ++j) {
        temp_sum += pij[tid * num_neighbors + j];
    }
    row_sum[tid] = temp_sum;
}
""".strip()
]
