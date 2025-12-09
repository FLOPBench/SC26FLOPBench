solution = [
 r"""__global__
void PostprocessNeighborIndicesKernel(
    volatile int* __restrict__ pij_indices,
    const long*   __restrict__ knn_indices,
    const int num_points,
    const int num_neighbors)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_points * num_neighbors)
        return;
    pij_indices[tid] = (int)knn_indices[tid];
}
""".strip()
]
