solution = [
 r"""__global__
void ComputePijKernel(
    volatile float* __restrict__ pij,
    const    float* __restrict__ squared_dist,
    const    float* __restrict__ betas,
    const unsigned int num_points,
    const unsigned int num_neighbors)
{
    int tid, i, j;
    float dist, beta;

    tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_points * num_neighbors)
        return;

    i = tid / num_neighbors;
    j = tid % num_neighbors;

    beta = betas[i];
    dist = squared_dist[tid];

    // condition deals with evaluation of pii
    // FAISS neighbor zero is i so ignore it
    pij[tid] = (j == 0 & dist == 0.0f) ? 0.0f : __expf(-beta * dist); //TODO: This probably never evaluates to true
}
""".strip()
]
