solution = [
 r"""__global__ void reduce_sum_kernel(
          float* __restrict__ attractive_forces,
    const float* __restrict__ workspace_x,
    const float* __restrict__ workspace_y,
    const int num_points,
    const int num_neighbors)
{
    int tid, jend, j;
    float acc_x, acc_y;
    tid = threadIdx.x + blockIdx.x * blockDim.x; // This is the location in the pij matrix
    if (tid >= num_points)
        return;

    acc_x = 0.0f;
    acc_y = 0.0f;
    jend = (tid + 1) * num_neighbors;
    for (j = tid * num_neighbors; j < jend; j++)
    {
        acc_x += workspace_x[j];
        acc_y += workspace_y[j];
    }

    attractive_forces[tid] = acc_x;
    attractive_forces[num_points + tid] = acc_y;
}
""".strip()
]
