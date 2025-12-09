solution = [
 r"""__global__ void ComputePijxQijKernelV3(
          float* __restrict__ workspace_x,
          float* __restrict__ workspace_y,
    const float* __restrict__ pij,
    const int*   __restrict__ pij_ind,
    const float* __restrict__ points,
    const int num_points,
    const int num_neighbors)
{
    int tid, i, j;
    float ix, iy, jx, jy, dx, dy, pijqij;
    tid = threadIdx.x + blockIdx.x * blockDim.x; // This is the location in the pij matrix
    if (tid >= num_points * num_neighbors)
        return;

    i = tid / num_neighbors;
    j = pij_ind[tid];

    ix = points[i];
    iy = points[num_points + i];
    jx = points[j];
    jy = points[num_points + j];
    dx = ix - jx; // X distance
    dy = iy - jy; // Y distance
    pijqij = pij[tid] / (1 + dx * dx + dy * dy);

    workspace_x[tid] = pijqij * dx;
    workspace_y[tid] = pijqij * dy;
}
""".strip()
]
