solution = [
 r"""__global__
void syv2k(
          float* __restrict__ pij_sym,
    const float* __restrict__ pij_non_sym,
    const int*   __restrict__ pij_indices,
    const int num_points,
    const int num_neighbors)
{
    int tid, i, j, jend;
    float pij_acc;

    tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_points * num_neighbors) {
        return;
    }

    i = tid / num_neighbors;
    j = pij_indices[tid];

    pij_acc = pij_non_sym[tid];
    jend = (j + 1) * num_neighbors;
    for (int jidx = j * num_neighbors; jidx < jend; jidx++) {
        pij_acc += pij_indices[jidx] == i ? pij_non_sym[jidx] : 0.0f;
    }
    pij_sym[tid] = pij_acc / (2.0f * num_points);
}
""".strip()
]
