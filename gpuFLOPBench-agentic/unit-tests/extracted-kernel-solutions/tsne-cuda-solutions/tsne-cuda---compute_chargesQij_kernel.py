solution = [
 r"""__global__ void compute_chargesQij_kernel(
    volatile float* __restrict__ chargesQij,
    const float* const xs,
    const float* const ys,
    const int num_points,
    const int n_terms)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_points)
        return;

    float x_pt, y_pt;
    x_pt = xs[tid];
    y_pt = ys[tid];

    chargesQij[tid * n_terms + 0] = 1;
    chargesQij[tid * n_terms + 1] = x_pt;
    chargesQij[tid * n_terms + 2] = y_pt;
    chargesQij[tid * n_terms + 3] = x_pt * x_pt + y_pt * y_pt;
}
""".strip()
]
