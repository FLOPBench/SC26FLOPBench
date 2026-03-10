solution = [
 r"""__global__ void compute_repulsive_forces_kernel(
    volatile float* __restrict__ repulsive_forces,
    volatile float* __restrict__ normalization_vec,
    const float* const xs,
    const float* const ys,
    const float* const potentialsQij,
    const int num_points,
    const int n_terms)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_points)
        return;

    float phi1, phi2, phi3, phi4, x_pt, y_pt;

    phi1 = potentialsQij[tid * n_terms + 0];
    phi2 = potentialsQij[tid * n_terms + 1];
    phi3 = potentialsQij[tid * n_terms + 2];
    phi4 = potentialsQij[tid * n_terms + 3];

    x_pt = xs[tid];
    y_pt = ys[tid];

    normalization_vec[tid] = (1 + x_pt * x_pt + y_pt * y_pt) * phi1 - 2 * (x_pt * phi2 + y_pt * phi3) + phi4;

    repulsive_forces[tid] = x_pt * phi1 - phi2;
    repulsive_forces[tid + num_points] = y_pt * phi1 - phi3;
}
""".strip()
]
