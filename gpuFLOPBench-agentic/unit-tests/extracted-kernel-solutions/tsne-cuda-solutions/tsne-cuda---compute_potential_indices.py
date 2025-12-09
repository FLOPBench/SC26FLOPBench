solution = [
 r"""__global__
void compute_potential_indices(
          float* __restrict__ potentialsQij,
    const int*   const point_box_indices,
    const float* const y_tilde_values,
    const float* const x_interpolated_values,
    const float* const y_interpolated_values,
    const int N,
    const int n_interpolation_points,
    const int n_boxes,
    const int n_terms)
{
    int tid, current_term, i, interp_i, interp_j, box_idx, box_i, box_j, idx;
    tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_terms * n_interpolation_points * n_interpolation_points * N)
        return;

    current_term = tid % n_terms;
    i = (tid / n_terms) % N;
    interp_j = ((tid / n_terms) / N) % n_interpolation_points;
    interp_i = ((tid / n_terms) / N) / n_interpolation_points;

    box_idx = point_box_indices[i];
    box_i = box_idx % n_boxes;
    box_j = box_idx / n_boxes;

    idx = (box_i * n_interpolation_points + interp_i) * (n_boxes * n_interpolation_points) +
          (box_j * n_interpolation_points) + interp_j;

    atomicAdd(
        potentialsQij + i * n_terms + current_term,
        x_interpolated_values[i + interp_i * N] * y_interpolated_values[i + interp_j * N] * y_tilde_values[idx * n_terms + current_term]);
}
""".strip()
]
