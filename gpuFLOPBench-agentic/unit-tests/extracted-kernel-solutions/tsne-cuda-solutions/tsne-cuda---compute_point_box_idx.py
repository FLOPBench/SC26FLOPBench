solution = [
 r"""__global__
void compute_point_box_idx(
    volatile int*   __restrict__ point_box_idx,
    volatile float* __restrict__ x_in_box,
    volatile float* __restrict__ y_in_box,
    const float* const xs,
    const float* const ys,
    const float* const box_lower_bounds,
    const float min_coord,
    const float box_width,
    const int n_boxes,
    const int n_total_boxes,
    const int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N)
        return;

    int x_idx = (int)((xs[tid] - min_coord) / box_width);
    int y_idx = (int)((ys[tid] - min_coord) / box_width);

    x_idx = max(0, x_idx);
    x_idx = min((int)(n_boxes - 1), x_idx);

    y_idx = max(0, y_idx);
    y_idx = min((int)(n_boxes - 1), y_idx);

    int box_idx = y_idx * n_boxes + x_idx;
    point_box_idx[tid] = box_idx;

    x_in_box[tid] = (xs[tid] - box_lower_bounds[box_idx])                 / box_width;
    y_in_box[tid] = (ys[tid] - box_lower_bounds[n_total_boxes + box_idx]) / box_width;
}
""".strip()
]
