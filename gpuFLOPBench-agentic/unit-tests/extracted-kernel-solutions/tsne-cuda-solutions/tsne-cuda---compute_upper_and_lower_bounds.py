solution = [
 r"""__global__
void compute_upper_and_lower_bounds(
    volatile float* __restrict__ box_upper_bounds,
    volatile float* __restrict__ box_lower_bounds,
    const    float box_width,
    const    float x_min,
    const    float y_min,
    const    int   n_boxes,
    const    int   n_total_boxes)
{
    int tid, i, j;
    tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_boxes * n_boxes)
        return;

    i = tid / n_boxes;
    j = tid % n_boxes;

    box_lower_bounds[i * n_boxes + j] =  j      * box_width + x_min;
    box_upper_bounds[i * n_boxes + j] = (j + 1) * box_width + x_min;

    box_lower_bounds[n_total_boxes + i * n_boxes + j] =  i      * box_width + y_min;
    box_upper_bounds[n_total_boxes + i * n_boxes + j] = (i + 1) * box_width + y_min;
}
""".strip()
]
