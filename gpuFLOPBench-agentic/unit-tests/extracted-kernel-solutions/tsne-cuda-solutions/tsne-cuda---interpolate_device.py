solution = [
 r"""__global__
void interpolate_device(
    volatile float* __restrict__ interpolated_values,
    const    float* const y_in_box,
    const    float* const y_tilde_spacings,
    const    float* const denominator,
    const int n_interpolation_points,
    const int N)
{
    int tid, i, j, k;
    float value, ybox_i;

    tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N * n_interpolation_points)
        return;

    i = tid % N;
    j = tid / N;

    value = 1;
    ybox_i = y_in_box[i];

    for (k = 0; k < n_interpolation_points; k++) {
        if (j != k) {
            value *= ybox_i - y_tilde_spacings[k];
        }
    }

    interpolated_values[j * N + i] = value / denominator[j];
}
""".strip()
]
