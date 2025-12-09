solution = [
 r"""__global__
void compute_kernel_tilde(
    volatile float* __restrict__ kernel_tilde,   // 780 x 780
    const    float x_min,
    const    float y_min,
    const    float h,
    const    int   n_interpolation_points_1d,    // 390
    const    int   n_fft_coeffs)                 // 390 x 2 = 780
{
    int tid, i, j;
    float tmp;
    tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_interpolation_points_1d * n_interpolation_points_1d)
        return;

    i = tid / n_interpolation_points_1d;
    j = tid % n_interpolation_points_1d;

    // TODO: Possibly issuing a memory pre-fetch here could help the code.
    tmp = squared_cauchy_2d(y_min + h / 2, x_min + h / 2, y_min + h / 2 + i * h, x_min + h / 2 + j * h);
    kernel_tilde[(n_interpolation_points_1d + i) * n_fft_coeffs + (n_interpolation_points_1d + j)] = tmp;
    kernel_tilde[(n_interpolation_points_1d - i) * n_fft_coeffs + (n_interpolation_points_1d + j)] = tmp;
    kernel_tilde[(n_interpolation_points_1d + i) * n_fft_coeffs + (n_interpolation_points_1d - j)] = tmp;
    kernel_tilde[(n_interpolation_points_1d - i) * n_fft_coeffs + (n_interpolation_points_1d - j)] = tmp;
}
""".strip()
]
