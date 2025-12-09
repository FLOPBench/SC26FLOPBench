solution = [
 r"""__global__
void copy_to_fft_input(
    volatile float* __restrict__ fft_input,
    const    float* w_coefficients_device,
    const int n_fft_coeffs,
    const int n_fft_coeffs_half,
    const int n_terms)
{
    int i, j;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_terms * n_fft_coeffs_half * n_fft_coeffs_half)
        return;

    int current_term = tid / (n_fft_coeffs_half * n_fft_coeffs_half);
    int current_loc  = tid % (n_fft_coeffs_half * n_fft_coeffs_half);

    i = current_loc / n_fft_coeffs_half;
    j = current_loc % n_fft_coeffs_half;

    fft_input[current_term * (n_fft_coeffs * n_fft_coeffs) + i * n_fft_coeffs + j] = w_coefficients_device[current_term + current_loc * n_terms];
}
""".strip()
]
