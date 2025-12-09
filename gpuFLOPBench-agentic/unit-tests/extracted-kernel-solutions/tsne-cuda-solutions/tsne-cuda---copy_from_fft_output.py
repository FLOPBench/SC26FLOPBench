solution = [
 r"""__global__
void copy_from_fft_output(
    volatile float* __restrict__ y_tilde_values,
    const    float* fft_output,
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

    i = current_loc / n_fft_coeffs_half + n_fft_coeffs_half;
    j = current_loc % n_fft_coeffs_half + n_fft_coeffs_half;

    y_tilde_values[current_term + n_terms * current_loc] = fft_output[current_term * (n_fft_coeffs * n_fft_coeffs) + i * n_fft_coeffs + j] / (float)(n_fft_coeffs * n_fft_coeffs);
}
""".strip()
]
