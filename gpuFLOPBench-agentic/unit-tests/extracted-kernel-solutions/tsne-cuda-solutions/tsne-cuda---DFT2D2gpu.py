solution = [
 r"""__global__
void DFT2D2gpu(thrust::complex<float>* din, thrust::complex<float>* dout, int num_rows, int num_cols)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= num_rows || i >= num_cols) {
        return;
    }
    
    float angle, cosf, sinf;
    thrust::complex<float> sum, twiddle;
    angle = -2.0f * PI * fdividef((float)i, (float)num_cols);
    sum = 0.0f;
    for (int k = 0; k < num_cols; ++k) {
        // sincosf(angle * k, &sinf, &cosf);
        // twiddle = thrust::complex<float>(cosf, sinf);
        TWIDDLE();
        sum = sum + din[j * num_cols + k] * twiddle;
    }

    dout[i * num_rows + j] = sum;
}
""".strip()
]
