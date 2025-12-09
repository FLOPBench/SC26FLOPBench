solution = [
 r"""__global__
void iDFT2D1gpu(thrust::complex<float>* din, thrust::complex<float>* dout, int num_rows, int num_cols)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= num_rows || i >= num_cols) {
        return;
    }
    
    float angle, cosf, sinf; 
    thrust::complex<float> sum, twiddle;
    angle = 2.0f * PI * fdividef((float)i, (float)num_cols);
    sum = 0.0f;
    for (int k = 0; k < num_cols/2+1; ++k) {
        // sincosf(angle * k, &sinf, &cosf);
        // twiddle = thrust::complex<float>(cosf, sinf);
        TWIDDLE();
        sum += din[j * (num_cols/2+1) + k] * twiddle;
    }
    for (int k = num_cols/2+1; k < num_cols; ++k) {
        TWIDDLE();
        sum += thrust::conj(din[((num_rows-j)%num_rows) * (num_cols/2+1) + ((num_cols-k)%num_cols)]) * twiddle;
    }

    dout[i * num_rows + j] = sum;
}
""".strip()
]
