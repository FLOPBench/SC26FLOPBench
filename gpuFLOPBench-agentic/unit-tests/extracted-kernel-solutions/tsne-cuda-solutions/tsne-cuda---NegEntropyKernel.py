solution = [
 r"""__global__
void NegEntropyKernel(
    volatile float* __restrict__ neg_entropy,
    const    float* __restrict__ pij,
    const unsigned int num_points,
    const unsigned int num_neighbors)
{
    int tid;
    tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_points) {
        return;
    }

    float temp_sum = 0.0f;
    for (int j = 0; j < num_neighbors; ++j) {
        float x = pij[tid * num_neighbors + j];
        temp_sum += (x == 0.0f ? 0.0f : x * __logf(x));
    }
    neg_entropy[tid] = -1.0f * temp_sum;
}
""".strip()
]
