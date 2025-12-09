solution = [
 r"""template<typename T, int N>
__global__ void addBiasResidualPostLayerNorm(
          T* out, 
    const T* __restrict__ input,
    const T* __restrict__ bias,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    const float layernorm_eps,
    const int n)
{
  __shared__ float s_mean;
  __shared__ float s_variance;
  float            mean     = 0.0f;
  float            variance = 0.0f;
  float            local_out_cache[N];

#pragma unroll N
  for (int idx = threadIdx.x, i = 0; idx < n && i < N; ++i) {
    float local_out = typeToFloat(add(out[blockIdx.x * n + idx], input[blockIdx.x * n + idx], bias[idx]));
    mean += local_out;
    // save local_out to local_out_cache to save some recompute
    local_out_cache[i] = local_out;
    idx += blockDim.x;
  }

  mean = blockReduceSum<float>(mean);
  if (threadIdx.x == 0) {
    s_mean = mean / n;
  }
  __syncthreads();

#pragma unroll N
  for (int idx = threadIdx.x, i = 0; idx < n && i < N; ++i) {
    float local_out = local_out_cache[i];
    variance += (local_out - s_mean) * (local_out - s_mean);
    idx += blockDim.x;
  }
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = variance / n + layernorm_eps;
  }
  __syncthreads();

#pragma unroll N
  for (int idx = threadIdx.x, i = 0; idx < n && i < N; ++i) {
    float local_out = local_out_cache[i];
    out[blockIdx.x * n + idx] =
      floatToType<T>(((local_out - s_mean) * rsqrtf(s_variance)) * typeToFloat(gamma[idx]) + typeToFloat(beta[idx]));
    idx += blockDim.x;
  }
}
""".strip()
]
