solution = [
 r"""template<typename T>
__global__ void generalAddBiasResidualPostLayerNorm(
          T* out, 
    const T* __restrict__ input,
    const T* __restrict__ bias,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    const float layernorm_eps,
    const int n)
{
  using T2 = typename TypeConverter<T>::Type;
  __shared__ float s_mean;
  __shared__ float s_variance;
  float            mean     = 0.0f;
  float            variance = 0.0f;

  T2*       out_ptr   = (T2*)out;
  const T2* input_ptr = (const T2*)input;
  const T2* bias_ptr  = (const T2*)bias;
  const T2* gamma_ptr = (const T2*)gamma;
  const T2* beta_ptr  = (const T2*)beta;

  float local_out = 0.0f;
  for (int idx = threadIdx.x; idx < n / 2; idx += blockDim.x) {
    int    id            = blockIdx.x * n / 2 + idx;
    T2     tmp           = add(add(out_ptr[id], input_ptr[id]), bias_ptr[idx]);
    float2 local_out_fp2 = type2ToFloat2(tmp);
    local_out += local_out_fp2.x;
    local_out += local_out_fp2.y;
    // save tmp to out_ptr to save some recomputation
    out_ptr[id] = tmp;
  }

  mean = blockReduceSum<float>(local_out);
  if (threadIdx.x == 0) {
    s_mean = mean / n;
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < n / 2; idx += blockDim.x) {
    int    id            = blockIdx.x * n / 2 + idx;
    float2 local_out_fp2 = type2ToFloat2(out_ptr[id]);
    variance += (local_out_fp2.x - s_mean) * (local_out_fp2.x - s_mean);
    variance += (local_out_fp2.y - s_mean) * (local_out_fp2.y - s_mean);
  }

  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / n + layernorm_eps);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < n / 2; idx += blockDim.x) {
    int    id            = blockIdx.x * n / 2 + idx;
    float2 local_out_fp2 = type2ToFloat2(out_ptr[id]);
    float2 gamma_val     = type2ToFloat2(gamma_ptr[idx]);
    float2 beta_val      = type2ToFloat2(beta_ptr[idx]);
    local_out_fp2.x      = (local_out_fp2.x - s_mean) * s_variance * gamma_val.x + beta_val.x;
    local_out_fp2.y      = (local_out_fp2.y - s_mean) * s_variance * gamma_val.y + beta_val.y;
    out_ptr[id]          = float2ToType2<T2>(local_out_fp2);
  }
}
""".strip()
]
