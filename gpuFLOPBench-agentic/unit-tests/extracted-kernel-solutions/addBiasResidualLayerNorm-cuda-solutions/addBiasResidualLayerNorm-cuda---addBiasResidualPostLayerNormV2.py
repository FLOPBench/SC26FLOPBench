solution = [
 r"""template<typename T>
__global__ void addBiasResidualPostLayerNormV2(
          T* out,
    const T* __restrict__ input,
    const T* __restrict__ bias,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    const float layernorm_eps,
    const int n)
{
  using T2             = typename TypeConverter<T>::Type;
  const int        ite = 4;
  const int        tid = threadIdx.x;
  const int        bid = blockIdx.x;
  __shared__ float s_mean;
  __shared__ float s_variance;
  float            mean     = 0.0f;
  float            variance = 0.0f;
  T2               local_out_half2[ite];

  T2*       out_ptr   = (T2*)out;
  const T2* input_ptr = (const T2*)input;
  const T2* bias_ptr  = (const T2*)bias;
  const T2* gamma_ptr = (const T2*)gamma;
  const T2* beta_ptr  = (const T2*)beta;

  T2 sum = floatToType2<T2>(0.0f);

  // ite = 4 and blockDim.x = n / 8
  // When n = 1024, blockDim.x = 128
  // col_id range: [0-127], [128-255], [256-383], [384-511]
  // block stride = n / 2 = 512
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id         = i * blockDim.x + tid;
    int id             = bid * n / 2 + col_id;
    local_out_half2[i] = add(out_ptr[id], input_ptr[id], bias_ptr[col_id]);
    sum                = add(sum, local_out_half2[i]);
  }

  mean = blockReduceSum<float>(typeToFloat(__hadd(sum.x , sum.y)));
  if (threadIdx.x == 0) {
    s_mean = mean / n;
  }
  __syncthreads();

  float var      = 0.0f;
  T2    s_mean_2 = floatToType2<T2>(s_mean);

#pragma unroll
  for (int i = 0; i < ite; i++) {
    local_out_half2[i] = sub(local_out_half2[i], s_mean_2);
    float v1           = typeToFloat(local_out_half2[i].x);
    float v2           = typeToFloat(local_out_half2[i].y);
    var += v1 * v1 + v2 * v2;
  }

  variance = blockReduceSum<float>(var);
  if (tid == 0) {
    s_variance = rsqrtf(variance / n + layernorm_eps);
  }
  __syncthreads();

  T2 s_var_2 = floatToType2<T2>(s_variance);
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id  = i * blockDim.x + tid;
    int id      = bid * n / 2 + col_id;
    out_ptr[id] = fma(local_out_half2[i], s_var_2,
                      gamma_ptr[col_id], beta_ptr[col_id]);
  }
}
""".strip()
]
