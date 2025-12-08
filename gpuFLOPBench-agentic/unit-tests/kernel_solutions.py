from __future__ import annotations

KERNEL_SOURCE_SOLUTIONS: dict[tuple[str, str], dict[str, str | int]] = {
    ("lulesh-cuda", "fill_sig"): {
        "file": "lulesh.cu",
        "line": 686,
        "source": """__global__ void fill_sig(
    Real_t *__restrict__ sigxx,
    Real_t *__restrict__ sigyy,
    Real_t *__restrict__ sigzz,
    const Real_t *__restrict__ p,
    const Real_t *__restrict__ q,
    const Index_t numElem )
{
  Index_t i = blockDim.x*blockIdx.x+threadIdx.x;
  if (i >= numElem) return;
  sigxx[i] = sigyy[i] = sigzz[i] = - p[i] - q[i] ;
}""",
    },
    ("tsne-cuda", "IntegrationKernel"): {
        "file": "apply_forces.cu",
        "line": 41,
        "source": """__global__ void IntegrationKernel(
    volatile float* __restrict__ points,        // num_points * 2
    volatile float* __restrict__ attr_forces,   // num_points * 2
    volatile float* __restrict__ rep_forces,    // num_points * 2
    volatile float* __restrict__ gains,         // num_points * 2
    volatile float* __restrict__ old_forces,    // num_points * 2
    const float eta,
    const float normalization,
    const float momentum,
    const float exaggeration,
    const int   num_points)
{
    int tid;
    float dx, dy, ux, uy, gx, gy;

    tid = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tid < num_points) {
        ux = old_forces[tid];
        uy = old_forces[tid + num_points];
        gx = gains[tid];
        gy = gains[tid + num_points];
        dx = exaggeration * attr_forces[tid]              - (rep_forces[tid]              / normalization);
        dy = exaggeration * attr_forces[tid + num_points] - (rep_forces[tid + num_points] / normalization);

        gx = (signbit(dx) != signbit(ux)) ? gx + 0.2f : gx * 0.8f;
        gy = (signbit(dy) != signbit(uy)) ? gy + 0.2f : gy * 0.8f;
        gx = (gx < 0.01f) ? 0.01f : gx;
        gy = (gy < 0.01f) ? 0.01f : gy;

        ux = momentum * ux - eta * gx * dx;
        uy = momentum * uy - eta * gy * dy;

        points[tid] += ux;
        points[tid + num_points] += uy;

        attr_forces[tid]              = 0.0f;
        attr_forces[tid + num_points] = 0.0f;
        rep_forces[tid]               = 0.0f;
        rep_forces[tid + num_points]  = 0.0f;
        old_forces[tid]               = ux;
        old_forces[tid + num_points]  = uy;
        gains[tid]                    = gx;
        gains[tid + num_points]       = gy;
    }
}""",
    },
    ("addBiasResidualLayerNorm-cuda", "addBiasResidualPostLayerNormV2"): {
        "file": "kernels.h",
        "line": 202,
        "source": """template<typename T>
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
}""",
    },
}
