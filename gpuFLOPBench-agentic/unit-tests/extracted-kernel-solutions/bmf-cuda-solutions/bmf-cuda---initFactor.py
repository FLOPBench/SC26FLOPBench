solution = [
r"""__global__
void initFactor(
  float * A,
  const int height,
  const uint8_t factorDim,
  const uint32_t seed, 
  const float threshold)
{
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int warpId = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
  const int warpLane = threadIdx.x % warpSize;

  if(warpId < height) {
    fast_kiss_state32_t state = get_initial_fast_kiss_state32(seed + tid);
    const int i = warpId;
    const int j = warpLane;

    A[i * factorDim + j] = j < factorDim ? fast_kiss32(state) < threshold : 0;
  }
}""".strip(),
r"""template<typename bit_vector_t, typename index_t>
__global__ void initFactor(
  bit_vector_t * Ab,
  const index_t height,
  const uint8_t factorDim,
  const uint32_t seed, 
  const float threshold)
{
  const index_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  if(tid < height) {
    bit_vector_t Ai = 0;

    const int randDepth = -log2f(threshold)+1;
    // if threshold very small simply initilize as 0s (also catch threshold=0)
    if(randDepth < 16) {
      fast_kiss_state32_t state = get_initial_fast_kiss_state32(seed + tid);

      Ai = ~bit_vector_t(0) >> (32-factorDim);
      for(int d=0; d<randDepth; ++d)
        Ai &= fast_kiss32(state);
    }
    Ab[tid] = Ai;
  }
}""".strip()
]
