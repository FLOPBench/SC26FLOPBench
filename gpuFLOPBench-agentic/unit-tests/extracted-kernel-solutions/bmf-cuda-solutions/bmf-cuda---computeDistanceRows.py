solution = [
r"""template<typename bit_factor_t, typename bit_matrix_t, typename index_t, typename error_t>
__global__ void computeDistanceRows(
  const bit_factor_t * __restrict__ Ab,
  const bit_factor_t * __restrict__ Bb,
  const bit_matrix_t * __restrict__ Cb, 
  const index_t height, const index_t width,
  const index_t padded_width,
  const uint8_t factorDim,
  const int weight,
  error_t *global_error)
{
  const index_t warpId = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
  const index_t warpLane = threadIdx.x % warpSize;

  const index_t i = warpId;
  error_t error_thread = 0;
  if (i < height) {
    const bit_factor_t A_i = Ab[i];

    for (index_t j = warpLane; j < width; j += warpSize) {
      const int product = (A_i & Bb[j]) ? 1 : 0;

      const index_t vecId = i / 32 * padded_width + j;
      const index_t vecLane = i % 32;
      const int C_ij = (Cb[vecId] >> vecLane) & 1;

      error_thread += error_measure(product, C_ij, weight);
    }
  }

  __shared__ error_t reductionArray[WARPSPERBLOCK];
  const error_t error_block = blockReduceSum(error_thread, reductionArray);
  // Thread with threadIdx.x==0 now has total error of block

  if (threadIdx.x == 0)
    atomicAdd(global_error, error_block);
}""".strip()
]
