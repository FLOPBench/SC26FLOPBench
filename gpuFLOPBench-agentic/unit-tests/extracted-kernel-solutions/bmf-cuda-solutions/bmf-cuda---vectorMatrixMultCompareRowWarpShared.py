solution = [
r"""template<typename bit_factor_t, typename bit_matrix_t, typename index_t, typename error_t>
__global__ void vectorMatrixMultCompareRowWarpShared(
        bit_factor_t * __restrict__ A,
  const bit_factor_t * __restrict__ B,
  const bit_matrix_t * __restrict__ C,
  const index_t height,
  const index_t width,
  const index_t padded_width,
  const uint8_t factorDim,
  const index_t startrow,
  error_t *global_error,
  const uint32_t seed, 
  const float temperature,
  const float flipManyChance,
  const uint32_t flipManyDepth,
  const error_t weight)
{
  __shared__ bit_factor_t B_block[ 32 * WARPSPERBLOCK ];
  __shared__ bit_matrix_t C_block[ 32 * WARPSPERBLOCK ];

  const index_t warpId = blockIdx.x * WARPSPERBLOCK + threadIdx.x / warpSize;
  const index_t warpLane = threadIdx.x % warpSize;

  const index_t padded_height_blocks = SDIV(height, WARPSPERBLOCK) * WARPSPERBLOCK;
  const index_t i = (startrow + warpId) % padded_height_blocks;

  fast_kiss_state32_t state;

  const bit_factor_t A_i = i < height ? A[i] : 0;
  bit_factor_t A_i_changed = 0;
  if (i < height) {
    state = get_initial_fast_kiss_state32(seed + warpId);

    A_i_changed = A_i ^ get_flip_mask(factorDim, state, flipManyChance, flipManyDepth);
  }

  const index_t vecRow = i / 32;
  const index_t vecFirst = vecRow * padded_width;
  const index_t vecLane = i % 32;
  const index_t col_in_tile = warpLane;
  const index_t padded_width_blocks = SDIV(width, WARPSPERBLOCK*32) * WARPSPERBLOCK*32;
  error_t error_thread = 0;
  for (index_t j = threadIdx.x; j < padded_width_blocks; j += WARPSPERBLOCK*32) {
    B_block[threadIdx.x] = (j < width) ? B[j] : 0;
    C_block[threadIdx.x] = (j < width) ? C[vecFirst + j] : 0;
    __syncthreads();

    if(i < height) {
      #pragma unroll
      for(index_t w = 0; w < WARPSPERBLOCK; ++w) {
        const bit_factor_t B_j = B_block[w*warpSize + warpLane];
        const int C_ij = (C_block[w*warpSize + col_in_tile] >> vecLane) & 1;

        const int product_new = (B_j & A_i_changed) ? 1 : 0;
        const int product_old = (B_j & A_i        ) ? 1 : 0;

        error_thread += error_measure(product_new, C_ij, weight)
          - error_measure(product_old, C_ij, weight);
      }
    }
    __syncthreads();
  }
  if(i < height) {
    const error_t error_warp = warpReduceSum(error_thread);
    // Thread with warpLane==0 now has total error of warp

    // Thread 0 checks if new low has been found and applies if necessary
    if (warpLane == 0) {
      // Metropolis–Hastings algorithm
      if (metro(state, error_warp, temperature, width)) {
        A[i] = A_i_changed;
        atomicAdd(global_error, error_warp);
      }
    }
  }
}""".strip()
]
