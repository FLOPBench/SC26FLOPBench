solution = [
r"""template<typename bit_factor_t, typename bit_matrix_t, typename index_t, typename error_t>
__global__ void vectorMatrixMultCompareColWarpShared(
  const bit_factor_t * __restrict__ A,
  bit_factor_t * __restrict__ B,
  const bit_matrix_t * __restrict__ C,
  const index_t height,
  const index_t width,
  const index_t padded_width,
  const uint8_t factorDim,
  const index_t startcol,
  error_t *global_error,
  const uint32_t seed,
  const float temperature,
  const float flipManyChance,
  const uint32_t flipManyDepth,
  const error_t weight)
{
  __shared__ bit_factor_t A_block[32*WARPSPERBLOCK];
  __shared__ bit_matrix_t C_block[32*WARPSPERBLOCK];

  const index_t warpIdIntern = threadIdx.x / warpSize;
  const index_t warpId = blockIdx.x * WARPSPERBLOCK + warpIdIntern;
  const index_t warpLane = threadIdx.x % warpSize;

  const index_t padded_width_blocks = SDIV(width, WARPSPERBLOCK) * WARPSPERBLOCK;
  const index_t j = (startcol + warpId) % padded_width_blocks;

  fast_kiss_state32_t state;

  const bit_factor_t B_j = j < width ? B[j] : 0;
  bit_factor_t B_j_changed = 0;
  if (j < width) {
    state = get_initial_fast_kiss_state32(seed + warpId);

    B_j_changed = B_j ^ get_flip_mask(factorDim, state, flipManyChance, flipManyDepth);
  }

  error_t error_thread = 0;
  const index_t vecLane = warpLane;
  const index_t col_in_tile = j % 32;
  const index_t colFirst = j / 32 * 32;
  const index_t padded_height_blocks = SDIV(height, WARPSPERBLOCK*32) * WARPSPERBLOCK*32;
  for (index_t i = threadIdx.x; i < padded_height_blocks; i += WARPSPERBLOCK*32) {
    A_block[threadIdx.x] = (i < height) ? A[i] : 0;
    const index_t vecRow = i / 32;
    const index_t vecFirst = vecRow * padded_width + colFirst;
    C_block[threadIdx.x] = (vecRow < SDIV(height,32)) ? C[vecFirst + warpLane] : 0;
    __syncthreads();

    if (j < width) {
      #pragma unroll
      for(index_t w = 0; w < WARPSPERBLOCK; ++w) {
        const bit_factor_t A_i = A_block[w*warpSize + warpLane];
        const int C_ij = (C_block[w*warpSize + col_in_tile] >> vecLane) & 1;

        const int product_new = (A_i & B_j_changed) ? 1 : 0;
        const int product_old = (A_i & B_j        ) ? 1 : 0;

        error_thread += error_measure(product_new, C_ij, weight)
          - error_measure(product_old, C_ij, weight);
      }
    }
    __syncthreads();
  }
  if (j < width) {
    const error_t error_warp = warpReduceSum(error_thread);
    // Thread with warpLane==0 now has total error of warp

    // Thread 0 checks if new low has been found and applies if necessary
    if (warpLane == 0) {
      // Metropolis–Hastings algorithm
      if (metro(state, error_warp, temperature, height)) {
        B[j] = B_j_changed;
        atomicAdd(global_error, error_warp);
      }
    }
  }
}""".strip()
]
