solution = [
r"""__global__
void computeDistanceRowsShared(
  const float * __restrict__ A,
  const float * __restrict__ B,
  const uint32_t * __restrict__ Cb, 
  const int height, const int width,
  const int padded_width,
  const uint8_t factorDim,
  const int inverse_density,
        int *__restrict__ global_error)
{
  const int warpIdIntern = threadIdx.x / warpSize;
  const int warpId = blockIdx.x * WARPSPERBLOCK + warpIdIntern;
  const int warpLane = threadIdx.x % warpSize;

  __shared__ int reductionArray[WARPSPERBLOCK];
  __shared__ float B_block[CHUNK_SIZE][32];
  __shared__ uint32_t C_block[CHUNK_SIZE];

  const uint32_t dim_mask = FULLMASK >> (32 - factorDim);

  const int i = warpId;
  const int k = warpLane;
  const bool A_i_k = A[i*warpSize + k] > 0.5f;

  const int vecRow = i / 32;
  const int vecFirst = vecRow * padded_width;
  const int vecLane = i % 32;
  int error_warp = 0;
  for (int j_chunk = 0; j_chunk < padded_width; j_chunk += CHUNK_SIZE) {
    #pragma unroll
    for(int j_local = warpIdIntern; j_local < CHUNK_SIZE; j_local += WARPSPERBLOCK) {
      const int j = j_chunk + j_local;
      B_block[j_local][k] = j < width ? B[j * warpSize + k] : 0;
    }
    if(threadIdx.x < CHUNK_SIZE) {
      const int vecId = vecFirst + j_chunk;
      C_block[threadIdx.x] = Cb[vecId + threadIdx.x];
    }
    __syncthreads();

    if (i < height) {
      #pragma unroll
      for(int j_local = 0; j_local < CHUNK_SIZE; ++j_local) {
        // int product = __any_sync(dim_mask, A_i_k && (B[j*warpSize + k] > 0.5f)) ? 1 : 0;
        const int product = __any_sync(dim_mask, A_i_k && (B_block[j_local][k] > 0.5f)) ? 1 : 0;

        const int C_ij = (C_block[j_local] >> vecLane) & 1;

        error_warp += error_measure(product, C_ij, inverse_density);
      }
    }
    __syncthreads();
  }

  if(warpLane == 0)
    reductionArray[warpIdIntern] = error_warp;
  __syncthreads();

  int error_block;
  if(warpIdIntern == 0) {
    error_block = warpReduceSum(reductionArray[warpLane], WARPSPERBLOCK);

    if (threadIdx.x == 0) {
      // Thread with threadIdx.x==0 now has total error of block
      atomicAdd(global_error, error_block);
    }
  }
}""".strip(),
r"""template<typename bit_factor_t, typename bit_matrix_t, typename index_t, typename error_t>
__global__ void computeDistanceRowsShared(
  const bit_factor_t * __restrict__ Ab,
  const bit_factor_t * __restrict__ Bb,
  const bit_matrix_t * __restrict__ Cb, 
  const index_t height,
  const index_t width,
  const index_t padded_width,
  const uint8_t factorDim,
  const error_t weight,
  error_t *global_error)
{
  __shared__ bit_factor_t B_block[ 32 * WARPSPERBLOCK ];
  __shared__ bit_matrix_t C_block[ 32 * WARPSPERBLOCK ];

  const index_t warpIdIntern = threadIdx.x / warpSize;
  const index_t warpId = blockIdx.x * WARPSPERBLOCK + warpIdIntern;
  const index_t warpLane = threadIdx.x % warpSize;

  const index_t blockSize = WARPSPERBLOCK*32;

  const index_t i = warpId;
  const bit_factor_t A_i = i < height ? Ab[i] : 0;

  const index_t vecRow = i / 32;
  const index_t vecFirst = vecRow * padded_width;
  const index_t vecLane = i % 32;
  const index_t col_in_tile = warpLane;
  const index_t padded_width_blocks = SDIV(width, blockSize) * blockSize;
  error_t error_thread = 0;
  for (index_t j = threadIdx.x; j < padded_width_blocks; j += blockSize) {
    B_block[threadIdx.x] = (j < width) ? Bb[j] : 0;
    C_block[threadIdx.x] = (j < width) ? Cb[vecFirst + j] : 0;
    __syncthreads();

    if(i < height) {
      #pragma unroll
      for(index_t w = 0; w < WARPSPERBLOCK; ++w) {
        const bit_factor_t B_j = B_block[w*warpSize + warpLane];

        const int C_ij = (C_block[w*warpSize + col_in_tile] >> vecLane) & 1;

        const int product = (B_j & A_i) ? 1 : 0;

        error_thread += error_measure(product, C_ij, weight);
      }
    }
    __syncthreads();
  }

  __shared__ error_t reductionArray[WARPSPERBLOCK];
  const error_t error_block = blockReduceSum(error_thread, reductionArray);
  // Thread with threadIdx.x==0 now has total error of block

  if (threadIdx.x == 0)
    atomicAdd(global_error, error_block);
}""".strip()
]
