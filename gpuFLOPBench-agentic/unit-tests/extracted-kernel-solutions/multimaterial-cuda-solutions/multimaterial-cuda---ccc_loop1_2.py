solution = [
 r"""__global__ void ccc_loop1_2(
  const double * __restrict__ rho_compact_list,
  const double * __restrict__  Vf_compact_list,
  const double * __restrict__  V,
  double * __restrict__ rho_ave_compact,
  const int * __restrict__ mmc_index,
  const int  mmc_cells,
  const int * __restrict__ mmc_i,
  const int * __restrict__ mmc_j,
  int sizex, int sizey)
{
  int c = threadIdx.x + blockIdx.x * blockDim.x;
  if (c >= mmc_cells) return;
  double ave = 0.0;
  for (int m = mmc_index[c]; m < mmc_index[c+1]; m++) {
    ave +=  rho_compact_list[m] * Vf_compact_list[m];
  }
  rho_ave_compact[mmc_i[c]+sizex*mmc_j[c]] = ave/V[mmc_i[c]+sizex*mmc_j[c]];
}
""".strip()
]
