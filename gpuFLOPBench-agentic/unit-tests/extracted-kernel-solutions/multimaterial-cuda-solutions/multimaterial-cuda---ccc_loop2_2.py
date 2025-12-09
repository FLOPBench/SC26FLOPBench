solution = [
 r"""__global__ void ccc_loop2_2(
  const int * __restrict__ matids,
  const double * __restrict__ rho_compact_list, 
  const double * __restrict__ t_compact_list,
  const double * __restrict__ Vf_compact_list,
  const double * __restrict__ n,
  double * __restrict__ p_compact_list,
  int * __restrict__ mmc_index,
  int mmc_cells)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= mmc_cells) return;
  double nm = n[matids[idx]];
  p_compact_list[idx] = (nm * rho_compact_list[idx] * t_compact_list[idx]) / Vf_compact_list[idx];
}
""".strip()
]
