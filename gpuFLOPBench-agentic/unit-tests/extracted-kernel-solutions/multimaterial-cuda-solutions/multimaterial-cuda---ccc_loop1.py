solution = [
 r"""__global__ void ccc_loop1(
  const int * __restrict__ imaterial,
  const int * __restrict__ nextfrac,
  const double * __restrict__ rho_compact,
  const double * __restrict__ rho_compact_list, 
  const double * __restrict__ Vf_compact_list,
  const double * __restrict__ V,
  double * __restrict__ rho_ave_compact,
  int sizex, int sizey,
  int * __restrict__ mmc_index)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i >= sizex || j >= sizey) return;
#ifdef FUSED
  double ave = 0.0;
  int ix = imaterial[i+sizex*j];

  if (ix <= 0) {
    // condition is 'ix >= 0', this is the equivalent of
    // 'until ix < 0' from the paper
#ifdef LINKED
    for (ix = -ix; ix >= 0; ix = nextfrac[ix]) {
      ave += rho_compact_list[ix] * Vf_compact_list[ix];
    }
#else
    for (int idx = mmc_index[-ix]; idx < mmc_index[-ix+1]; idx++) {
      ave += rho_compact_list[idx] * Vf_compact_list[idx];  
    }
#endif
    rho_ave_compact[i+sizex*j] = ave/V[i+sizex*j];
  }
  else {
#endif
    // We use a distinct output array for averages.
    // In case of a pure cell, the average density equals to the total.
    rho_ave_compact[i+sizex*j] = rho_compact[i+sizex*j] / V[i+sizex*j];
#ifdef FUSED
  }
#endif
}
""".strip()
]
