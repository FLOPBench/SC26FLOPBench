solution = [
 r"""__global__ void ccc_loop2(
  const int * __restrict__ imaterial,
  const int * __restrict__ matids,
  const int * __restrict__ nextfrac,
  const double * __restrict__ rho_compact,
  const double * __restrict__ rho_compact_list, 
  const double * __restrict__ t_compact,
  const double * __restrict__ t_compact_list, 
  const double * __restrict__  Vf_compact_list,
  const double * __restrict__ n,
  double * __restrict__  p_compact,
  double * __restrict__ p_compact_list,
  int sizex, int sizey,
  int * __restrict__ mmc_index)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i >= sizex || j >= sizey) return;

  int ix = imaterial[i+sizex*j];
  if (ix <= 0) {
#ifdef FUSED
    // NOTE: I think the paper describes this algorithm (Alg. 9) wrong.
    // The solution below is what I believe to good.

    // condition is 'ix >= 0', this is the equivalent of
    // 'until ix < 0' from the paper
#ifdef LINKED
    for (ix = -ix; ix >= 0; ix = nextfrac[ix]) {
      double nm = n[matids[ix]];
      p_compact_list[ix] = (nm * rho_compact_list[ix] * t_compact_list[ix]) / Vf_compact_list[ix];
    }
#else
    for (int idx = mmc_index[-ix]; idx < mmc_index[-ix+1]; idx++) {
      double nm = n[matids[idx]];
      p_compact_list[idx] = (nm * rho_compact_list[idx] * t_compact_list[idx]) / Vf_compact_list[idx];
    }
#endif
#endif
  }
  else {
    // NOTE: HACK: we index materials from zero, but zero can be a list index
    int mat = ix - 1;
    // NOTE: There is no division by Vf here, because the fractional volume is 1.0 in the pure cell case.
    p_compact[i+sizex*j] = n[mat] * rho_compact[i+sizex*j] * t_compact[i+sizex*j];;
  }
}
""".strip()
]
