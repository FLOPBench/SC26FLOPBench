solution = [
 r"""__global__ void ccc_loop3(
  const int * __restrict__ imaterial,
  const int * __restrict__ nextfrac,
  const int * __restrict__ matids,
  const double * __restrict__ rho_compact, 
  const double * __restrict__ rho_compact_list, 
  double * __restrict__ rho_mat_ave_compact, 
  double * __restrict__ rho_mat_ave_compact_list, 
  const double * __restrict__ x,
  const double * __restrict__ y,
  int sizex, int sizey,
  int * __restrict__ mmc_index)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i >= sizex-1 || j >= sizey-1 || i < 1 || j < 1) return;

  // o: outer
  double xo = x[i+sizex*j];
  double yo = y[i+sizex*j];

  // There are at most 9 neighbours in 2D case.
  double dsqr[9];

  // for all neighbours
  for (int nj = -1; nj <= 1; nj++) {

    for (int ni = -1; ni <= 1; ni++) {

      dsqr[(nj+1)*3 + (ni+1)] = 0.0;

      // i: inner
      double xi = x[(i+ni)+sizex*(j+nj)];
      double yi = y[(i+ni)+sizex*(j+nj)];

      dsqr[(nj+1)*3 + (ni+1)] += (xo - xi) * (xo - xi);
      dsqr[(nj+1)*3 + (ni+1)] += (yo - yi) * (yo - yi);
    }
  }

  int ix = imaterial[i+sizex*j];

  if (ix <= 0) {
    // condition is 'ix >= 0', this is the equivalent of
    // 'until ix < 0' from the paper
#ifdef LINKED
    for (ix = -ix; ix >= 0; ix = nextfrac[ix]) {
#else
      for (int ix = mmc_index[-imaterial[i+sizex*j]]; ix < mmc_index[-imaterial[i+sizex*j]+1]; ix++) {
#endif
        int mat = matids[ix];
        double rho_sum = 0.0;
        int Nn = 0;

        // for all neighbours
        for (int nj = -1; nj <= 1; nj++) {

          for (int ni = -1; ni <= 1; ni++) {

            int ci = i+ni, cj = j+nj;
            int jx = imaterial[ci+sizex*cj];

            if (jx <= 0) {
              // condition is 'jx >= 0', this is the equivalent of
              // 'until jx < 0' from the paper
#ifdef LINKED
              for (jx = -jx; jx >= 0; jx = nextfrac[jx]) {
#else
                for (int jx = mmc_index[-imaterial[ci+sizex*cj]]; jx < mmc_index[-imaterial[ci+sizex*cj]+1]; jx++) {
#endif
                  if (matids[jx] == mat) {
                    rho_sum += rho_compact_list[jx] / dsqr[(nj+1)*3 + (ni+1)];
                    Nn += 1;

                    // The loop has an extra condition: "and not found".
                    // This makes sense, if the material is found, there won't be any more of the same.
                    break;
                  }
                }
              }
              else {
                // NOTE: In this case, the neighbour is a pure cell, its material index is in jx.
                // In contrast, Algorithm 10 loads matids[jx] which I think is wrong.

                // NOTE: HACK: we index materials from zero, but zero can be a list index
                int mat_neighbour = jx - 1;
                if (mat == mat_neighbour) {
                  rho_sum += rho_compact[ci+sizex*cj] / dsqr[(nj+1)*3 + (ni+1)];
                  Nn += 1;
                }
              } // end if (jx <= 0)
            } // end for (int ni)
          } // end for (int nj)

          rho_mat_ave_compact_list[ix] = rho_sum / Nn;
        } // end for (ix = -ix)
      } // end if (ix <= 0)
      else {
        // NOTE: In this case, the cell is a pure cell, its material index is in ix.
        // In contrast, Algorithm 10 loads matids[ix] which I think is wrong.

        // NOTE: HACK: we index materials from zero, but zero can be a list index
        int mat = ix - 1;

        double rho_sum = 0.0;
        int Nn = 0;

        // for all neighbours
        for (int nj = -1; nj <= 1; nj++) {
          if ((j + nj < 0) || (j + nj >= sizey)) // TODO: better way?
            continue;

          for (int ni = -1; ni <= 1; ni++) {
            if ((i + ni < 0) || (i + ni >= sizex)) // TODO: better way?
              continue;

            int ci = i+ni, cj = j+nj;
            int jx = imaterial[ci+sizex*cj];

            if (jx <= 0) {
              // condition is 'jx >= 0', this is the equivalent of
              // 'until jx < 0' from the paper
#ifdef LINKED
              for (jx = -jx; jx >= 0; jx = nextfrac[jx]) {
#else
                for (int jx = mmc_index[-imaterial[ci+sizex*cj]]; jx < mmc_index[-imaterial[ci+sizex*cj]+1]; jx++) {
#endif
                  if (matids[jx] == mat) {
                    rho_sum += rho_compact_list[jx] / dsqr[(nj+1)*3 + (ni+1)];
                    Nn += 1;

                    // The loop has an extra condition: "and not found".
                    // This makes sense, if the material is found, there won't be any more of the same.
                    break;
                  }
                }
              }
              else {
                // NOTE: In this case, the neighbour is a pure cell, its material index is in jx.
                // In contrast, Algorithm 10 loads matids[jx] which I think is wrong.

                // NOTE: HACK: we index materials from zero, but zero can be a list index
                int mat_neighbour = jx - 1;
                if (mat == mat_neighbour) {
                  rho_sum += rho_compact[ci+sizex*cj] / dsqr[(nj+1)*3 + (ni+1)];
                  Nn += 1;
                }
              } // end if (jx <= 0)
            } // end for (int ni)
          } // end for (int nj)

          rho_mat_ave_compact[i+sizex*j] = rho_sum / Nn;
        } // end else
      }
}
""".strip()
]
