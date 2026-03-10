solution = [
r"""__global__ void fb (
    const Real_t *__restrict__ dvdx,
    const Real_t *__restrict__ dvdy,
    const Real_t *__restrict__ dvdz,
    const Real_t *__restrict__ x8n,
    const Real_t *__restrict__ y8n,
    const Real_t *__restrict__ z8n,
    const Real_t *__restrict__ determ,
    const Real_t *__restrict__ xd,
    const Real_t *__restrict__ yd,
    const Real_t *__restrict__ zd,
    const Real_t *__restrict__ ss,
    const Real_t *__restrict__ elemMass,
    const Index_t *__restrict__ nodelist,
    const Real_t *__restrict__ gamma,
    Real_t *__restrict__ fx_elem,
    Real_t *__restrict__ fy_elem,
    Real_t *__restrict__ fz_elem,
    Real_t hgcoef,
    const Index_t numElem )
{
  Index_t i2 = blockDim.x*blockIdx.x+threadIdx.x;
  if (i2 >= numElem) return;

  Index_t i3 = 8*i2;

  const Index_t* elemToNode = nodelist + i3;

  Real_t hgfx[8], hgfy[8], hgfz[8] ;

  Real_t coefficient;

  Real_t hourgam[8][4];
  Real_t xd1[8], yd1[8], zd1[8] ;

  Real_t volinv = ONE/determ[i2];
  Real_t ss1, mass1, volume13 ;

  for(Index_t i1=0;i1<4;++i1) {

    Real_t hourmodx =
      x8n[i3]   * gamma[i1*8+0] + x8n[i3+1] * gamma[i1*8+1] +
      x8n[i3+2] * gamma[i1*8+2] + x8n[i3+3] * gamma[i1*8+3] +
      x8n[i3+4] * gamma[i1*8+4] + x8n[i3+5] * gamma[i1*8+5] +
      x8n[i3+6] * gamma[i1*8+6] + x8n[i3+7] * gamma[i1*8+7];

    Real_t hourmody =
      y8n[i3]   * gamma[i1*8+0] + y8n[i3+1] * gamma[i1*8+1] +
      y8n[i3+2] * gamma[i1*8+2] + y8n[i3+3] * gamma[i1*8+3] +
      y8n[i3+4] * gamma[i1*8+4] + y8n[i3+5] * gamma[i1*8+5] +
      y8n[i3+6] * gamma[i1*8+6] + y8n[i3+7] * gamma[i1*8+7];

    Real_t hourmodz =
      z8n[i3]   * gamma[i1*8+0] + z8n[i3+1] * gamma[i1*8+1] +
      z8n[i3+2] * gamma[i1*8+2] + z8n[i3+3] * gamma[i1*8+3] +
      z8n[i3+4] * gamma[i1*8+4] + z8n[i3+5] * gamma[i1*8+5] +
      z8n[i3+6] * gamma[i1*8+6] + z8n[i3+7] * gamma[i1*8+7];

    hourgam[0][i1] = gamma[i1*8+0] - volinv*(dvdx[i3  ] * hourmodx +
        dvdy[i3  ] * hourmody +
        dvdz[i3  ] * hourmodz );

    hourgam[1][i1] = gamma[i1*8+1] - volinv*(dvdx[i3+1] * hourmodx +
        dvdy[i3+1] * hourmody +
        dvdz[i3+1] * hourmodz );

    hourgam[2][i1] = gamma[i1*8+2] - volinv*(dvdx[i3+2] * hourmodx +
        dvdy[i3+2] * hourmody +
        dvdz[i3+2] * hourmodz );

    hourgam[3][i1] = gamma[i1*8+3] - volinv*(dvdx[i3+3] * hourmodx +
        dvdy[i3+3] * hourmody +
        dvdz[i3+3] * hourmodz );

    hourgam[4][i1] = gamma[i1*8+4] - volinv*(dvdx[i3+4] * hourmodx +
        dvdy[i3+4] * hourmody +
        dvdz[i3+4] * hourmodz );

    hourgam[5][i1] = gamma[i1*8+5] - volinv*(dvdx[i3+5] * hourmodx +
        dvdy[i3+5] * hourmody +
        dvdz[i3+5] * hourmodz );

    hourgam[6][i1] = gamma[i1*8+6] - volinv*(dvdx[i3+6] * hourmodx +
        dvdy[i3+6] * hourmody +
        dvdz[i3+6] * hourmodz );

    hourgam[7][i1] = gamma[i1*8+7] - volinv*(dvdx[i3+7] * hourmodx +
        dvdy[i3+7] * hourmody +
        dvdz[i3+7] * hourmodz );

  }

  /* compute forces */
  /* store forces into h arrays (force arrays) */

  ss1 = ss[i2];
  mass1 = elemMass[i2];
  volume13 = cbrt(determ[i2]);

  Index_t n0si2 = elemToNode[0];
  Index_t n1si2 = elemToNode[1];
  Index_t n2si2 = elemToNode[2];
  Index_t n3si2 = elemToNode[3];
  Index_t n4si2 = elemToNode[4];
  Index_t n5si2 = elemToNode[5];
  Index_t n6si2 = elemToNode[6];
  Index_t n7si2 = elemToNode[7];

  xd1[0] = xd[n0si2];
  xd1[1] = xd[n1si2];
  xd1[2] = xd[n2si2];
  xd1[3] = xd[n3si2];
  xd1[4] = xd[n4si2];
  xd1[5] = xd[n5si2];
  xd1[6] = xd[n6si2];
  xd1[7] = xd[n7si2];

  yd1[0] = yd[n0si2];
  yd1[1] = yd[n1si2];
  yd1[2] = yd[n2si2];
  yd1[3] = yd[n3si2];
  yd1[4] = yd[n4si2];
  yd1[5] = yd[n5si2];
  yd1[6] = yd[n6si2];
  yd1[7] = yd[n7si2];

  zd1[0] = zd[n0si2];
  zd1[1] = zd[n1si2];
  zd1[2] = zd[n2si2];
  zd1[3] = zd[n3si2];
  zd1[4] = zd[n4si2];
  zd1[5] = zd[n5si2];
  zd1[6] = zd[n6si2];
  zd1[7] = zd[n7si2];

  coefficient = hgcoef * Real_t(-0.01) * ss1 * mass1 / volume13;

  Real_t hxx[4], hyy[4], hzz[4];

  for(Index_t i = 0; i < 4; i++) {
    hxx[i] = hourgam[0][i] * xd1[0] + hourgam[1][i] * xd1[1] +
      hourgam[2][i] * xd1[2] + hourgam[3][i] * xd1[3] +
      hourgam[4][i] * xd1[4] + hourgam[5][i] * xd1[5] +
      hourgam[6][i] * xd1[6] + hourgam[7][i] * xd1[7];
  }
  for(Index_t i = 0; i < 8; i++) {
    hgfx[i] = coefficient *
      (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
       hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
  }
  for(Index_t i = 0; i < 4; i++) {
    hyy[i] = hourgam[0][i] * yd1[0] + hourgam[1][i] * yd1[1] +
      hourgam[2][i] * yd1[2] + hourgam[3][i] * yd1[3] +
      hourgam[4][i] * yd1[4] + hourgam[5][i] * yd1[5] +
      hourgam[6][i] * yd1[6] + hourgam[7][i] * yd1[7];
  }
  for(Index_t i = 0; i < 8; i++) {
    hgfy[i] = coefficient *
      (hourgam[i][0] * hyy[0] + hourgam[i][1] * hyy[1] +
       hourgam[i][2] * hyy[2] + hourgam[i][3] * hyy[3]);
  }
  for(Index_t i = 0; i < 4; i++) {
    hzz[i] = hourgam[0][i] * zd1[0] + hourgam[1][i] * zd1[1] +
      hourgam[2][i] * zd1[2] + hourgam[3][i] * zd1[3] +
      hourgam[4][i] * zd1[4] + hourgam[5][i] * zd1[5] +
      hourgam[6][i] * zd1[6] + hourgam[7][i] * zd1[7];
  }
  for(Index_t i = 0; i < 8; i++) {
    hgfz[i] = coefficient *
      (hourgam[i][0] * hzz[0] + hourgam[i][1] * hzz[1] +
       hourgam[i][2] * hzz[2] + hourgam[i][3] * hzz[3]);
  }

  // With the threaded version, we write into local arrays per elem
  // so we don't have to worry about race conditions

  Real_t *fx_local = fx_elem + i3 ;
  fx_local[0] = hgfx[0];
  fx_local[1] = hgfx[1];
  fx_local[2] = hgfx[2];
  fx_local[3] = hgfx[3];
  fx_local[4] = hgfx[4];
  fx_local[5] = hgfx[5];
  fx_local[6] = hgfx[6];
  fx_local[7] = hgfx[7];

  Real_t *fy_local = fy_elem + i3 ;
  fy_local[0] = hgfy[0];
  fy_local[1] = hgfy[1];
  fy_local[2] = hgfy[2];
  fy_local[3] = hgfy[3];
  fy_local[4] = hgfy[4];
  fy_local[5] = hgfy[5];
  fy_local[6] = hgfy[6];
  fy_local[7] = hgfy[7];

  Real_t *fz_local = fz_elem + i3 ;
  fz_local[0] = hgfz[0];
  fz_local[1] = hgfz[1];
  fz_local[2] = hgfz[2];
  fz_local[3] = hgfz[3];
  fz_local[4] = hgfz[4];
  fz_local[5] = hgfz[5];
  fz_local[6] = hgfz[6];
  fz_local[7] = hgfz[7];
}""".strip()
]
