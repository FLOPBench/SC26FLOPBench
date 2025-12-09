solution = [
r"""__global__ void hgc (
    Real_t *__restrict__ dvdx,
    Real_t *__restrict__ dvdy,
    Real_t *__restrict__ dvdz,
    Real_t *__restrict__ x8n,
    Real_t *__restrict__ y8n,
    Real_t *__restrict__ z8n,
    Real_t *__restrict__ determ,

    const Real_t *__restrict__ x,
    const Real_t *__restrict__ y,
    const Real_t *__restrict__ z,
    const Index_t *__restrict__ nodelist,
    const Real_t *__restrict__ volo,
    const Real_t *__restrict__ v,
    int *__restrict__ vol_error,
    const Index_t numElem )
{
  Index_t i = blockDim.x*blockIdx.x+threadIdx.x;
  if (i >= numElem) return;

  Real_t  x1[8],  y1[8],  z1[8] ;
  Real_t pfx[8], pfy[8], pfz[8] ;

  const Index_t* elemToNode = nodelist + Index_t(8)*i;

  // CollectDomainNodesToElemNodes(domain, elemToNode, x1, y1, z1);

  // inline the function manually
  Index_t nd0i = elemToNode[0] ;
  Index_t nd1i = elemToNode[1] ;
  Index_t nd2i = elemToNode[2] ;
  Index_t nd3i = elemToNode[3] ;
  Index_t nd4i = elemToNode[4] ;
  Index_t nd5i = elemToNode[5] ;
  Index_t nd6i = elemToNode[6] ;
  Index_t nd7i = elemToNode[7] ;

  x1[0] = x[nd0i];
  x1[1] = x[nd1i];
  x1[2] = x[nd2i];
  x1[3] = x[nd3i];
  x1[4] = x[nd4i];
  x1[5] = x[nd5i];
  x1[6] = x[nd6i];
  x1[7] = x[nd7i];

  y1[0] = y[nd0i];
  y1[1] = y[nd1i];
  y1[2] = y[nd2i];
  y1[3] = y[nd3i];
  y1[4] = y[nd4i];
  y1[5] = y[nd5i];
  y1[6] = y[nd6i];
  y1[7] = y[nd7i];

  z1[0] = z[nd0i];
  z1[1] = z[nd1i];
  z1[2] = z[nd2i];
  z1[3] = z[nd3i];
  z1[4] = z[nd4i];
  z1[5] = z[nd5i];
  z1[6] = z[nd6i];
  z1[7] = z[nd7i];

  CalcElemVolumeDerivative(pfx, pfy, pfz, x1, y1, z1);

  /* load into temporary storage for FB Hour Glass control */
  for(Index_t ii=0;ii<8;++ii){
    Index_t jj=8*i+ii;

    dvdx[jj] = pfx[ii];
    dvdy[jj] = pfy[ii];
    dvdz[jj] = pfz[ii];
    x8n[jj]  = x1[ii];
    y8n[jj]  = y1[ii];
    z8n[jj]  = z1[ii];
  }

  determ[i] = volo[i] * v[i];

  /* Do a check for negative volumes */
  if ( v[i] <= Real_t(0.0) ) {
    vol_error[0] = i;
  }
}""".strip()
]
