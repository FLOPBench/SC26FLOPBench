solution = [
r"""__global__ void integrateStress (
    Real_t *__restrict__ fx_elem,
    Real_t *__restrict__ fy_elem,
    Real_t *__restrict__ fz_elem,
    const Real_t *__restrict__ x,
    const Real_t *__restrict__ y,
    const Real_t *__restrict__ z,
    const Index_t *__restrict__ nodelist,
    const Real_t *__restrict__ sigxx,
    const Real_t *__restrict__ sigyy,
    const Real_t *__restrict__ sigzz,
    Real_t *__restrict__ determ,
    const Index_t numElem) 
{
  Index_t k = blockDim.x*blockIdx.x+threadIdx.x;
  if (k >= numElem) return;

  const Index_t* const elemToNode = nodelist + Index_t(8)*k;
  Real_t B[3][8] ;// shape function derivatives
  Real_t x_local[8] ;
  Real_t y_local[8] ;
  Real_t z_local[8] ;
  determ[k] = Real_t(10.0);

  // get nodal coordinates from global arrays and copy into local arrays.
  Index_t nd0i = elemToNode[0] ;
  Index_t nd1i = elemToNode[1] ;
  Index_t nd2i = elemToNode[2] ;
  Index_t nd3i = elemToNode[3] ;
  Index_t nd4i = elemToNode[4] ;
  Index_t nd5i = elemToNode[5] ;
  Index_t nd6i = elemToNode[6] ;
  Index_t nd7i = elemToNode[7] ;

  x_local[0] = x[nd0i];
  x_local[1] = x[nd1i];
  x_local[2] = x[nd2i];
  x_local[3] = x[nd3i];
  x_local[4] = x[nd4i];
  x_local[5] = x[nd5i];
  x_local[6] = x[nd6i];
  x_local[7] = x[nd7i];

  y_local[0] = y[nd0i];
  y_local[1] = y[nd1i];
  y_local[2] = y[nd2i];
  y_local[3] = y[nd3i];
  y_local[4] = y[nd4i];
  y_local[5] = y[nd5i];
  y_local[6] = y[nd6i];
  y_local[7] = y[nd7i];

  z_local[0] = z[nd0i];
  z_local[1] = z[nd1i];
  z_local[2] = z[nd2i];
  z_local[3] = z[nd3i];
  z_local[4] = z[nd4i];
  z_local[5] = z[nd5i];
  z_local[6] = z[nd6i];
  z_local[7] = z[nd7i];

  // Volume calculation involves extra work for numerical consistency
  CalcElemShapeFunctionDerivatives(x_local, y_local, z_local, B, &determ[k]);

  CalcElemNodeNormals( B[0], B[1], B[2], x_local, y_local, z_local );

  // Eliminate thread writing conflicts at the nodes by giving
  // each element its own copy to write to
  SumElemStressesToNodeForces( B, sigxx[k], sigyy[k], sigzz[k],
      &fx_elem[k*8],
      &fy_elem[k*8],
      &fz_elem[k*8] ) ;
}""".strip()
]
