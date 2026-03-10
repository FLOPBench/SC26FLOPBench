solution = [
r"""__global__ void calcKinematicsForElems ( 
    const Real_t *__restrict__ xd,
    const Real_t *__restrict__ yd,
    const Real_t *__restrict__ zd,
    const Real_t *__restrict__ x,
    const Real_t *__restrict__ y,
    const Real_t *__restrict__ z,
    const Index_t *__restrict__ nodeList,
    const Real_t *__restrict__ volo,
    const Real_t *__restrict__ v,
    Real_t *__restrict__ delv,
    Real_t *__restrict__ arealg,
    Real_t *__restrict__ dxx,
    Real_t *__restrict__ dyy,
    Real_t *__restrict__ dzz,
    Real_t *__restrict__ vnew,
    const Real_t deltaTime,
    const Index_t numElem )
{
  Index_t k = blockDim.x*blockIdx.x+threadIdx.x;
  if (k >= numElem) return;

  Real_t B[3][8] ; // shape function derivatives 
  Real_t D[6] ;
  Real_t x_local[8] ;
  Real_t y_local[8] ;
  Real_t z_local[8] ;
  Real_t xd_local[8] ;
  Real_t yd_local[8] ;
  Real_t zd_local[8] ;
  Real_t detJ = Real_t(0.0) ;

  Real_t volume ;
  Real_t relativeVolume ;
  const Index_t* elemToNode = nodeList + Index_t(8)*k;

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

  // volume calculations
  volume = CalcElemVolume(x_local, y_local, z_local );
  relativeVolume = volume / volo[k] ;
  vnew[k] = relativeVolume ;
  delv[k] = relativeVolume - v[k] ;

  // set characteristic length
  arealg[k] = CalcElemCharacteristicLength(x_local, y_local, z_local,
      volume);

  // get nodal velocities from global array and copy into local arrays.
  for( Index_t lnode=0 ; lnode<8 ; ++lnode )
  {
    Index_t gnode = elemToNode[lnode];
    xd_local[lnode] = xd[gnode];
    yd_local[lnode] = yd[gnode];
    zd_local[lnode] = zd[gnode];
  }

  Real_t dt2 = Real_t(0.5) * deltaTime;
  for ( Index_t j=0 ; j<8 ; ++j )
  {
    x_local[j] -= dt2 * xd_local[j];
    y_local[j] -= dt2 * yd_local[j];
    z_local[j] -= dt2 * zd_local[j];
  }

  CalcElemShapeFunctionDerivatives( x_local, y_local, z_local,
      B, &detJ );

  CalcElemVelocityGradient( xd_local, yd_local, zd_local,
      B, detJ, D );

  // put velocity gradient quantities into their global arrays.
  dxx[k] = D[0];
  dyy[k] = D[1];
  dzz[k] = D[2];
}""".strip()
]
