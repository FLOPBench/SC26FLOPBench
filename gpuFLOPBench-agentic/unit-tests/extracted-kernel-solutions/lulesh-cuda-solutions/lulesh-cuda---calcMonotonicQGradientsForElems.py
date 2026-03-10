solution = [
r"""__global__ void calcMonotonicQGradientsForElems (
    const Real_t *__restrict__ xd,
    const Real_t *__restrict__ yd,
    const Real_t *__restrict__ zd,
    const Real_t *__restrict__ x,
    const Real_t *__restrict__ y,
    const Real_t *__restrict__ z,
    const Index_t *__restrict__ nodelist,
    const Real_t *__restrict__ volo,
    Real_t *__restrict__ delv_eta,
    Real_t *__restrict__ delx_eta,
    Real_t *__restrict__ delv_zeta,
    Real_t *__restrict__ delx_zeta,
    Real_t *__restrict__ delv_xi,
    Real_t *__restrict__ delx_xi,
    const Real_t *__restrict__ vnew,
    const Index_t numElem )
{
  Index_t i = blockDim.x*blockIdx.x+threadIdx.x;
  if (i >= numElem) return;

  Real_t ax,ay,az ;
  Real_t dxv,dyv,dzv ;

  const Index_t *elemToNode = nodelist + Index_t(8) * i;
  Index_t n0 = elemToNode[0] ;
  Index_t n1 = elemToNode[1] ;
  Index_t n2 = elemToNode[2] ;
  Index_t n3 = elemToNode[3] ;
  Index_t n4 = elemToNode[4] ;
  Index_t n5 = elemToNode[5] ;
  Index_t n6 = elemToNode[6] ;
  Index_t n7 = elemToNode[7] ;

  Real_t x0 = x[n0] ;
  Real_t x1 = x[n1] ;
  Real_t x2 = x[n2] ;
  Real_t x3 = x[n3] ;
  Real_t x4 = x[n4] ;
  Real_t x5 = x[n5] ;
  Real_t x6 = x[n6] ;
  Real_t x7 = x[n7] ;

  Real_t y0 = y[n0] ;
  Real_t y1 = y[n1] ;
  Real_t y2 = y[n2] ;
  Real_t y3 = y[n3] ;
  Real_t y4 = y[n4] ;
  Real_t y5 = y[n5] ;
  Real_t y6 = y[n6] ;
  Real_t y7 = y[n7] ;

  Real_t z0 = z[n0] ;
  Real_t z1 = z[n1] ;
  Real_t z2 = z[n2] ;
  Real_t z3 = z[n3] ;
  Real_t z4 = z[n4] ;
  Real_t z5 = z[n5] ;
  Real_t z6 = z[n6] ;
  Real_t z7 = z[n7] ;

  Real_t xv0 = xd[n0] ;
  Real_t xv1 = xd[n1] ;
  Real_t xv2 = xd[n2] ;
  Real_t xv3 = xd[n3] ;
  Real_t xv4 = xd[n4] ;
  Real_t xv5 = xd[n5] ;
  Real_t xv6 = xd[n6] ;
  Real_t xv7 = xd[n7] ;

  Real_t yv0 = yd[n0] ;
  Real_t yv1 = yd[n1] ;
  Real_t yv2 = yd[n2] ;
  Real_t yv3 = yd[n3] ;
  Real_t yv4 = yd[n4] ;
  Real_t yv5 = yd[n5] ;
  Real_t yv6 = yd[n6] ;
  Real_t yv7 = yd[n7] ;

  Real_t zv0 = zd[n0] ;
  Real_t zv1 = zd[n1] ;
  Real_t zv2 = zd[n2] ;
  Real_t zv3 = zd[n3] ;
  Real_t zv4 = zd[n4] ;
  Real_t zv5 = zd[n5] ;
  Real_t zv6 = zd[n6] ;
  Real_t zv7 = zd[n7] ;

  Real_t vol = volo[i] * vnew[i] ;
  Real_t norm = Real_t(1.0) / ( vol + PTINY ) ;

  Real_t dxj = Real_t(-0.25)*((x0+x1+x5+x4) - (x3+x2+x6+x7)) ;
  Real_t dyj = Real_t(-0.25)*((y0+y1+y5+y4) - (y3+y2+y6+y7)) ;
  Real_t dzj = Real_t(-0.25)*((z0+z1+z5+z4) - (z3+z2+z6+z7)) ;

  Real_t dxi = Real_t( 0.25)*((x1+x2+x6+x5) - (x0+x3+x7+x4)) ;
  Real_t dyi = Real_t( 0.25)*((y1+y2+y6+y5) - (y0+y3+y7+y4)) ;
  Real_t dzi = Real_t( 0.25)*((z1+z2+z6+z5) - (z0+z3+z7+z4)) ;

  Real_t dxk = Real_t( 0.25)*((x4+x5+x6+x7) - (x0+x1+x2+x3)) ;
  Real_t dyk = Real_t( 0.25)*((y4+y5+y6+y7) - (y0+y1+y2+y3)) ;
  Real_t dzk = Real_t( 0.25)*((z4+z5+z6+z7) - (z0+z1+z2+z3)) ;

  /* find delvk and delxk ( i cross j ) */

  ax = dyi*dzj - dzi*dyj ;
  ay = dzi*dxj - dxi*dzj ;
  az = dxi*dyj - dyi*dxj ;

  delx_zeta[i] = vol / sqrt(ax*ax + ay*ay + az*az + PTINY) ;

  ax *= norm ;
  ay *= norm ;
  az *= norm ;

  dxv = Real_t(0.25)*((xv4+xv5+xv6+xv7) - (xv0+xv1+xv2+xv3)) ;
  dyv = Real_t(0.25)*((yv4+yv5+yv6+yv7) - (yv0+yv1+yv2+yv3)) ;
  dzv = Real_t(0.25)*((zv4+zv5+zv6+zv7) - (zv0+zv1+zv2+zv3)) ;

  delv_zeta[i] = ax*dxv + ay*dyv + az*dzv ;

  /* find delxi and delvi ( j cross k ) */

  ax = dyj*dzk - dzj*dyk ;
  ay = dzj*dxk - dxj*dzk ;
  az = dxj*dyk - dyj*dxk ;

  delx_xi[i] = vol / sqrt(ax*ax + ay*ay + az*az + PTINY) ;

  ax *= norm ;
  ay *= norm ;
  az *= norm ;

  dxv = Real_t(0.25)*((xv1+xv2+xv6+xv5) - (xv0+xv3+xv7+xv4)) ;
  dyv = Real_t(0.25)*((yv1+yv2+yv6+yv5) - (yv0+yv3+yv7+yv4)) ;
  dzv = Real_t(0.25)*((zv1+zv2+zv6+zv5) - (zv0+zv3+zv7+zv4)) ;

  delv_xi[i] = ax*dxv + ay*dyv + az*dzv ;

  /* find delxj and delvj ( k cross i ) */

  ax = dyk*dzi - dzk*dyi ;
  ay = dzk*dxi - dxk*dzi ;
  az = dxk*dyi - dyk*dxi ;

  delx_eta[i] = vol / sqrt(ax*ax + ay*ay + az*az + PTINY) ;

  ax *= norm ;
  ay *= norm ;
  az *= norm ;

  dxv = Real_t(-0.25)*((xv0+xv1+xv5+xv4) - (xv3+xv2+xv6+xv7)) ;
  dyv = Real_t(-0.25)*((yv0+yv1+yv5+yv4) - (yv3+yv2+yv6+yv7)) ;
  dzv = Real_t(-0.25)*((zv0+zv1+zv5+zv4) - (zv3+zv2+zv6+zv7)) ;

  delv_eta[i] = ax*dxv + ay*dyv + az*dzv ;
}""".strip()
]
