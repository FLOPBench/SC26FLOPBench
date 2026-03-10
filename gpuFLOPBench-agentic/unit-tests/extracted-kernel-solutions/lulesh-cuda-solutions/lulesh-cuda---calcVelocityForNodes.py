solution = [
r"""__global__ void  calcVelocityForNodes (
    Real_t *__restrict__ xd,
    Real_t *__restrict__ yd,
    Real_t *__restrict__ zd,
    const Real_t *__restrict__ xdd,
    const Real_t *__restrict__ ydd,
    const Real_t *__restrict__ zdd,
    const Real_t deltaTime,
    const Real_t u_cut,
    const Index_t numNode )
{
  Index_t i = blockDim.x*blockIdx.x+threadIdx.x;
  if (i >= numNode) return;

  Real_t xdtmp = xd[i] + xdd[i] * deltaTime;
  // FABS is not compiled with target regions in mind
  // To get around this, compute the absolute value manually:
  // if( xdtmp > Real_t(0.0) && xdtmp < u_cut || Real_t(-1.0) * xdtmp < u_cut)
  if( fabs(xdtmp) < u_cut ) xdtmp = Real_t(0.0);
  xd[i] = xdtmp ;

  Real_t ydtmp = yd[i] + ydd[i] * deltaTime;
  if( fabs(ydtmp) < u_cut ) ydtmp = Real_t(0.0);
  yd[i] = ydtmp ;

  Real_t zdtmp = zd[i] + zdd[i] * deltaTime;
  if( fabs(zdtmp) < u_cut ) zdtmp = Real_t(0.0);
  zd[i] = zdtmp ;
}""".strip()
]
