solution = [
r"""__global__ void calcPositionForNodes (
    Real_t *__restrict__ x,
    Real_t *__restrict__ y,
    Real_t *__restrict__ z,
    const Real_t *__restrict__ xd,
    const Real_t *__restrict__ yd,
    const Real_t *__restrict__ zd,
    const Real_t deltaTime,
    const Index_t numNode) 
{
  Index_t i = blockDim.x*blockIdx.x+threadIdx.x;
  if (i >= numNode) return;
  x[i] += xd[i] * deltaTime;
  y[i] += yd[i] * deltaTime;
  z[i] += zd[i] * deltaTime;
}""".strip()
]
