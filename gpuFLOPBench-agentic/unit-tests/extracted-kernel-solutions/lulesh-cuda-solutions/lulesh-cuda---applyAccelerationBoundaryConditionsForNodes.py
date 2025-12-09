solution = [
r"""__global__ void applyAccelerationBoundaryConditionsForNodes (
    const Index_t *__restrict__ symmX,
    const Index_t *__restrict__ symmY,
    const Index_t *__restrict__ symmZ,
    Real_t *__restrict__ xdd,
    Real_t *__restrict__ ydd,
    Real_t *__restrict__ zdd,
    const Index_t s1,
    const Index_t s2,
    const Index_t s3,
    const Index_t numNodeBC ) 
{
  Index_t i = blockDim.x*blockIdx.x+threadIdx.x;
  if (i >= numNodeBC) return;
  if (s1 == 0) 
    xdd[symmX[i]] = Real_t(0.0);
  if (s2 == 0) ydd[symmY[i]] = Real_t(0.0);
  if (s3 == 0) zdd[symmZ[i]] = Real_t(0.0);
}""".strip()
]
