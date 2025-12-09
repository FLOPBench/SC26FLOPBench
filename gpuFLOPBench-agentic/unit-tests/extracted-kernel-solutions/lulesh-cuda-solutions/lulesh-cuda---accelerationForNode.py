solution = [
r"""__global__  void accelerationForNode (
    const Real_t *__restrict__ fx,
    const Real_t *__restrict__ fy,
    const Real_t *__restrict__ fz,
    const Real_t *__restrict__ nodalMass,
    Real_t *__restrict__ xdd,
    Real_t *__restrict__ ydd,
    Real_t *__restrict__ zdd,
    const Index_t numNode)
{
  Index_t i = blockDim.x*blockIdx.x+threadIdx.x;
  if (i >= numNode) return;
  Real_t one_over_nMass = Real_t(1.) / nodalMass[i];
  xdd[i] = fx[i] * one_over_nMass;
  ydd[i] = fy[i] * one_over_nMass;
  zdd[i] = fz[i] * one_over_nMass;
}""".strip()
]
