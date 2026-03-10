solution = [
r"""__global__ void fill_sig(
    Real_t *__restrict__ sigxx,
    Real_t *__restrict__ sigyy,
    Real_t *__restrict__ sigzz,
    const Real_t *__restrict__ p,
    const Real_t *__restrict__ q,
    const Index_t numElem )
{
  Index_t i = blockDim.x*blockIdx.x+threadIdx.x;
  if (i >= numElem) return;
  sigxx[i] = sigyy[i] = sigzz[i] = - p[i] - q[i] ;
}""".strip()
]
