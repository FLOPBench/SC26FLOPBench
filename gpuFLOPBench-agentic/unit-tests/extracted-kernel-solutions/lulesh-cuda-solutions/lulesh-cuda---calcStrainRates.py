solution = [
r"""__global__ void calcStrainRates(
    Real_t *__restrict__ dxx,
    Real_t *__restrict__ dyy,
    Real_t *__restrict__ dzz,
    const Real_t *__restrict__ vnew,
    Real_t *__restrict__ vdov,
    int *__restrict__ vol_error,
    const Index_t numElem )
{
  Index_t k = blockDim.x*blockIdx.x+threadIdx.x;
  if (k >= numElem) return;

  // calc strain rate and apply as constraint (only done in FB element)
  Real_t vvdov = dxx[k] + dyy[k] + dzz[k] ;
  Real_t vdovthird = vvdov/Real_t(3.0) ;

  // make the rate of deformation tensor deviatoric
  vdov[k] = vvdov;
  dxx[k] -= vdovthird ;  //LG:   why to update dxx?  it is deallocated right after
  dyy[k] -= vdovthird ;
  dzz[k] -= vdovthird ;

  // See if any volumes are negative, and take appropriate action.
  if (vnew[k] <= Real_t(0.0))
  {
    vol_error[0] = k;
  }
}""".strip()
]
