solution = [
r"""__global__ void acc_final_force (
    const Real_t *__restrict__ fx_elem,
    const Real_t *__restrict__ fy_elem,
    const Real_t *__restrict__ fz_elem,
    Real_t *__restrict__ fx,
    Real_t *__restrict__ fy,
    Real_t *__restrict__ fz,
    const Index_t *__restrict__ nodeElemStart,
    const Index_t *__restrict__ nodeElemCornerList,
    const Index_t numNode) 
{
  Index_t gnode = blockDim.x*blockIdx.x+threadIdx.x;
  if (gnode >= numNode) return;
  // element count
  const Index_t count = nodeElemStart[gnode+1] - nodeElemStart[gnode];//domain.nodeElemCount(gnode) ;
  // list of all corners
  const Index_t *cornerList = nodeElemCornerList + nodeElemStart[gnode];//domain.nodeElemCornerList(gnode) ;
  Real_t fx_tmp = Real_t(0.0) ;
  Real_t fy_tmp = Real_t(0.0) ;
  Real_t fz_tmp = Real_t(0.0) ;
  for (Index_t i=0 ; i < count ; ++i) {
    Index_t elem = cornerList[i] ;
    fx_tmp += fx_elem[elem] ;
    fy_tmp += fy_elem[elem] ;
    fz_tmp += fz_elem[elem] ;
  }
  fx[gnode] = fx_tmp ;
  fy[gnode] = fy_tmp ;
  fz[gnode] = fz_tmp ;
}""".strip()
]
