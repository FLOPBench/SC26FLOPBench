solution = [
    r"""template <typename VectorType> 
    __global__  void dyx_kernel(
        const int n,
        const typename VectorType::ScalarType *xcoefs, 
        typename VectorType::ScalarType *ycoefs) 
    {

      int idx=blockIdx.x*blockDim.x+threadIdx.x;
      if (idx<n) ycoefs[idx] += xcoefs[idx];
    }""".strip(),
]
