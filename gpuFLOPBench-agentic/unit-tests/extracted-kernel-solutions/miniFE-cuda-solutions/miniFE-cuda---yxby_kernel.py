solution = [
    r"""template <typename VectorType> 
    __global__  void yxby_kernel(
        const int n,
        const typename VectorType::ScalarType *xcoefs, 
        const typename VectorType::ScalarType beta,
        typename VectorType::ScalarType *ycoefs)
    {
      int idx=blockIdx.x*blockDim.x+threadIdx.x;
      if (idx<n) ycoefs[idx] = xcoefs[idx] + beta * ycoefs[idx];
    }""".strip(),
]
