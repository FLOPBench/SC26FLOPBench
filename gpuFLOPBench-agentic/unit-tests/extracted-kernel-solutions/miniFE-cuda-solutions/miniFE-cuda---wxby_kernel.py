solution = [
    r"""template <typename VectorType> 
    __global__  void wxby_kernel(
        const int n,
        const typename VectorType::ScalarType *xcoefs, 
        const typename VectorType::ScalarType beta,
        const typename VectorType::ScalarType *ycoefs, 
        typename VectorType::ScalarType *wcoefs) 
    {
      int idx=blockIdx.x*blockDim.x+threadIdx.x;
      if (idx<n) wcoefs[idx] = xcoefs[idx] + beta * ycoefs[idx];
    }""".strip(),
]
