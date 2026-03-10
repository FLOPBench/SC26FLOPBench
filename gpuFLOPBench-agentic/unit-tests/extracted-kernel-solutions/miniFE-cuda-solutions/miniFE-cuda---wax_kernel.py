solution = [
    r"""template <typename VectorType> 
    __global__  void wax_kernel(
        const int n,
        const typename VectorType::ScalarType alpha, 
        const typename VectorType::ScalarType *xcoefs, 
        typename VectorType::ScalarType *wcoefs) 
    {

      int idx=blockIdx.x*blockDim.x+threadIdx.x;
      if (idx<n) wcoefs[idx] = alpha * xcoefs[idx];
    }""".strip(),
]
