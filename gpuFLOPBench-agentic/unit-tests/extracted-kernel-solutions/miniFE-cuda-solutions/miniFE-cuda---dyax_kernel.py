solution = [
    r"""template <typename VectorType> 
    __global__  void dyax_kernel(
        const int n,
        const typename VectorType::ScalarType alpha, 
        const typename VectorType::ScalarType *xcoefs, 
        typename VectorType::ScalarType *ycoefs) 
    {

      int idx=blockIdx.x*blockDim.x+threadIdx.x;
      if (idx<n) ycoefs[idx] += alpha * xcoefs[idx];
    }""".strip(),
]
