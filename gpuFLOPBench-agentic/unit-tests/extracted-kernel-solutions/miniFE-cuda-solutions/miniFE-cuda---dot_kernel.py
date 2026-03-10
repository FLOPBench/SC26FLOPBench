solution = [
    r"""template<typename Scalar>
    __global__ void dot_kernel(const MINIFE_LOCAL_ORDINAL n, 
        const Scalar* x, 
        const Scalar* y, 
              Scalar* d) 
    {
      Scalar sum=0;
      for(int idx=blockIdx.x*blockDim.x+threadIdx.x;idx<n;idx+=gridDim.x*blockDim.x) {
        sum+=x[idx]*y[idx];
      }

      //Do a shared memory reduction on the dot product
      __shared__ Scalar red[256];
      red[threadIdx.x]=sum;
#pragma unroll
      for (int n = 128; n > 0; n = n/2) {
        __syncthreads();
        if(threadIdx.x<n)  {sum+=red[threadIdx.x+n]; red[threadIdx.x]=sum;}
      }

      //save partial dot products
      if(threadIdx.x==0) d[blockIdx.x]=sum;
    }""".strip(),
]
