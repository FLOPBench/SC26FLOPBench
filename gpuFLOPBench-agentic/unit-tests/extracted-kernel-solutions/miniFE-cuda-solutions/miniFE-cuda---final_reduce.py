solution = [
    r"""template<typename Scalar>
    __global__ void final_reduce(Scalar *d) {
      Scalar sum = d[threadIdx.x];
      __shared__ Scalar red[256];

      red[threadIdx.x]=sum;
#pragma unroll
      for (int n = 128; n > 0; n = n/2) {
        __syncthreads();
        if(threadIdx.x<n)  {sum+=red[threadIdx.x+n]; red[threadIdx.x]=sum;}
      }
      //save final dot product at the front
      if(threadIdx.x==0) d[0]=sum;
    }""".strip(),
]
