solution = [
 r"""__global__ void atomic_reduction_v2(int *in, int* out, int arrayLength) {
  int sum=0;
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  for(int i= idx*2;i<arrayLength;i+=blockDim.x*gridDim.x*2) {
    sum+=in[i] + in[i+1];
  }
  atomicAdd(out,sum);
}
""".strip()
]
