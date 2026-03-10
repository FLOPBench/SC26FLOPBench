solution = [
 r"""__global__ void atomic_reduction_v4(int *in, int* out, int arrayLength) {
  int sum=0;
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  for(int i= idx*4;i<arrayLength;i+=blockDim.x*gridDim.x*4) {
    sum+=in[i] + in[i+1] + in[i+2] + in[i+3];
  }
  atomicAdd(out,sum);
}
""".strip()
]
