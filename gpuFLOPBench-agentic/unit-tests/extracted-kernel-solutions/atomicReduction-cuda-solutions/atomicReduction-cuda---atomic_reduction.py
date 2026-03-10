solution = [
 r"""__global__ void atomic_reduction(int *in, int* out, int arrayLength) {
  int sum=0;
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  for(int i= idx;i<arrayLength;i+=blockDim.x*gridDim.x) {
    sum+=in[i];
  }
  atomicAdd(out,sum);
}
""".strip()
]
