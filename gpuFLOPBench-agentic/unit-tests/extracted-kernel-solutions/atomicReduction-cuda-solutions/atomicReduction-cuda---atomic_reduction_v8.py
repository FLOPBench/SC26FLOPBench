solution = [
 r"""__global__ void atomic_reduction_v8(int *in, int* out, int arrayLength) {
  int sum=0;
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  for(int i= idx*8;i<arrayLength;i+=blockDim.x*gridDim.x*8) {
    sum+=in[i] + in[i+1] + in[i+2] + in[i+3] +in[i+4] +in[i+5] +in[i+6] +in[i+7];
  }
  atomicAdd(out,sum);
}
""".strip()
]
