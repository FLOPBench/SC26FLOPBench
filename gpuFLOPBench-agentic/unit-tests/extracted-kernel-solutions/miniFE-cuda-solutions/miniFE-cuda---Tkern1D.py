solution = [
    r"""template <class WDP>
__global__ void
Tkern1D(int length, WDP wd, int stride)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  while(i < length) {
    wd(i);
    i += stride;
  }
}""".strip(),
]
