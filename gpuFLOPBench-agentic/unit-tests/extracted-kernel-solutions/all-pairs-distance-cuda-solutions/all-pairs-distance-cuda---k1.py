solution = [
 r"""/*  coalesced GPU implementation of the all-pairs kernel using
    character data types and registers */
__global__ void k1 (const char *data, int *distance) {
  int idx = threadIdx.x;
  int gx = blockIdx.x;
  int gy = blockIdx.y;

  for(int i = 4*idx; i < ATTRIBUTES; i+=THREADS*4) {
    char4 j = *(char4 *)(data + i + ATTRIBUTES*gx);
    char4 k = *(char4 *)(data + i + ATTRIBUTES*gy);

    /* use a local variable (stored in register) to hold intermediate
       values. This reduces writes to global memory */
    char count = 0;

    if(j.x ^ k.x) 
      count++; 
    if(j.y ^ k.y)
      count++;
    if(j.z ^ k.z)
      count++;
    if(j.w ^ k.w)
      count++;

    /* atomic write to global memory */
    atomicAdd(distance + INSTANCES*gx + gy, count);
  }
}
""".strip()
]
