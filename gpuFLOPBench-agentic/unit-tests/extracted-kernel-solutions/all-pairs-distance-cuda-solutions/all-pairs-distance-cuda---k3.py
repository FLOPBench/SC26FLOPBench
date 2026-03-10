solution = [
 r"""/*  coalesced GPU implementation of the all-pairs kernel using
    character data types, registers, and CUB block reduction */
__global__ void k3 (const char *data, int *distance) {
  int idx = threadIdx.x;
  int gx = blockIdx.x;
  int gy = blockIdx.y;

  typedef cub::BlockReduce<int, THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int dist = 0;

  for(int i = idx*4; i < ATTRIBUTES; i+=THREADS*4) {
    char4 j = *(char4 *)(data + i + ATTRIBUTES*gx);
    char4 k = *(char4 *)(data + i + ATTRIBUTES*gy);
    char count = 0;

    if(j.x ^ k.x) 
      count++;
    if(j.y ^ k.y)
      count++;
    if(j.z ^ k.z)
      count++;
    if(j.w ^ k.w)
      count++;

    dist += count;
  }

  int sum = BlockReduce(temp_storage).Sum(dist);

  if(idx == 0) {
    distance[INSTANCES*gy + gx] = sum;
  }
}
""".strip()
]
