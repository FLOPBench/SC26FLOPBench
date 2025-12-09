solution = [
 r"""/*  coalesced GPU implementation of the all-pairs kernel using
    character data types, registers, and shared memory */
__global__ void k2 (const char *data, int *distance) {
  int idx = threadIdx.x;
  int gx = blockIdx.x;
  int gy = blockIdx.y;

  /* Shared memory is the other major memory (other than registers and
     global). It is used to store values between multiple threads. In
     particular, the shared memory access is defined by the __shared__
     attribute and it is a special area of memory on the GPU
     itself. Because the memory is on the chip, it is a lot faster
     than global memory. Multiple threads can still access it, though,
     provided they are in the same block.
   */
  __shared__ int dist[THREADS];

  /* each thread initializes its own location of the shared array */ 
  dist[idx] = 0;

  /* At this point, the threads must be synchronized to ensure that
     the shared array is fully initialized. */
  __syncthreads();

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

    /* Increment shared array */
    dist[idx] += count;
  }

  /* Synchronize threads to make sure all have completed their updates
     of the shared array. Since the distances for each thread are read
     by thread 0 below, this must be ensured. Above, it was not
     necessary because each thread was accessing its own memory
   */
  __syncthreads();

  /* Reduction: Thread 0 will add the value of all other threads to
     its own */ 
  if(idx == 0) {
    for(int i = 1; i < THREADS; i++) {
      dist[0] += dist[i];
    }

    /* Thread 0 will then write the output to global memory. Note that
       this does not need to be performed atomically, because only one
       thread per block is writing to global memory, and each block
       corresponds to a unique memory address. 
     */
    distance[INSTANCES*gy + gx] = dist[0];
  }
}
""".strip()
]
