solution = [
r"""__global__ void
mstep_means(float* fcs_data, clusters_t* clusters, int num_dimensions, int num_clusters, int num_events) {
  // One block per cluster, per dimension:  (M x D) grid of blocks
  int tid = threadIdx.x;
  int num_threads = blockDim.x;
  int c = blockIdx.x; // cluster number
  int d = blockIdx.y; // dimension number

  __shared__ float temp_sum[NUM_THREADS_MSTEP];
  float sum = 0.0f;

  for(int event=tid; event < num_events; event+= num_threads) {
    sum += fcs_data[d*num_events+event]*clusters->memberships[c*num_events+event];
  }
  temp_sum[tid] = sum;

  __syncthreads();

  // Reduce partial sums
  sum = parallelSum(temp_sum,NUM_THREADS_MSTEP);
  if(tid == 0) {
    clusters->means[c*num_dimensions+d] = sum;
  }

  /*if(tid == 0) {
    for(int i=1; i < num_threads; i++) {
    temp_sum[0] += temp_sum[i];
    }
    clusters->means[c*num_dimensions+d] = temp_sum[0];
  //clusters->means[c*num_dimensions+d] = temp_sum[0] / clusters->N[c];
  }*/
}""".strip()
]
