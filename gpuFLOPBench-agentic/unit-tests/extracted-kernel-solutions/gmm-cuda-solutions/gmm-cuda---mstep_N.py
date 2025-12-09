solution = [
r"""__global__ void
mstep_N(clusters_t* clusters, int num_dimensions, int num_clusters, int num_events) {

  int tid = threadIdx.x;
  int num_threads = blockDim.x;
  int c = blockIdx.x;


  // Need to store the sum computed by each thread so in the end
  // a single thread can reduce to get the final sum
  __shared__ float temp_sums[NUM_THREADS_MSTEP];

  // Compute new N
  float sum = 0.0f;
  // Break all the events accross the threads, add up probabilities
  for(int event=tid; event < num_events; event += num_threads) {
    sum += clusters->memberships[c*num_events+event];
  }
  temp_sums[tid] = sum;

  __syncthreads();

  sum = parallelSum(temp_sums,NUM_THREADS_MSTEP);
  if(tid == 0) {
    clusters->N[c] = sum;
    clusters->pi[c] = sum;
  }

  // Let the first thread add up all the intermediate sums
  // Could do a parallel reduction...doubt it's really worth it for so few elements though
  /*if(tid == 0) {
    clusters->N[c] = 0.0;
    for(int j=0; j<num_threads; j++) {
    clusters->N[c] += temp_sums[j];
    }
  //printf("clusters[%d].N = %f\n",c,clusters[c].N);

  // Set PI to the # of expected items, and then normalize it later
  clusters->pi[c] = clusters->N[c];
  }*/
}""".strip()
]
