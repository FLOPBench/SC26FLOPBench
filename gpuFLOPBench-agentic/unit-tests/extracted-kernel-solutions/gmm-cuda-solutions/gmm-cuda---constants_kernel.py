solution = [
r"""__global__ void
constants_kernel(clusters_t* clusters, int num_clusters, int num_dimensions) {

  // compute_constants(clusters,num_clusters,num_dimensions);

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int num_threads = blockDim.x;
  int num_elements = num_dimensions*num_dimensions;

  __shared__ float determinant_arg; // only one thread computes the inverse so we need a shared argument
  __shared__ float sum;
  __shared__ float matrix[NUM_DIMENSIONS*NUM_DIMENSIONS];

  float log_determinant;


  // Invert the matrix for every cluster

  // Copy the R matrix into shared memory for doing the matrix inversion
  for(int i=tid; i<num_elements; i+= num_threads ) {
    matrix[i] = clusters->R[bid*num_dimensions*num_dimensions+i];
  }

  __syncthreads(); 
#if DIAG_ONLY
  if(tid == 0) { 
    determinant_arg = 1.0f;
    for(int i=0; i < num_dimensions; i++) {
      determinant_arg *= matrix[i*num_dimensions+i];
      matrix[i*num_dimensions+i] = 1.0f / matrix[i*num_dimensions+i];
    }
    determinant_arg = logf(determinant_arg);
  }
#else 
  invert(matrix,num_dimensions,&determinant_arg);
#endif
  __syncthreads(); 
  log_determinant = determinant_arg;

  // Copy the matrx from shared memory back into the cluster memory
  for(int i=tid; i<num_elements; i+= num_threads) {
    clusters->Rinv[bid*num_dimensions*num_dimensions+i] = matrix[i];
  }

  __syncthreads();

  // Compute the constant
  // Equivilent to: log(1/((2*PI)^(M/2)*det(R)^(1/2)))
  // This constant is used in all E-step likelihood calculations
  if(tid == 0) {
    clusters->constant[bid] = -num_dimensions*0.5f*logf(2.0f*PI) - 0.5f*log_determinant;
  }

  __syncthreads();

  if(bid == 0) {
    // compute_pi(clusters,num_clusters);

    if(tid == 0) {
      sum = 0.0;
      for(int i=0; i<num_clusters; i++) {
        sum += clusters->N[i];
      }
    }

    __syncthreads();

    for(int i = tid; i < num_clusters; i += num_threads) {
      if(clusters->N[i] < 0.5f) {
        clusters->pi[tid] = 1e-10;
      } else {
        clusters->pi[tid] = clusters->N[i] / sum;
      }
    }
  }
}""".strip()
]
