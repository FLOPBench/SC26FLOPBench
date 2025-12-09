solution = [
r"""__global__ void
mstep_covariance1(float* fcs_data, clusters_t* clusters, int num_dimensions, int num_clusters, int num_events) {
  int tid = threadIdx.x; // easier variable name for our thread ID

  // Determine what row,col this matrix is handling, also handles the symmetric element
  int row,col,c;
  compute_row_col(num_dimensions, &row, &col);
  //row = blockIdx.y / num_dimensions;
  //col = blockIdx.y % num_dimensions;

  __syncthreads();

  c = blockIdx.x; // Determines what cluster this block is handling    

  int matrix_index = row * num_dimensions + col;

#if DIAG_ONLY
  if(row != col) {
    clusters->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0;
    matrix_index = col*num_dimensions+row;
    clusters->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0;
    return;
  }
#endif 

  // Store the means in shared memory to speed up the covariance computations
  __shared__ float means[NUM_DIMENSIONS];
  // copy the means for this cluster into shared memory
  if(tid < num_dimensions) {
    means[tid] = clusters->means[c*num_dimensions+tid];
  }

  // Sync to wait for all params to be loaded to shared memory
  __syncthreads();

  __shared__ float temp_sums[NUM_THREADS_MSTEP];

  float cov_sum = 0.0;

  for(int event=tid; event < num_events; event+=NUM_THREADS_MSTEP) {
    cov_sum += (fcs_data[row*num_events+event]-means[row])*
      (fcs_data[col*num_events+event]-means[col])*clusters->memberships[c*num_events+event]; 
  }
  temp_sums[tid] = cov_sum;

  __syncthreads();

  cov_sum = parallelSum(temp_sums,NUM_THREADS_MSTEP);

  if(tid == 0) {
    clusters->R[c*num_dimensions*num_dimensions+matrix_index] = cov_sum;
    // Set the symmetric value
    matrix_index = col*num_dimensions+row;
    clusters->R[c*num_dimensions*num_dimensions+matrix_index] = cov_sum;

    // Regularize matrix - adds some variance to the diagonal elements
    // Helps keep covariance matrix non-singular (so it can be inverted)
    // The amount added is scaled down based on COVARIANCE_DYNAMIC_RANGE constant defined at top of this file
    if(row == col) {
      clusters->R[c*num_dimensions*num_dimensions+matrix_index] += clusters->avgvar[c];
    }
  }
}""".strip()
]
