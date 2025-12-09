solution = [
r"""__global__ void
mstep_covariance2(float* fcs_data, clusters_t* clusters, int num_dimensions, int num_clusters, int num_events) {
  int tid = threadIdx.x; // easier variable name for our thread ID

  // Determine what row,col this matrix is handling, also handles the symmetric element
  int row,col,c1;
  compute_row_col(num_dimensions, &row, &col);

  __syncthreads();

  c1 = blockIdx.x * NUM_CLUSTERS_PER_BLOCK; // Determines what cluster this block is handling    

#if DIAG_ONLY
  if(row != col) {
    clusters->R[c*num_dimensions*num_dimensions+row*num_dimensions+col] = 0.0f;
    clusters->R[c*num_dimensions*num_dimensions+col*num_dimensions+row] = 0.0f;
    return;
  }
#endif 

  // Store the means in shared memory to speed up the covariance computations
  __shared__ float means_row[NUM_CLUSTERS_PER_BLOCK];
  __shared__ float means_col[NUM_CLUSTERS_PER_BLOCK];

  //if(tid < NUM_CLUSTERS_PER_BLOCK) {  
  if ( (tid < min(num_clusters, NUM_CLUSTERS_PER_BLOCK))  // c1 = 0
      && (c1+tid < num_clusters)) { 
    means_row[tid] = clusters->means[(c1+tid)*num_dimensions+row];
    means_col[tid] = clusters->means[(c1+tid)*num_dimensions+col];
  }

  // Sync to wait for all params to be loaded to shared memory
  __syncthreads();

  // 256 * 6
  __shared__ float temp_sums[NUM_THREADS_MSTEP*NUM_CLUSTERS_PER_BLOCK];

  float cov_sum1 = 0.0f;
  float cov_sum2 = 0.0f;
  float cov_sum3 = 0.0f;
  float cov_sum4 = 0.0f;
  float cov_sum5 = 0.0f;
  float cov_sum6 = 0.0f;
  float val1,val2;

  for(int c=0; c < NUM_CLUSTERS_PER_BLOCK; c++) {
    temp_sums[c*NUM_THREADS_MSTEP+tid] = 0.0;
  } 

  for(int event=tid; event < num_events; event+=NUM_THREADS_MSTEP) {
    val1 = fcs_data[row*num_events+event];
    val2 = fcs_data[col*num_events+event];
    cov_sum1 += (val1-means_row[0])*(val2-means_col[0])*clusters->memberships[c1*num_events+event]; 
    cov_sum2 += (val1-means_row[1])*(val2-means_col[1])*clusters->memberships[(c1+1)*num_events+event]; 
    cov_sum3 += (val1-means_row[2])*(val2-means_col[2])*clusters->memberships[(c1+2)*num_events+event]; 
    cov_sum4 += (val1-means_row[3])*(val2-means_col[3])*clusters->memberships[(c1+3)*num_events+event]; 
    cov_sum5 += (val1-means_row[4])*(val2-means_col[4])*clusters->memberships[(c1+4)*num_events+event]; 
    cov_sum6 += (val1-means_row[5])*(val2-means_col[5])*clusters->memberships[(c1+5)*num_events+event]; 
  }
  temp_sums[0*NUM_THREADS_MSTEP+tid] = cov_sum1;
  temp_sums[1*NUM_THREADS_MSTEP+tid] = cov_sum2;
  temp_sums[2*NUM_THREADS_MSTEP+tid] = cov_sum3;
  temp_sums[3*NUM_THREADS_MSTEP+tid] = cov_sum4;
  temp_sums[4*NUM_THREADS_MSTEP+tid] = cov_sum5;
  temp_sums[5*NUM_THREADS_MSTEP+tid] = cov_sum6;

  __syncthreads();

  for(int c=0; c < NUM_CLUSTERS_PER_BLOCK; c++) {
    temp_sums[c*NUM_THREADS_MSTEP+tid] = parallelSum(&temp_sums[c*NUM_THREADS_MSTEP],NUM_THREADS_MSTEP);
    __syncthreads();
  }

  if(tid == 0) {
    for(int c=0; c < NUM_CLUSTERS_PER_BLOCK && (c+c1) < num_clusters; c++) {
      int offset = (c+c1)*num_dimensions*num_dimensions;
      cov_sum1 = temp_sums[c*NUM_THREADS_MSTEP];
      clusters->R[offset+row*num_dimensions+col] = cov_sum1;
      // Set the symmetric value
      clusters->R[offset+col*num_dimensions+row] = cov_sum1;

      // Regularize matrix - adds some variance to the diagonal elements
      // Helps keep covariance matrix non-singular (so it can be inverted)
      // The amount added is scaled down based on COVARIANCE_DYNAMIC_RANGE constant defined in gaussian.h
      if(row == col) {
        clusters->R[offset+row*num_dimensions+col] += clusters->avgvar[c+c1];
      }
    }
  }
}""".strip()
]
