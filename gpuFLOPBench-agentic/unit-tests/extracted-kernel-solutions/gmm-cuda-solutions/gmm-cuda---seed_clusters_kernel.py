solution = [
r"""__global__ void
seed_clusters_kernel( const float* fcs_data, 
    clusters_t* clusters, 
    const int num_dimensions, 
    const int num_clusters, 
    const int num_events) 
{
  int tid = threadIdx.x;
  int num_threads = blockDim.x;
  int row, col;
  float seed;

  // Number of elements in the covariance matrix
  int num_elements = num_dimensions*num_dimensions; 

  // shared memory
  __shared__ float means[NUM_DIMENSIONS];
  __shared__ float avgvar;
  __shared__ float variances[NUM_DIMENSIONS];
  __shared__ float total_variance;

  // Compute the means
  // mvtmeans(fcs_data, num_dimensions, num_events, means);

  if(tid < num_dimensions) {
    means[tid] = 0.0;

    // Sum up all the values for each dimension
    for(int i = 0; i < num_events; i++) {
      means[tid] += fcs_data[i*num_dimensions+tid];
    }

    // Divide by the # of elements to get the average
    means[tid] /= (float) num_events;
  }

  __syncthreads();

  // Compute the average variance
  // averageVariance(fcs_data, means, num_dimensions, num_events, &avgvar);

  // Compute average variance for each dimension
  if(tid < num_dimensions) {
    variances[tid] = 0.0;
    // Sum up all the variance
    for(int i = 0; i < num_events; i++) {
      // variance = (data - mean)^2
      variances[tid] += (fcs_data[i*num_dimensions + tid])*(fcs_data[i*num_dimensions + tid]);
    }
    variances[tid] /= (float) num_events;
    variances[tid] -= means[tid]*means[tid];
  }

  __syncthreads();

  if(tid == 0) {
    total_variance = 0.0;
    for(int i=0; i<num_dimensions;i++) {
      total_variance += variances[i];
    }
    avgvar = total_variance / (float) num_dimensions;
  }

  __syncthreads();

  if(num_clusters > 1) {
    seed = (num_events-1.0f)/(num_clusters-1.0f);
  } else {
    seed = 0.0;
  }

  // Seed the pi, means, and covariances for every cluster
  for(int c=0; c < num_clusters; c++) {
    if(tid < num_dimensions) {
      clusters->means[c*num_dimensions+tid] = fcs_data[((int)(c*seed))*num_dimensions+tid];
    }

    for(int i=tid; i < num_elements; i+= num_threads) {
      // Add the average variance divided by a constant, this keeps the cov matrix from becoming singular
      row = (i) / num_dimensions;
      col = (i) % num_dimensions;

      if(row == col) {
        clusters->R[c*num_dimensions*num_dimensions+i] = 1.0f;
      } else {
        clusters->R[c*num_dimensions*num_dimensions+i] = 0.0f;
      }
    }
    if(tid == 0) {
      clusters->pi[c] = 1.0f/((float)num_clusters);
      clusters->N[c] = ((float) num_events) / ((float)num_clusters);
      clusters->avgvar[c] = avgvar / COVARIANCE_DYNAMIC_RANGE;
    }
  }
}""".strip()
]
