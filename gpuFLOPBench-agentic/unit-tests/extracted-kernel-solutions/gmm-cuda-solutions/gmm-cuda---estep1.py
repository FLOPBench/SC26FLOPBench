solution = [
r"""__global__ void
estep1(float* data, clusters_t* clusters, int num_dimensions, int num_events) {

  // Cached cluster parameters
  __shared__ float means[NUM_DIMENSIONS];
  __shared__ float Rinv[NUM_DIMENSIONS*NUM_DIMENSIONS];
  float cluster_pi;
  float constant;
  const unsigned int tid = threadIdx.x;

  int start_index;
  int end_index;

  int c = blockIdx.x;

  compute_indices(num_events,&start_index,&end_index);

  float like;

  // This loop computes the expectation of every event into every cluster
  //
  // P(k|n) = L(x_n|mu_k,R_k)*P(k) / P(x_n)
  //
  // Compute log-likelihood for every cluster for each event
  // L = constant*exp(-0.5*(x-mu)*Rinv*(x-mu))
  // log_L = log_constant - 0.5*(x-u)*Rinv*(x-mu)
  // the constant stored in clusters[c].constant is already the log of the constant

  // copy the means for this cluster into shared memory
  if(tid < num_dimensions) {
    means[tid] = clusters->means[c*num_dimensions+tid];
  }

  // copy the covariance inverse into shared memory
  for(int i=tid; i < num_dimensions*num_dimensions; i+= NUM_THREADS_ESTEP) {
    Rinv[i] = clusters->Rinv[c*num_dimensions*num_dimensions+i]; 
  }

  cluster_pi = clusters->pi[c];
  constant = clusters->constant[c];

  // Sync to wait for all params to be loaded to shared memory
  __syncthreads();

  for(int event=start_index; event<end_index; event += NUM_THREADS_ESTEP) {
    like = 0.0f;
    // this does the loglikelihood calculation
#if DIAG_ONLY
    for(int j=0; j<num_dimensions; j++) {
      like += (data[j*num_events+event]-means[j]) * (data[j*num_events+event]-means[j]) * Rinv[j*num_dimensions+j];
    }
#else
    for(int i=0; i<num_dimensions; i++) {
      for(int j=0; j<num_dimensions; j++) {
        like += (data[i*num_events+event]-means[i]) * (data[j*num_events+event]-means[j]) * Rinv[i*num_dimensions+j];
      }
    }
#endif
    // numerator of the E-step probability computation
    clusters->memberships[c*num_events+event] = -0.5f * like + constant + logf(cluster_pi);
  }
}""".strip()
]
