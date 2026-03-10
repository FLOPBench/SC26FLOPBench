solution = [
r"""__global__ void
estep2(float* fcs_data, clusters_t* clusters, int num_dimensions, int num_clusters, int num_events, float* likelihood) {
  float temp;
  float thread_likelihood = 0.0f;
  __shared__ float total_likelihoods[NUM_THREADS_ESTEP];
  float max_likelihood;
  float denominator_sum;

  // Break up the events evenly between the blocks
  int num_pixels_per_block = num_events / gridDim.x;
  // Make sure the events being accessed by the block are aligned to a multiple of 16
  num_pixels_per_block = num_pixels_per_block - (num_pixels_per_block % 16);
  int tid = threadIdx.x;

  int start_index;
  int end_index;
  start_index = blockIdx.x * num_pixels_per_block + tid;

  // Last block will handle the leftover events
  if(blockIdx.x == gridDim.x-1) {
    end_index = num_events;
  } else {
    end_index = (blockIdx.x+1) * num_pixels_per_block;
  }

  total_likelihoods[tid] = 0.0;

  // P(x_n) = sum of likelihoods weighted by P(k) (their probability, cluster[c].pi)
  //  log(a+b) != log(a) + log(b) so we need to do the log of the sum of the exponentials

  //  For the sake of numerical stability, we first find the max and scale the values
  //  That way, the maximum value ever going into the exp function is 0 and we avoid overflow

  //  log-sum-exp formula:
  //  log(sum(exp(x_i)) = max(z) + log(sum(exp(z_i-max(z))))
  for(int pixel=start_index; pixel<end_index; pixel += NUM_THREADS_ESTEP) {
    // find the maximum likelihood for this event
    max_likelihood = clusters->memberships[pixel];
    for(int c=1; c<num_clusters; c++) {
      max_likelihood = fmaxf(max_likelihood,clusters->memberships[c*num_events+pixel]);
    }

    // Compute P(x_n), the denominator of the probability (sum of weighted likelihoods)
    denominator_sum = 0.0;
    for(int c=0; c<num_clusters; c++) {
      temp = expf(clusters->memberships[c*num_events+pixel]-max_likelihood);
      denominator_sum += temp;
    }
    denominator_sum = max_likelihood + logf(denominator_sum);
    thread_likelihood += denominator_sum;

    // Divide by denominator, also effectively normalize probabilities
    // exp(log(p) - log(denom)) == p / denom
    for(int c=0; c<num_clusters; c++) {
      clusters->memberships[c*num_events+pixel] = expf(clusters->memberships[c*num_events+pixel] - denominator_sum);
      //printf("Probability that pixel #%d is in cluster #%d: %f\n",pixel,c,clusters->memberships[c*num_events+pixel]);
    }
  }

  total_likelihoods[tid] = thread_likelihood;
  __syncthreads();

  temp = parallelSum(total_likelihoods,NUM_THREADS_ESTEP);
  if(tid == 0) {
    likelihood[blockIdx.x] = temp;
  }
}""".strip()
]
