solution = [
 r"""__global__
void PerplexitySearchKernel(
    volatile float* __restrict__ betas,
    volatile float* __restrict__ lower_bound,
    volatile float* __restrict__ upper_bound,
    volatile int*   __restrict__ found,
    const    float* __restrict__ neg_entropy,
    const    float* __restrict__ row_sum,
    const float perplexity_target,  // 50.0f
    const float epsilon,            // 1e-4
    const int num_points)
{
    int tid, is_found;
    float perplexity, neg_ent, sum_P, perplexity_diff, beta, min_beta, max_beta;

    tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_points)
        return;

    neg_ent  = neg_entropy[tid];
    sum_P    = row_sum[tid];
    beta     = betas[tid];
    min_beta = lower_bound[tid];
    max_beta = upper_bound[tid];

    perplexity      = (neg_ent / sum_P) + __logf(sum_P);
    perplexity_diff = perplexity - __logf(perplexity_target);
    is_found        = (perplexity_diff < epsilon && -perplexity_diff < epsilon);
    if (!is_found)
    {
        if (perplexity_diff > 0)
        {
            min_beta = beta;
            beta = (max_beta == FLT_MAX || max_beta == -FLT_MAX) ? beta * 2.0f : (beta + max_beta) / 2.0f;
        }
        else
        {
            max_beta = beta;
            beta = (min_beta == -FLT_MAX || min_beta == FLT_MAX) ? beta / 2.0f : (beta + min_beta) / 2.0f;
        }
        betas[tid] = beta;
        lower_bound[tid] = min_beta;
        upper_bound[tid] = max_beta;
    }
    found[tid] = is_found;
}
""".strip()
]
