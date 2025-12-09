solution = [
 r"""__global__ void IntegrationKernel(
    volatile float* __restrict__ points,        // num_points * 2
    volatile float* __restrict__ attr_forces,   // num_points * 2
    volatile float* __restrict__ rep_forces,    // num_points * 2
    volatile float* __restrict__ gains,         // num_points * 2
    volatile float* __restrict__ old_forces,    // num_points * 2
    const float eta,
    const float normalization,
    const float momentum,
    const float exaggeration,
    const int   num_points)
{
    int tid;
    float dx, dy, ux, uy, gx, gy;

    tid = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tid < num_points) {
        ux = old_forces[tid];
        uy = old_forces[tid + num_points];
        gx = gains[tid];
        gy = gains[tid + num_points];
        dx = exaggeration * attr_forces[tid]              - (rep_forces[tid]              / normalization);
        dy = exaggeration * attr_forces[tid + num_points] - (rep_forces[tid + num_points] / normalization);

        gx = (signbit(dx) != signbit(ux)) ? gx + 0.2f : gx * 0.8f;
        gy = (signbit(dy) != signbit(uy)) ? gy + 0.2f : gy * 0.8f;
        gx = (gx < 0.01f) ? 0.01f : gx;
        gy = (gy < 0.01f) ? 0.01f : gy;

        ux = momentum * ux - eta * gx * dx;
        uy = momentum * uy - eta * gy * dy;

        points[tid] += ux;
        points[tid + num_points] += uy;

        attr_forces[tid]              = 0.0f;
        attr_forces[tid + num_points] = 0.0f;
        rep_forces[tid]               = 0.0f;
        rep_forces[tid + num_points]  = 0.0f;
        old_forces[tid]               = ux;
        old_forces[tid + num_points]  = uy;
        gains[tid]                    = gx;
        gains[tid + num_points]       = gy;
    }
}
""".strip()
]
