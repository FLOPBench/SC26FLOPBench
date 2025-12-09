EXPECTED_TREE = (
    "tsne-cuda/\n"
    "  data/\n"
    "    cifar10_faissed/\n"
    "      distances\n"
    "      indices\n"
    "    mnist_faissed/\n"
    "      distances\n"
    "      indices\n"
    "  apply_forces.cu\n"
    "  apply_forces.h\n"
    "  attr_forces.cu\n"
    "  attr_forces.h\n"
    "  common.h\n"
    "  cuda_utils.cu\n"
    "  cuda_utils.h\n"
    "  cxxopts.hpp\n"
    "  debug_utils.cu\n"
    "  debug_utils.h\n"
    "  distance_utils.cu\n"
    "  distance_utils.h\n"
    "  fit_tsne.cu\n"
    "  fit_tsne.h\n"
    "  main.cu\n"
    "  Makefile\n"
    "  math_utils.cu\n"
    "  math_utils.h\n"
    "  matrix_broadcast_utils.cu\n"
    "  matrix_broadcast_utils.h\n"
    "  nbodyfft.cu\n"
    "  nbodyfft.h\n"
    "  options.h\n"
    "  perplexity_search.cu\n"
    "  perplexity_search.h\n"
    "  rep_forces.cu\n"
    "  rep_forces.h\n"
    "  thrust_transform_functions.h"
)

EXPECTED_KERNELS = [
    {"file": "apply_forces.cu", "kernel": "IntegrationKernel", "line": 41},
    {"file": "attr_forces.cu", "kernel": "ComputePijxQijKernelV3", "line": 41},
    {"file": "attr_forces.cu", "kernel": "reduce_sum_kernel", "line": 71},
    {
        "file": "distance_utils.cu",
        "kernel": "PostprocessNeighborIndicesKernel",
        "line": 94,
    },
    {"file": "math_utils.cu", "kernel": "syv2k", "line": 62},
    {
        "file": "matrix_broadcast_utils.cu",
        "kernel": "BroadcastRowVector",
        "line": 46,
    },
    {
        "file": "matrix_broadcast_utils.cu",
        "kernel": "BroadcastColumnVector",
        "line": 65,
    },
    {"file": "nbodyfft.cu", "kernel": "copy_to_fft_input", "line": 50},
    {"file": "nbodyfft.cu", "kernel": "copy_from_fft_output", "line": 72},
    {"file": "nbodyfft.cu", "kernel": "compute_point_box_idx", "line": 94},
    {"file": "nbodyfft.cu", "kernel": "interpolate_device", "line": 128},
    {
        "file": "nbodyfft.cu",
        "kernel": "compute_interpolated_indices",
        "line": 159,
    },
    {"file": "nbodyfft.cu", "kernel": "compute_potential_indices", "line": 193},
    {"file": "nbodyfft.cu", "kernel": "compute_kernel_tilde", "line": 233},
    {
        "file": "nbodyfft.cu",
        "kernel": "compute_upper_and_lower_bounds",
        "line": 259,
    },
    {"file": "nbodyfft.cu", "kernel": "DFT2D1gpu", "line": 285},
    {"file": "nbodyfft.cu", "kernel": "DFT2D2gpu", "line": 307},
    {"file": "nbodyfft.cu", "kernel": "iDFT2D1gpu", "line": 331},
    {"file": "nbodyfft.cu", "kernel": "iDFT2D2gpu", "line": 359},
    {"file": "perplexity_search.cu", "kernel": "ComputePijKernel", "line": 40},
    {"file": "perplexity_search.cu", "kernel": "RowSumKernel", "line": 66},
    {"file": "perplexity_search.cu", "kernel": "NegEntropyKernel", "line": 86},
    {
        "file": "perplexity_search.cu",
        "kernel": "PerplexitySearchKernel",
        "line": 107,
    },
    {
        "file": "rep_forces.cu",
        "kernel": "compute_repulsive_forces_kernel",
        "line": 33,
    },
    {
        "file": "rep_forces.cu",
        "kernel": "compute_chargesQij_kernel",
        "line": 90,
    },
]
