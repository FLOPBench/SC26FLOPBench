EXPECTED_TREE = "all-pairs-distance-cuda/\n  main.cu\n  Makefile"

EXPECTED_KERNELS = [
    {"file": "main.cu", "kernel": "k1", "line": 37},
    {"file": "main.cu", "kernel": "k2", "line": 66},
    {"file": "main.cu", "kernel": "k3", "line": 131},
]
