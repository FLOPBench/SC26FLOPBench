EXPECTED_TREE = (
    "multimaterial-cuda/\n"
    "  compact.cu\n"
    "  full_matrix.cu\n"
    "  Makefile\n"
    "  multimat.cu\n"
    "  volfrac.dat.tgz"
)

EXPECTED_MAIN_FILES = ["multimat.cu"]

EXPECTED_KERNELS = [
    {"file": "compact.cu", "kernel": "ccc_loop1", "line": 18},
    {"file": "compact.cu", "kernel": "ccc_loop1_2", "line": 60},
    {"file": "compact.cu", "kernel": "ccc_loop2", "line": 80},
    {"file": "compact.cu", "kernel": "ccc_loop2_2", "line": 128},
    {"file": "compact.cu", "kernel": "ccc_loop3", "line": 144},
]
