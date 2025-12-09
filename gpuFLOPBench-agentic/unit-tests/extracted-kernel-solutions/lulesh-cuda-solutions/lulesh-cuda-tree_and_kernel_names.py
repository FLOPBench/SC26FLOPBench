EXPECTED_TREE = (
    "lulesh-cuda/\n"
    "  lulesh-init.cu\n"
    "  lulesh-util.cu\n"
    "  lulesh-viz.cu\n"
    "  lulesh.cu\n"
    "  lulesh.h\n"
    "  Makefile"
)

EXPECTED_MAIN_FILES = ["lulesh.cu"]

EXPECTED_KERNELS = [
    {"file": "lulesh.cu", "kernel": "fill_sig", "line": 686},
    {"file": "lulesh.cu", "kernel": "integrateStress", "line": 699},
    {"file": "lulesh.cu", "kernel": "acc_final_force", "line": 773},
    {"file": "lulesh.cu", "kernel": "hgc", "line": 804},
    {"file": "lulesh.cu", "kernel": "fb", "line": 891},
    {"file": "lulesh.cu", "kernel": "collect_final_force", "line": 1098},
    {"file": "lulesh.cu", "kernel": "accelerationForNode", "line": 1129},
    {
        "file": "lulesh.cu",
        "kernel": "applyAccelerationBoundaryConditionsForNodes",
        "line": 1147,
    },
    {"file": "lulesh.cu", "kernel": "calcVelocityForNodes", "line": 1167},
    {"file": "lulesh.cu", "kernel": "calcPositionForNodes", "line": 1197},
    {"file": "lulesh.cu", "kernel": "calcKinematicsForElems", "line": 1214},
    {"file": "lulesh.cu", "kernel": "calcStrainRates", "line": 1327},
    {
        "file": "lulesh.cu",
        "kernel": "calcMonotonicQGradientsForElems",
        "line": 1356,
    },
    {"file": "lulesh.cu", "kernel": "calcMonotonicQForElems", "line": 1514},
    {"file": "lulesh.cu", "kernel": "applyMaterialPropertiesForElems", "line": 1686},
]
