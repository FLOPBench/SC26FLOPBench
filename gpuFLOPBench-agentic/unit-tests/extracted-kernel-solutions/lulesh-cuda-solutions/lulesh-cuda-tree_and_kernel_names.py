EXPECTED_TREE = (
    "/\n"
    "  lulesh-init.cu\n"
    "  lulesh-util.cu\n"
    "  lulesh-viz.cu\n"
    "  lulesh.cu\n"
    "  lulesh.h\n"
    "  Makefile"
)

EXPECTED_MAIN_FILES = ["lulesh.cu"]

EXPECTED_INCLUDE_TREES = {
    "lulesh.cu": """lulesh.cu
  #include <math.h> (DNE)
  #include <stdio.h> (DNE)
  #include <stdlib.h> (DNE)
  #include <string.h> (DNE)
  #include <ctype.h> (DNE)
  #include <time.h> (DNE)
  #include <sys/time.h> (DNE)
  #include <unistd.h> (DNE)
  #include <climits> (DNE)
  #include <iostream> (DNE)
  #include <sstream> (DNE)
  #include <limits> (DNE)
  #include <fstream> (DNE)
  #include <string> (DNE)
  #include <random> (DNE)
  #include <cassert> (DNE)
  #include "lulesh.h"
    #include <math.h> (DNE)
    #include <vector> (DNE)
    #include <cuda.h> (DNE)

""",
}

EXPECTED_KERNELS = [
    {"file": "lulesh.cu", "kernel": "fill_sig", "line": 686, "offset": 685, "lines": 13},
    {"file": "lulesh.cu", "kernel": "integrateStress", "line": 699, "offset": 698, "lines": 74},
    {"file": "lulesh.cu", "kernel": "acc_final_force", "line": 773, "offset": 772, "lines": 31},
    {"file": "lulesh.cu", "kernel": "hgc", "line": 804, "offset": 803, "lines": 87},
    {"file": "lulesh.cu", "kernel": "fb", "line": 891, "offset": 890, "lines": 207},
    {"file": "lulesh.cu", "kernel": "collect_final_force", "line": 1098, "offset": 1097, "lines": 31},
    {"file": "lulesh.cu", "kernel": "accelerationForNode", "line": 1129, "offset": 1128, "lines": 18},
    {"file": "lulesh.cu", "kernel": "applyAccelerationBoundaryConditionsForNodes", "line": 1147, "offset": 1146, "lines": 20},
    {"file": "lulesh.cu", "kernel": "calcVelocityForNodes", "line": 1167, "offset": 1166, "lines": 30},
    {"file": "lulesh.cu", "kernel": "calcPositionForNodes", "line": 1197, "offset": 1196, "lines": 17},
    {"file": "lulesh.cu", "kernel": "calcKinematicsForElems", "line": 1214, "offset": 1213, "lines": 113},
    {"file": "lulesh.cu", "kernel": "calcStrainRates", "line": 1327, "offset": 1326, "lines": 29},
    {"file": "lulesh.cu", "kernel": "calcMonotonicQGradientsForElems", "line": 1356, "offset": 1355, "lines": 158},
    {"file": "lulesh.cu", "kernel": "calcMonotonicQForElems", "line": 1514, "offset": 1513, "lines": 172},
    {"file": "lulesh.cu", "kernel": "applyMaterialPropertiesForElems", "line": 1686, "offset": 1685, "lines": 227}
]
