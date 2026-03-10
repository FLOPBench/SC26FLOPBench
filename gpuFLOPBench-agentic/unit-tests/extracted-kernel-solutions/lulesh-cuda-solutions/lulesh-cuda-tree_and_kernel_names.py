EXPECTED_TREE = '/\n  CMakeLists.txt\n  lulesh-init.cu\n  lulesh-util.cu\n  lulesh-viz.cu\n  lulesh.cu\n  lulesh.h\n  Makefile'

EXPECTED_MAIN_FILES = ['lulesh.cu']

EXPECTED_INCLUDE_TREES = {'lulesh.cu': 'lulesh.cu\n'
              '  #include <math.h> (DNE)\n'
              '  #include <stdio.h> (DNE)\n'
              '  #include <stdlib.h> (DNE)\n'
              '  #include <string.h> (DNE)\n'
              '  #include <ctype.h> (DNE)\n'
              '  #include <time.h> (DNE)\n'
              '  #include <sys/time.h> (DNE)\n'
              '  #include <unistd.h> (DNE)\n'
              '  #include <climits> (DNE)\n'
              '  #include <iostream> (DNE)\n'
              '  #include <sstream> (DNE)\n'
              '  #include <limits> (DNE)\n'
              '  #include <fstream> (DNE)\n'
              '  #include <string> (DNE)\n'
              '  #include <random> (DNE)\n'
              '  #include <cassert> (DNE)\n'
              '  #include "lulesh.h"\n'
              '    #include <math.h> (DNE)\n'
              '    #include <vector> (DNE)\n'
              '    #include <cuda.h> (DNE)'}

EXPECTED_KERNELS = [{'file': 'lulesh.cu',
  'kernel': 'fill_sig',
  'line': 686,
  'lines': 12,
  'offset': 686},
 {'file': 'lulesh.cu',
  'kernel': 'integrateStress',
  'line': 699,
  'lines': 73,
  'offset': 699},
 {'file': 'lulesh.cu',
  'kernel': 'acc_final_force',
  'line': 773,
  'lines': 30,
  'offset': 773},
 {'file': 'lulesh.cu',
  'kernel': 'hgc',
  'line': 804,
  'lines': 86,
  'offset': 804},
 {'file': 'lulesh.cu',
  'kernel': 'fb',
  'line': 891,
  'lines': 206,
  'offset': 891},
 {'file': 'lulesh.cu',
  'kernel': 'collect_final_force',
  'line': 1098,
  'lines': 30,
  'offset': 1098},
 {'file': 'lulesh.cu',
  'kernel': 'accelerationForNode',
  'line': 1129,
  'lines': 17,
  'offset': 1129},
 {'file': 'lulesh.cu',
  'kernel': 'applyAccelerationBoundaryConditionsForNodes',
  'line': 1147,
  'lines': 19,
  'offset': 1147},
 {'file': 'lulesh.cu',
  'kernel': 'calcVelocityForNodes',
  'line': 1167,
  'lines': 29,
  'offset': 1167},
 {'file': 'lulesh.cu',
  'kernel': 'calcPositionForNodes',
  'line': 1197,
  'lines': 16,
  'offset': 1197},
 {'file': 'lulesh.cu',
  'kernel': 'calcKinematicsForElems',
  'line': 1214,
  'lines': 112,
  'offset': 1214},
 {'file': 'lulesh.cu',
  'kernel': 'calcStrainRates',
  'line': 1327,
  'lines': 28,
  'offset': 1327},
 {'file': 'lulesh.cu',
  'kernel': 'calcMonotonicQGradientsForElems',
  'line': 1356,
  'lines': 157,
  'offset': 1356},
 {'file': 'lulesh.cu',
  'kernel': 'calcMonotonicQForElems',
  'line': 1514,
  'lines': 171,
  'offset': 1514},
 {'file': 'lulesh.cu',
  'kernel': 'applyMaterialPropertiesForElems',
  'line': 1686,
  'lines': 226,
  'offset': 1686}]
