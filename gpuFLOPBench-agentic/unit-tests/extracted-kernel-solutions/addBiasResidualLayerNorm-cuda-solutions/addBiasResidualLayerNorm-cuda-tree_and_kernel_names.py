EXPECTED_TREE = '/\n  CMakeLists.txt\n  kernels.h\n  main.cu\n  Makefile'

EXPECTED_MAIN_FILES = ['main.cu']

EXPECTED_INCLUDE_TREES = {'main.cu': 'main.cu\n'
            '  #include <algorithm> (DNE)\n'
            '  #include <chrono> (DNE)\n'
            '  #include <random> (DNE)\n'
            '  #include <cuda.h> (DNE)\n'
            '  #include <cuda_fp16.h> (DNE)\n'
            '  #include <cuda_bf16.h> (DNE)\n'
            '  #include "kernels.h"'}

EXPECTED_KERNELS = [{'file': 'kernels.h',
  'kernel': 'addBiasResidualPostLayerNormV2',
  'line': 202,
  'lines': 72,
  'offset': 201},
 {'file': 'kernels.h',
  'kernel': 'addBiasResidualPostLayerNorm',
  'line': 275,
  'lines': 51,
  'offset': 274},
 {'file': 'kernels.h',
  'kernel': 'generalAddBiasResidualPostLayerNorm',
  'line': 327,
  'lines': 62,
  'offset': 326}]
