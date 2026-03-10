EXPECTED_TREE = '/\n  CMakeLists.txt\n  kernels.h\n  Makefile\n  reduction.cu'

EXPECTED_MAIN_FILES = ['reduction.cu']

EXPECTED_INCLUDE_TREES = {'reduction.cu': 'reduction.cu\n'
                 '  #include <cstdio> (DNE)\n'
                 '  #include <cstdlib> (DNE)\n'
                 '  #include <iostream> (DNE)\n'
                 '  #include <chrono> (DNE)\n'
                 '  #include <cuda.h> (DNE)\n'
                 '  #include "kernels.h"'}

EXPECTED_KERNELS = [{'file': 'kernels.h',
  'kernel': 'atomic_reduction',
  'line': 1,
  'lines': 8,
  'offset': 1},
 {'file': 'kernels.h',
  'kernel': 'atomic_reduction_v2',
  'line': 10,
  'lines': 8,
  'offset': 10},
 {'file': 'kernels.h',
  'kernel': 'atomic_reduction_v4',
  'line': 19,
  'lines': 8,
  'offset': 19},
 {'file': 'kernels.h',
  'kernel': 'atomic_reduction_v8',
  'line': 28,
  'lines': 8,
  'offset': 28},
 {'file': 'kernels.h',
  'kernel': 'atomic_reduction_v16',
  'line': 37,
  'lines': 9,
  'offset': 37}]
