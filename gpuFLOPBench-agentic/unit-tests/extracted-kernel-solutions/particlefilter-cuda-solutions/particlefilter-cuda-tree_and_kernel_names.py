EXPECTED_TREE = '/\n  CMakeLists.txt\n  kernel_find_index.h\n  kernel_likelihood.h\n  kernel_normalize_weights.h\n  kernel_sum.h\n  main.cu\n  Makefile'

EXPECTED_MAIN_FILES = ['main.cu']

EXPECTED_INCLUDE_TREES = {'main.cu': 'main.cu\n'
            '  #include <stdio.h> (DNE)\n'
            '  #include <stdlib.h> (DNE)\n'
            '  #include <string.h> (DNE)\n'
            '  #include <limits.h> (DNE)\n'
            '  #include <math.h> (DNE)\n'
            '  #include <unistd.h> (DNE)\n'
            '  #include <fcntl.h> (DNE)\n'
            '  #include <float.h> (DNE)\n'
            '  #include <time.h> (DNE)\n'
            '  #include <sys/time.h> (DNE)\n'
            '  #include <iostream> (DNE)\n'
            '  #include <cuda.h> (DNE)\n'
            '  #include "kernel_find_index.h"\n'
            '  #include "kernel_likelihood.h"\n'
            '  #include "kernel_normalize_weights.h"\n'
            '  #include "kernel_sum.h"'}

EXPECTED_KERNELS = [{'file': 'kernel_find_index.h',
  'kernel': 'kernel_find_index',
  'line': 1,
  'lines': 29,
  'offset': 1},
 {'file': 'kernel_likelihood.h',
  'kernel': 'kernel_likelihood',
  'line': 1,
  'lines': 92,
  'offset': 1},
 {'file': 'kernel_normalize_weights.h',
  'kernel': 'kernel_normalize_weights',
  'line': 1,
  'lines': 44,
  'offset': 1},
 {'file': 'kernel_sum.h',
  'kernel': 'kernel_sum',
  'line': 1,
  'lines': 11,
  'offset': 1}]
