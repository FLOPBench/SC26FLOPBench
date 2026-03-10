EXPECTED_TREE = '/\n  CMakeLists.txt\n  compact.cu\n  full_matrix.cu\n  Makefile\n  multimat.cu\n  volfrac.dat.tgz'

EXPECTED_MAIN_FILES = ['multimat.cu']

EXPECTED_INCLUDE_TREES = {'multimat.cu': 'multimat.cu\n'
                '  #include <math.h> (DNE)\n'
                '  #include <stdio.h> (DNE)\n'
                '  #include <stdlib.h> (DNE)\n'
                '  #include <string.h> (DNE)\n'
                '  #include <algorithm> (DNE)\n'
                '  #include <chrono> (DNE)\n'
                '  #include <hbwmalloc.h> (DNE)'}

EXPECTED_KERNELS = [{'file': 'compact.cu',
  'kernel': 'ccc_loop1',
  'line': 18,
  'lines': 41,
  'offset': 18},
 {'file': 'compact.cu',
  'kernel': 'ccc_loop1_2',
  'line': 60,
  'lines': 19,
  'offset': 60},
 {'file': 'compact.cu',
  'kernel': 'ccc_loop2',
  'line': 80,
  'lines': 47,
  'offset': 80},
 {'file': 'compact.cu',
  'kernel': 'ccc_loop2_2',
  'line': 128,
  'lines': 15,
  'offset': 128},
 {'file': 'compact.cu',
  'kernel': 'ccc_loop3',
  'line': 144,
  'lines': 154,
  'offset': 144}]
