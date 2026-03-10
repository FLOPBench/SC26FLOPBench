EXPECTED_TREE = '/\n  CMakeLists.txt\n  Copyright\n  kernel.h\n  License\n  main.cu\n  Makefile\n  rep.h'

EXPECTED_MAIN_FILES = ['main.cu']

EXPECTED_INCLUDE_TREES = {'main.cu': 'main.cu\n'
            '  #include <stdio.h> (DNE)\n'
            '  #include <stdlib.h> (DNE)\n'
            '  #include <stdint.h> (DNE)\n'
            '  #include <sys/time.h> (DNE)\n'
            '  #include "kernel.h"\n'
            '    #include <inttypes.h> (DNE)\n'
            '    #include <type_traits> (DNE)\n'
            '    #include <typeinfo> (DNE)\n'
            '    #include <cuda_fp16.h> (DNE)\n'
            '    #include "rep.h"'}

EXPECTED_KERNELS = [{'file': 'kernel.h',
  'kernel': 'block_stride',
  'line': 39,
  'lines': 44,
  'offset': 38},
 {'file': 'kernel.h',
  'kernel': 'block_stride',
  'line': 85,
  'lines': 44,
  'offset': 84}]
