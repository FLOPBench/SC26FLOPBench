EXPECTED_TREE = '/\n  cluster.cu\n  CMakeLists.txt\n  data.tar.gz\n  gaussian.h\n  gaussian_kernel.cu\n  LICENSE\n  main.cu\n  Makefile\n  readData.cu\n  README.txt'

EXPECTED_MAIN_FILES = ['main.cu']

EXPECTED_INCLUDE_TREES = {'main.cu': 'main.cu\n'
            '  #include <stdlib.h> (DNE)\n'
            '  #include <stdio.h> (DNE)\n'
            '  #include <string.h> (DNE)\n'
            '  #include <math.h> (DNE)\n'
            '  #include <time.h> (DNE)\n'
            '  #include <stdlib.h> (DNE)\n'
            '  #include <float.h> (DNE)\n'
            '  #include <chrono> (DNE)\n'
            '  #include <iostream> (DNE)\n'
            '  #include <fstream> (DNE)\n'
            '  #include <vector> (DNE)\n'
            '  #include <cuda.h> (DNE)\n'
            '  #include "gaussian.h"\n'
            '    #include <stdio.h> (DNE)\n'
            '  #include "gaussian_kernel.cu"\n'
            '    #include "gaussian.h"\n'
            '      #include <stdio.h> (DNE)\n'
            '  #include "cluster.cu"\n'
            '  #include "readData.cu"'}

EXPECTED_KERNELS = [{'file': 'gaussian_kernel.cu',
  'kernel': 'constants_kernel',
  'line': 210,
  'lines': 77,
  'offset': 210},
 {'file': 'gaussian_kernel.cu',
  'kernel': 'seed_clusters_kernel',
  'line': 294,
  'lines': 95,
  'offset': 294},
 {'file': 'gaussian_kernel.cu',
  'kernel': 'estep1',
  'line': 420,
  'lines': 62,
  'offset': 420},
 {'file': 'gaussian_kernel.cu',
  'kernel': 'estep2',
  'line': 483,
  'lines': 67,
  'offset': 483},
 {'file': 'gaussian_kernel.cu',
  'kernel': 'mstep_means',
  'line': 559,
  'lines': 32,
  'offset': 559},
 {'file': 'gaussian_kernel.cu',
  'kernel': 'mstep_N',
  'line': 596,
  'lines': 41,
  'offset': 596},
 {'file': 'gaussian_kernel.cu',
  'kernel': 'mstep_covariance1',
  'line': 664,
  'lines': 63,
  'offset': 664},
 {'file': 'gaussian_kernel.cu',
  'kernel': 'mstep_covariance2',
  'line': 728,
  'lines': 90,
  'offset': 728}]
