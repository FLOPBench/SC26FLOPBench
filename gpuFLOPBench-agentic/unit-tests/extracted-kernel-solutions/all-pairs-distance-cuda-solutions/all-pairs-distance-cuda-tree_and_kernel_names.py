EXPECTED_TREE = '/\n  CMakeLists.txt\n  main.cu\n  Makefile'

EXPECTED_MAIN_FILES = ['main.cu']

EXPECTED_INCLUDE_TREES = {'main.cu': 'main.cu\n'
            '  #include <cstdio> (DNE)\n'
            '  #include <cstdlib> (DNE)\n'
            '  #include <cstring> (DNE)\n'
            '  #include <cuda.h> (DNE)\n'
            '  #include <cub/cub.cuh> (DNE)\n'
            '  #include <sys/time.h> (DNE)'}

EXPECTED_KERNELS = [{'file': 'main.cu', 'kernel': 'k1', 'line': 37, 'lines': 26, 'offset': 37},
 {'file': 'main.cu', 'kernel': 'k2', 'line': 66, 'lines': 64, 'offset': 66},
 {'file': 'main.cu', 'kernel': 'k3', 'line': 133, 'lines': 33, 'offset': 133}]
