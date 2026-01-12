EXPECTED_TREE = (
    "/\n"
    "  src/\n"
    "    helper/\n"
    "      args_parser.h\n"
    "      clipp.h\n"
    "      confusion.h\n"
    "      cuda_helpers.cuh\n"
    "      rngpu.hpp\n"
    "    bit_vector_functions.h\n"
    "    bit_vector_kernels.cuh\n"
    "    config.h\n"
    "    cuBool_cpu.h\n"
    "    cuBool_gpu.cuh\n"
    "    float_kernels.cuh\n"
    "    io_and_allocation.hpp\n"
    "    main.cu\n"
    "    updates_and_measures.cuh\n"
    "  LICENSE\n"
    "  Makefile\n"
    "  README.md"
)

EXPECTED_MAIN_FILES = ["src/main.cu"]

EXPECTED_INCLUDE_TREES = {
    "src/main.cu": """src/main.cu
  #include <vector> (DNE)
  #include <iostream> (DNE)
  #include <string> (DNE)
  #include "helper/clipp.h"
    #include <cstring> (DNE)
    #include <string> (DNE)
    #include <cstdlib> (DNE)
    #include <cstring> (DNE)
    #include <cctype> (DNE)
    #include <memory> (DNE)
    #include <vector> (DNE)
    #include <limits> (DNE)
    #include <stack> (DNE)
    #include <algorithm> (DNE)
    #include <sstream> (DNE)
    #include <utility> (DNE)
    #include <iterator> (DNE)
    #include <functional> (DNE)
  #include "io_and_allocation.hpp"
    #include <iostream> (DNE)
    #include <string> (DNE)
    #include <sstream> (DNE)
    #include <fstream> (DNE)
    #include <vector> (DNE)
    #include <bitset> (DNE)
    #include <ctime> (DNE)
    #include <algorithm> (DNE)
    #include <numeric> (DNE)
    #include <random> (DNE)
    #include <cmath> (DNE)
    #include <omp.h> (DNE)
    #include "config.h"
      #include <cstdlib> (DNE)
      #include <cstdint> (DNE)
      #include <limits> (DNE)
    #include "helper/rngpu.hpp"
      #include <stdint.h> (DNE)
      #include <math.h> (DNE)
  #include "bit_vector_functions.h"
    #include <vector> (DNE)
    #include <bitset> (DNE)
    #include "helper/confusion.h"
      #include <cstdlib> (DNE)
    #include "config.h"
      #include <cstdlib> (DNE)
      #include <cstdint> (DNE)
      #include <limits> (DNE)
    #include "io_and_allocation.hpp"
      #include <iostream> (DNE)
      #include <string> (DNE)
      #include <sstream> (DNE)
      #include <fstream> (DNE)
      #include <vector> (DNE)
      #include <bitset> (DNE)
      #include <ctime> (DNE)
      #include <algorithm> (DNE)
      #include <numeric> (DNE)
      #include <random> (DNE)
      #include <cmath> (DNE)
      #include <omp.h> (DNE)
      #include "config.h"
        #include <cstdlib> (DNE)
        #include <cstdint> (DNE)
        #include <limits> (DNE)
      #include "helper/rngpu.hpp"
        #include <stdint.h> (DNE)
        #include <math.h> (DNE)
    #include "updates_and_measures.cuh"
      #include "helper/cuda_helpers.cuh"
        #include <iostream> (DNE)
        #include <cstdint> (DNE)
        #include "../config.h"
          #include <cstdlib> (DNE)
          #include <cstdint> (DNE)
          #include <limits> (DNE)
        #include <chrono> (DNE)
      #include "helper/rngpu.hpp"
        #include <stdint.h> (DNE)
        #include <math.h> (DNE)
  #include "cuBool_gpu.cuh"
    #include <vector> (DNE)
    #include <iostream> (DNE)
    #include <sstream> (DNE)
    #include <limits> (DNE)
    #include <type_traits> (DNE)
    #include <omp.h> (DNE)
    #include "helper/rngpu.hpp"
      #include <stdint.h> (DNE)
      #include <math.h> (DNE)
    #include "helper/cuda_helpers.cuh"
      #include <iostream> (DNE)
      #include <cstdint> (DNE)
      #include "../config.h"
        #include <cstdlib> (DNE)
        #include <cstdint> (DNE)
        #include <limits> (DNE)
      #include <chrono> (DNE)
    #include "config.h"
      #include <cstdlib> (DNE)
      #include <cstdint> (DNE)
      #include <limits> (DNE)
    #include "io_and_allocation.hpp"
      #include <iostream> (DNE)
      #include <string> (DNE)
      #include <sstream> (DNE)
      #include <fstream> (DNE)
      #include <vector> (DNE)
      #include <bitset> (DNE)
      #include <ctime> (DNE)
      #include <algorithm> (DNE)
      #include <numeric> (DNE)
      #include <random> (DNE)
      #include <cmath> (DNE)
      #include <omp.h> (DNE)
      #include "config.h"
        #include <cstdlib> (DNE)
        #include <cstdint> (DNE)
        #include <limits> (DNE)
      #include "helper/rngpu.hpp"
        #include <stdint.h> (DNE)
        #include <math.h> (DNE)
    #include "bit_vector_kernels.cuh"
  #include "cuBool_cpu.h"
    #include <vector> (DNE)
    #include <iostream> (DNE)
    #include <limits> (DNE)
    #include <cmath> (DNE)
    #include <omp.h> (DNE)
    #include "helper/rngpu.hpp"
      #include <stdint.h> (DNE)
      #include <math.h> (DNE)
    #include "helper/confusion.h"
      #include <cstdlib> (DNE)
    #include "config.h"
      #include <cstdlib> (DNE)
      #include <cstdint> (DNE)
      #include <limits> (DNE)
    #include "io_and_allocation.hpp"
      #include <iostream> (DNE)
      #include <string> (DNE)
      #include <sstream> (DNE)
      #include <fstream> (DNE)
      #include <vector> (DNE)
      #include <bitset> (DNE)
      #include <ctime> (DNE)
      #include <algorithm> (DNE)
      #include <numeric> (DNE)
      #include <random> (DNE)
      #include <cmath> (DNE)
      #include <omp.h> (DNE)
      #include "config.h"
        #include <cstdlib> (DNE)
        #include <cstdint> (DNE)
        #include <limits> (DNE)
      #include "helper/rngpu.hpp"
        #include <stdint.h> (DNE)
        #include <math.h> (DNE)
    #include "bit_vector_functions.h"
      #include <vector> (DNE)
      #include <bitset> (DNE)
      #include "helper/confusion.h"
        #include <cstdlib> (DNE)
      #include "config.h"
        #include <cstdlib> (DNE)
        #include <cstdint> (DNE)
        #include <limits> (DNE)
      #include "io_and_allocation.hpp"
        #include <iostream> (DNE)
        #include <string> (DNE)
        #include <sstream> (DNE)
        #include <fstream> (DNE)
        #include <vector> (DNE)
        #include <bitset> (DNE)
        #include <ctime> (DNE)
        #include <algorithm> (DNE)
        #include <numeric> (DNE)
        #include <random> (DNE)
        #include <cmath> (DNE)
        #include <omp.h> (DNE)
        #include "config.h"
          #include <cstdlib> (DNE)
          #include <cstdint> (DNE)
          #include <limits> (DNE)
        #include "helper/rngpu.hpp"
          #include <stdint.h> (DNE)
          #include <math.h> (DNE)
      #include "updates_and_measures.cuh"
        #include "helper/cuda_helpers.cuh"
          #include <iostream> (DNE)
          #include <cstdint> (DNE)
          #include "../config.h"
            #include <cstdlib> (DNE)
            #include <cstdint> (DNE)
            #include <limits> (DNE)
          #include <chrono> (DNE)
        #include "helper/rngpu.hpp"
          #include <stdint.h> (DNE)
          #include <math.h> (DNE)

""",
}

EXPECTED_KERNELS = [
    {"file": "src/bit_vector_kernels.cuh", "kernel": "initFactor", "line": 31},
    {"file": "src/bit_vector_kernels.cuh", "kernel": "computeDistanceRows", "line": 59},
    {"file": "src/bit_vector_kernels.cuh", "kernel": "computeDistanceRowsShared", "line": 97},
    {"file": "src/bit_vector_kernels.cuh", "kernel": "vectorMatrixMultCompareRowWarpShared", "line": 158},
    {"file": "src/bit_vector_kernels.cuh", "kernel": "vectorMatrixMultCompareColWarpShared", "line": 236},
    {"file": "src/float_kernels.cuh", "kernel": "initFactor", "line": 5},
    {"file": "src/float_kernels.cuh", "kernel": "computeDistanceRowsShared", "line": 28},
]

