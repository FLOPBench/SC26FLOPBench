solution = [
    r"""template<typename MatrixType>
    __global__ void matvec_kernel(const MINIFE_LOCAL_ORDINAL rows_size,
        const typename MatrixType::LocalOrdinalType *Arowoffsets,
        const typename MatrixType::GlobalOrdinalType *Acols,
        const typename MatrixType::ScalarType *Acoefs,
        const typename MatrixType::ScalarType *xcoefs,
        typename MatrixType::ScalarType *ycoefs)
    {
      MINIFE_LOCAL_ORDINAL row = blockIdx.x * blockDim.x + threadIdx.x;
      if (row < rows_size) {
        MINIFE_GLOBAL_ORDINAL row_start = Arowoffsets[row];
        MINIFE_GLOBAL_ORDINAL row_end   = Arowoffsets[row+1];
        MINIFE_SCALAR sum = 0;

        // Use the unroll factor in the OpenMP program 
#pragma unroll 27
        for(MINIFE_GLOBAL_ORDINAL i = row_start; i < row_end; ++i) {
          sum += Acoefs[i] * xcoefs[Acols[i]];
        }
        ycoefs[row] = sum;
      }
    }""".strip(),
]
