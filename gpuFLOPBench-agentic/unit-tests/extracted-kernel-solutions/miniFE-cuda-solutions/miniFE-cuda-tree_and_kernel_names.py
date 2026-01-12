EXPECTED_TREE = (
    "/\n"
    "  basic/\n"
    "    gold_files/\n"
    "      1x1x2_A.mtx.1.0\n"
    "      1x1x2_A.mtx.2.0\n"
    "      1x1x2_A.mtx.2.1\n"
    "      1x1x2_b.vec.1.0\n"
    "      1x1x2_b.vec.2.0\n"
    "      1x1x2_b.vec.2.1\n"
    "      1x1x2_x.vec.1.0\n"
    "      1x1x2_x.vec.2.0\n"
    "      1x1x2_x.vec.2.1\n"
    "    optional/\n"
    "      cuda/\n"
    "        CudaCall.hpp\n"
    "        CudaMemoryModel.hpp\n"
    "        CudaNode.cpp\n"
    "        CudaNode.cuh\n"
    "        CudaNode.hpp\n"
    "        CudaNodeImpl.hpp\n"
    "        cutil_inline_runtime.h\n"
    "        Matrix.cu\n"
    "        Vector.cu\n"
    "      ThreadPool/\n"
    "        cmake/\n"
    "          Dependencies.cmake\n"
    "          ThreadPool_config.h.in\n"
    "        config/\n"
    "          acx_pthread.m4\n"
    "          config.guess\n"
    "          config.sub\n"
    "          depcomp\n"
    "          generate-makeoptions.pl\n"
    "          install-sh\n"
    "          missing\n"
    "          replace-install-prefix.pl\n"
    "          string-replace.pl\n"
    "          strip_dup_incl_paths.pl\n"
    "          strip_dup_libs.pl\n"
    "          tac_arg_check_mpi.m4\n"
    "          tac_arg_config_mpi.m4\n"
    "          tac_arg_enable_export-makefiles.m4\n"
    "          tac_arg_enable_feature.m4\n"
    "          tac_arg_enable_feature_sub_check.m4\n"
    "          tac_arg_with_ar.m4\n"
    "          tac_arg_with_flags.m4\n"
    "          tac_arg_with_incdirs.m4\n"
    "          tac_arg_with_libdirs.m4\n"
    "          tac_arg_with_libs.m4\n"
    "          tac_arg_with_perl.m4\n"
    "          token-replace.pl\n"
    "        src/\n"
    "          Makefile.am\n"
    "          Makefile.in\n"
    "          ThreadPool_config.h.in\n"
    "          TPI.c\n"
    "          TPI.h\n"
    "          TPI.hpp\n"
    "          TPI_Walltime.c\n"
    "        test/\n"
    "          hhpccg/\n"
    "            BoxPartitionIB.c\n"
    "            BoxPartitionIB.h\n"
    "            CGSolver.c\n"
    "            CGSolver.h\n"
    "            dcrs_matrix.c\n"
    "            dcrs_matrix.h\n"
    "            main.c\n"
    "            tpi_vector.c\n"
    "            tpi_vector.h\n"
    "          hpccg/\n"
    "            BoxPartition.c\n"
    "            BoxPartition.h\n"
    "            CGSolver.c\n"
    "            CGSolver.h\n"
    "            main.c\n"
    "            tpi_vector.c\n"
    "            tpi_vector.h\n"
    "          build_gnu\n"
    "          build_intel\n"
    "          build_pgi\n"
    "          Makefile.am\n"
    "          Makefile.in\n"
    "          test_c_dnax.c\n"
    "          test_mpi_sum.c\n"
    "          test_pthreads.c\n"
    "          test_tpi.cpp\n"
    "          test_tpi_unit.c\n"
    "        aclocal.m4\n"
    "        bootstrap\n"
    "        configure\n"
    "        configure.ac\n"
    "        Makefile.am\n"
    "        Makefile.export.threadpool.in\n"
    "        Makefile.in\n"
    "        ThreadPool_config.h\n"
    "      copy_from_trilinos\n"
    "      make_targets\n"
    "      README\n"
    "    analytic_soln.hpp\n"
    "    assemble_FE_data.hpp\n"
    "    Box.hpp\n"
    "    box_utils.hpp\n"
    "    BoxIterator.hpp\n"
    "    BoxPartition.cpp\n"
    "    BoxPartition.hpp\n"
    "    cg_solve.hpp\n"
    "    compute_matrix_stats.hpp\n"
    "    ComputeNodeType.hpp\n"
    "    CSRMatrix.hpp\n"
    "    DotOp.hpp\n"
    "    driver.hpp\n"
    "    ELLMatrix.hpp\n"
    "    exchange_externals.hpp\n"
    "    FEComputeElem.hpp\n"
    "    FusedMatvecDotOp.hpp\n"
    "    generate_matrix_structure.hpp\n"
    "    get_common_files\n"
    "    GetNodesCoords.hpp\n"
    "    Hex8_box_utils.hpp\n"
    "    imbalance.hpp\n"
    "    Lock.hpp\n"
    "    LockingMatrix.hpp\n"
    "    LockingVector.hpp\n"
    "    main.cpp\n"
    "    make_local_matrix.hpp\n"
    "    make_targets\n"
    "    makefile\n"
    "    makefile.cuda.gnu.serial\n"
    "    makefile.cuda.tbb.gnu.serial\n"
    "    makefile.debug\n"
    "    makefile.gnu.purify\n"
    "    makefile.gnu.quantify\n"
    "    makefile.gnu.serial\n"
    "    makefile.intel.serial\n"
    "    makefile.redstorm\n"
    "    makefile.tbb\n"
    "    makefile.tbb.gnu.serial\n"
    "    makefile.tpi\n"
    "    makefile.tpi.gnu.serial\n"
    "    MatrixCopyOp.hpp\n"
    "    MatrixInitOp.hpp\n"
    "    MatvecOp.hpp\n"
    "    MemInitOp.hpp\n"
    "    NoOpMemoryModel.hpp\n"
    "    perform_element_loop.hpp\n"
    "    perform_element_loop_TBB_pipe.hpp\n"
    "    perform_element_loop_TBB_pllfor1.hpp\n"
    "    perform_element_loop_TBB_pllfor2.hpp\n"
    "    run_one_test\n"
    "    run_tests\n"
    "    SerialComputeNode.hpp\n"
    "    sharedmem.cuh\n"
    "    simple_mesh_description.hpp\n"
    "    SparseMatrix_functions.hpp\n"
    "    SumInLinSys.hpp\n"
    "    TBBNode.cpp\n"
    "    TBBNode.hpp\n"
    "    time_kernels.hpp\n"
    "    TPINode.hpp\n"
    "    TypeTraits.hpp\n"
    "    utest.cpp\n"
    "    utest_case.hpp\n"
    "    utest_cases.hpp\n"
    "    Vector.hpp\n"
    "    Vector_functions.hpp\n"
    "    verify_solution.hpp\n"
    "    WaxpbyOp.hpp\n"
    "  fem/\n"
    "    analytic_soln.hpp\n"
    "    ElemData.hpp\n"
    "    gauss_pts.hpp\n"
    "    Hex8.hpp\n"
    "    Hex8_ElemData.hpp\n"
    "    Hex8_enums.hpp\n"
    "    matrix_algebra_3x3.hpp\n"
    "    verify_solution.hpp\n"
    "  src/\n"
    "    assemble_FE_data.hpp\n"
    "    cg_solve.hpp\n"
    "    CSRMatrix.hpp\n"
    "    driver.hpp\n"
    "    ELLMatrix.hpp\n"
    "    exchange_externals.hpp\n"
    "    generate_info_header\n"
    "    generate_matrix_structure.hpp\n"
    "    get_common_files\n"
    "    GetNodesCoords.hpp\n"
    "    Hex8_box_utils.hpp\n"
    "    main.cpp\n"
    "    make_local_matrix.hpp\n"
    "    make_targets\n"
    "    Makefile\n"
    "    MatrixCopyOp.hpp\n"
    "    MatrixInitOp.hpp\n"
    "    omp-SparseMatrix_functions.hpp\n"
    "    omp-Vector_functions.hpp\n"
    "    perform_element_loop.hpp\n"
    "    README\n"
    "    simple_mesh_description.hpp\n"
    "    SparseMatrix_functions.hpp\n"
    "    time_kernels.hpp\n"
    "    Vector.hpp\n"
    "    Vector_functions.hpp\n"
    "    YAML_Doc.cpp\n"
    "    YAML_Doc.hpp\n"
    "    YAML_Element.cpp\n"
    "    YAML_Element.hpp\n"
    "  utils/\n"
    "    Box.hpp\n"
    "    box_utils.hpp\n"
    "    BoxIterator.hpp\n"
    "    BoxPartition.cpp\n"
    "    BoxPartition.hpp\n"
    "    compute_matrix_stats.hpp\n"
    "    imbalance.hpp\n"
    "    miniFE_no_info.hpp\n"
    "    miniFE_version.h\n"
    "    mytimer.cpp\n"
    "    mytimer.hpp\n"
    "    outstream.hpp\n"
    "    param_utils.cpp\n"
    "    param_utils.hpp\n"
    "    Parameters.hpp\n"
    "    TypeTraits.hpp\n"
    "    utils.cpp\n"
    "    utils.hpp\n"
    "  LICENSE"
)

EXPECTED_MAIN_FILES = [
    "basic/BoxPartition.cpp",
    "basic/main.cpp",
    "basic/optional/ThreadPool/test/hhpccg/BoxPartitionIB.c",
    "basic/optional/ThreadPool/test/hhpccg/main.c",
    "basic/optional/ThreadPool/test/hpccg/BoxPartition.c",
    "basic/optional/ThreadPool/test/hpccg/main.c",
    "basic/optional/ThreadPool/test/test_c_dnax.c",
    "basic/optional/ThreadPool/test/test_mpi_sum.c",
    "basic/optional/ThreadPool/test/test_tpi.cpp",
    "basic/optional/ThreadPool/test/test_tpi_unit.c",
    "basic/utest.cpp",
    "src/main.cpp",
    "utils/BoxPartition.cpp",
]

EXPECTED_FUNCTION_DEFINITIONS = {
    "basic/Box.hpp": """int * operator[](int xyz) (defnt)
int * operator[](int xyz) const (defnt)""",
"basic/BoxIterator.hpp": """~BoxIterator() (defnt)
static BoxIterator begin(const Box& box) (defnt)
static BoxIterator end(const Box& box) (defnt)
BoxIterator & operator=(const BoxIterator& src) (defnt)
BoxIterator & operator++() (defnt)
BoxIterator operator++(int) (defnt)
bool operator==(const BoxIterator& rhs) const (defnt)
bool operator!=(const BoxIterator& rhs) const (defnt)
BoxIterator(const Box& box, bool at_end = false) (defnt)""",
"basic/BoxPartition.cpp": """static int box_map_local_entry( const Box& box , const int ghost , int local_x , int local_y , int local_z ) (defnt)
int box_map_local( const Box& box_local, const int ghost , const int box_local_map[] , const int local_x , const int local_y , const int local_z ) (defnt)
void box_partition( int ip , int up , int axis , const Box& box, Box* p_box ) (defnt)
static int box_disjoint( const Box& a , const Box& b) (defnt)
static void resize_int( int ** a , int * allocLen , int newLen ) (defnt)
static void box_partition_maps( const int np , const int my_p , const Box* pbox, const int ghost , int ** map_local_id , int ** map_recv_pc , int ** map_send_pc , int ** map_send_id ) (defnt)
void box_partition_rcb( const int np , const int my_p , const Box& root_box, const int ghost , Box** pbox, int ** map_local_id , int ** map_recv_pc , int ** map_send_pc , int ** map_send_id ) (defnt)
static int box_contain( const Box& a , const Box& b ) (defnt)
static void box_print( FILE * fp , const Box& a ) (defnt)
static void test_box( const Box& box , const int np ) (defnt)
static void test_maps( const Box& root_box , const int np ) (defnt)
int main( int argc , char * argv[] ) (defnt)""",
    "basic/CSRMatrix.hpp": """template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename ComputeNode> CSRMatrix(ComputeNode& comp_node) (defnt)
template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename ComputeNode> ~CSRMatrix() (defnt)
template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename ComputeNode> size_t num_nonzeros() const (defnt)
template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename ComputeNode> void reserve_space(unsigned nrows, unsigned ncols_per_row) (defnt)
template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename ComputeNode> void get_row_pointers(GlobalOrdinalType row, size_t& row_length, GlobalOrdinalType*& cols, ScalarType*& coefs) (defnt)""",
    "basic/DotOp.hpp": """template <class Scalar> inline DotOp() (defnt)
template <class Scalar> identity() (defnt)
template <class Scalar> reduce(ReductionType u, ReductionType v) const (defnt)
template <class Scalar> inline KERNEL_PREFIX generate(int i) const (defnt)""",
    "basic/ELLMatrix.hpp": """template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename ComputeNode> ELLMatrix(ComputeNode& comp_node) (defnt)
template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename ComputeNode> ~ELLMatrix() (defnt)
template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename ComputeNode> size_t num_nonzeros() const (defnt)
template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename ComputeNode> void reserve_space(unsigned nrows, unsigned ncols_per_row) (defnt)
template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename ComputeNode> void get_row_pointers(GlobalOrdinalType row, size_t& row_length, GlobalOrdinalType*& cols_ptr, ScalarType*& coefs_ptr) (defnt)""",
    "basic/FEComputeElem.hpp": """template <typename GlobalOrdinal,typename Scalar> inline KERNEL_PREFIX operator()(int i) (defnt)""",
    "basic/FusedMatvecDotOp.hpp": """template <typename MatrixType, typename VectorType> inline FusedMatvecDotOp() (defnt)
template <typename MatrixType, typename VectorType> identity() (defnt)
template <typename MatrixType, typename VectorType> reduce(ReductionType u, ReductionType v) const (defnt)
template <typename MatrixType, typename VectorType> inline KERNEL_PREFIX generate(int row) (defnt)""",
    "basic/GetNodesCoords.hpp": """template <typename GlobalOrdinal,typename Scalar> inline void operator()(int i) (defnt)""",
    "basic/Hex8_box_utils.hpp": """template <typename GlobalOrdinal> void get_hex8_node_ids(int nx, int ny, GlobalOrdinal node0, GlobalOrdinal* elem_node_ids) (defnt)
template <typename Scalar> void get_hex8_node_coords_3d(Scalar x, Scalar y, Scalar z, Scalar hx, Scalar hy, Scalar hz, Scalar* elem_node_coords) (defnt)
template <typename GlobalOrdinal, typename Scalar> void get_elem_nodes_and_coords(const simple_mesh_description<GlobalOrdinal>& mesh, GlobalOrdinal elemID, GlobalOrdinal* node_ords, Scalar* node_coords) (defnt)
template <typename GlobalOrdinal, typename Scalar> void get_elem_nodes_and_coords(const simple_mesh_description<GlobalOrdinal>& mesh, GlobalOrdinal elemID, ElemData<GlobalOrdinal,Scalar>& elem_data) (defnt)""",
    "basic/Lock.hpp": """template <typename T> LockM(tbb::atomic<T>& row) (defnt)
template <typename T> ~LockM() (defnt)
template <typename T> LockV(tbb::atomic<T>& row) (defnt)
template <typename T> ~LockV() (defnt)""",
    "basic/LockingMatrix.hpp": """template <typename MatrixType> LockingMatrix(MatrixType& A) (defnt)
template <typename MatrixType> void sum_in(GlobalOrdinal row, size_t row_len, const GlobalOrdinal* col_indices, const Scalar* values) (defnt)""",
    "basic/LockingVector.hpp": """template <typename VectorType> LockingVector(VectorType& x) (defnt)
template <typename VectorType> void sum_in(size_t num_indices, const GlobalOrdinal* indices, const Scalar* values) (defnt)""",
    "basic/MatrixCopyOp.hpp": """template <typename MatrixType> inline void operator()(int i) (defnt)""",
    "basic/MatrixInitOp.hpp": """template <typename GlobalOrdinal> void sort_if_needed(GlobalOrdinal* list, GlobalOrdinal list_len) (defnt)
template <> MatrixInitOp(const std::vector<MINIFE_GLOBAL_ORDINAL>& rows_vec, const std::vector<MINIFE_LOCAL_ORDINAL>& row_offsets_vec, const std::vector<int>& row_coords_vec, int global_nx, int global_ny, int global_nz, MINIFE_GLOBAL_ORDINAL global_n_rows, const miniFE::simple_mesh_description<MINIFE_GLOBAL_ORDINAL>& input_mesh, miniFE::CSRMatrix<MINIFE_SCALAR,MINIFE_LOCAL_ORDINAL,MINIFE_GLOBAL_ORDINAL,ComputeNodeType>& matrix) (defnt)
template <> inline void operator()(int i) (defnt)
template <> MatrixInitOp(const std::vector<MINIFE_GLOBAL_ORDINAL>& rows_vec, const std::vector<MINIFE_LOCAL_ORDINAL>& /*row_offsets_vec*/, const std::vector<int>& row_coords_vec, int global_nx, int global_ny, int global_nz, MINIFE_GLOBAL_ORDINAL global_n_rows, const miniFE::simple_mesh_description<MINIFE_GLOBAL_ORDINAL>& input_mesh, miniFE::ELLMatrix<MINIFE_SCALAR,MINIFE_LOCAL_ORDINAL,MINIFE_GLOBAL_ORDINAL,ComputeNodeType>& matrix) (defnt)
template <> inline void operator()(int i) (defnt)""",
    "basic/MatvecOp.hpp": """template <> MatvecOp(miniFE::CSRMatrix<MINIFE_SCALAR,MINIFE_LOCAL_ORDINAL,MINIFE_GLOBAL_ORDINAL, ComputeNodeType>& A) (defnt)
template <> inline KERNEL_PREFIX operator()(int row) (defnt)
template <> MatvecOp(miniFE::ELLMatrix<MINIFE_SCALAR,MINIFE_LOCAL_ORDINAL,MINIFE_GLOBAL_ORDINAL, ComputeNodeType>& A) (defnt)
template <> inline KERNEL_PREFIX operator()(int row) (defnt)""",
    "basic/MemInitOp.hpp": """template <class Scalar> inline void operator()(size_t i) (defnt)""",
    "basic/NoOpMemoryModel.hpp": """NoOpMemoryModel() (defnt)
~NoOpMemoryModel() (defnt)
template <class T> T * get_buffer(const T* host_ptr, size_t buf_size) (defnt)
template <class T> void destroy_buffer(T*& device_ptr) (defnt)
template <class T> void copy_to_buffer(const T* host_ptr, size_t buf_size, T* device_ptr) (defnt)
template <class T> void copy_from_buffer(T* host_ptr, size_t buf_size, const T* device_ptr) (defnt)""",
    "basic/SerialComputeNode.hpp": """template <class WDP> void parallel_for(unsigned int length, WDP wd) (defnt)
template <class WDP> void parallel_reduce(unsigned int length, WDP &wd) (defnt)""",
    "basic/SparseMatrix_functions.hpp": """template <typename MatrixType> void init_matrix(MatrixType& M, const std::vector<typename MatrixType::GlobalOrdinalType>& rows, const std::vector<typename MatrixType::LocalOrdinalType>& row_offsets, const std::vector<int>& row_coords, int global_nodes_x, int global_nodes_y, int global_nodes_z, typename MatrixType::GlobalOrdinalType global_nrows, const simple_mesh_description<typename MatrixType::GlobalOrdinalType>& mesh) (defnt)
template <typename T, typename U> void sort_with_companions(ptrdiff_t len, T* array, U* companions) (defnt)
template <typename MatrixType> void write_matrix(const std::string& filename, MatrixType& mat) (defnt)
template <typename GlobalOrdinal,typename Scalar> void sum_into_row(int row_len, GlobalOrdinal* row_indices, Scalar* row_coefs, int num_inputs, const GlobalOrdinal* input_indices, const Scalar* input_coefs) (defnt)
template <typename MatrixType> void sum_into_row(typename MatrixType::GlobalOrdinalType row, size_t num_indices, const typename MatrixType::GlobalOrdinalType* col_inds, const typename MatrixType::ScalarType* coefs, MatrixType& mat) (defnt)
template <typename MatrixType> void sum_in_symm_elem_matrix(size_t num, const typename MatrixType::GlobalOrdinalType* indices, const typename MatrixType::ScalarType* coefs, MatrixType& mat) (defnt)
template <typename MatrixType> void sum_in_elem_matrix(size_t num, const typename MatrixType::GlobalOrdinalType* indices, const typename MatrixType::ScalarType* coefs, MatrixType& mat) (defnt)
template <typename GlobalOrdinal, typename Scalar, typename MatrixType, typename VectorType> void sum_into_global_linear_system(ElemData<GlobalOrdinal,Scalar>& elem_data, MatrixType& A, VectorType& b) (defnt)
template <typename MatrixType> void sum_in_elem_matrix(size_t num, const typename MatrixType::GlobalOrdinalType* indices, const typename MatrixType::ScalarType* coefs, LockingMatrix<MatrixType>& mat) (defnt)
template <typename GlobalOrdinal, typename Scalar, typename MatrixType, typename VectorType> void sum_into_global_linear_system(ElemData<GlobalOrdinal,Scalar>& elem_data, LockingMatrix<MatrixType>& A, LockingVector<VectorType>& b) (defnt)
template <typename MatrixType> void add_to_diagonal(typename MatrixType::ScalarType value, MatrixType& mat) (defnt)
template <typename MatrixType> double parallel_memory_overhead_MB(const MatrixType& A) (defnt)
template <typename MatrixType> void rearrange_matrix_local_external(MatrixType& A) (defnt)
template <typename MatrixType> void zero_row_and_put_1_on_diagonal(MatrixType& A, typename MatrixType::GlobalOrdinalType row) (defnt)
template <typename MatrixType, typename VectorType> void impose_dirichlet(typename MatrixType::ScalarType prescribed_value, MatrixType& A, VectorType& b, int global_nx, int global_ny, int global_nz, const std::set<typename MatrixType::GlobalOrdinalType>& bc_rows) (defnt)
template <typename MatrixType, typename VectorType> typename TypeTraits<typename VectorType::ScalarType>::magnitude_type matvec_and_dot(MatrixType& A, VectorType& x, VectorType& y) (defnt)
template <typename MatrixType, typename VectorType> void operator()(MatrixType& A, VectorType& x, VectorType& y) (defnt)
template <typename MatrixType, typename VectorType> void matvec(MatrixType& A, VectorType& x, VectorType& y) (defnt)
template <typename MatrixType, typename VectorType> void operator()(MatrixType& A, VectorType& x, VectorType& y) (defnt)""",
    "basic/SumInLinSys.hpp": """template <typename GlobalOrdinal,typename Scalar, typename MatrixType, typename VectorType> inline void operator()(int i) (defnt)""",
    "basic/TBBNode.hpp": """template <class WDPin> BlockedRangeWDP(WDPin &in) (defnt)
template <class WDPin> inline void operator()(tbb::blocked_range<int> &rng) const (defnt)
template <class WDPin> BlockedRangeWDPReducer(WDPin &in) (defnt)
template <class WDPin> BlockedRangeWDPReducer(BlockedRangeWDPReducer &in, tbb::split) (defnt)
template <class WDPin> void operator()(tbb::blocked_range<int> &rng) (defnt)
template <class WDPin> inline void join( const BlockedRangeWDPReducer<WDPin> &other ) (defnt)
TBBNode(int numThreads=0) (defnt)
~TBBNode() (defnt)
template <class WDP> void parallel_for(int length, WDP wd) (defnt)
template <class WDP> void parallel_reduce(int length, WDP &wd) (defnt)""",
    "basic/TPINode.hpp": """inline void tpi_work_span(TPI_Work* work, int n, size_t& ibeg, size_t& iend) (defnt)
template <class WDP> void tpi_execute(TPI_Work * work) (defnt)
template <class WDP> void tpi_reduction_work(TPI_Work * work) (defnt)
template <class WDP> void tpi_reduction_join(TPI_Work * work, const void* src) (defnt)
template <class WDP> void tpi_reduction_init(TPI_Work * work) (defnt)
TPINode(int numThreads=0) (defnt)
~TPINode() (defnt)
template <class WDP> void parallel_for(int length, WDP & wd ) (defnt)
template <class WDP> void parallel_reduce(int length, WDP & wd ) (defnt)""",
    "basic/TypeTraits.hpp": """template <> static char * name() (defnt)
template <> static MPI_Datatype mpi_type() (defnt)
template <> static char * name() (defnt)
template <> static MPI_Datatype mpi_type() (defnt)
template <> static char * name() (defnt)
template <> static MPI_Datatype mpi_type() (defnt)
template <> static char * name() (defnt)
template <> static MPI_Datatype mpi_type() (defnt)
template <> static char * name() (defnt)
template <> static MPI_Datatype mpi_type() (defnt)
template <> static char * name() (defnt)
template <> static MPI_Datatype mpi_type() (defnt)
template <> static char * name() (defnt)
template <> static MPI_Datatype mpi_type() (defnt)
template <> static char * name() (defnt)
template <> static MPI_Datatype mpi_type() (defnt)""",
    "basic/Vector.hpp": """template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename ComputeNode> Vector(GlobalOrdinal startIdx, LocalOrdinal local_sz, ComputeNode& cn) (defnt)
template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename ComputeNode> ~Vector() (defnt)""",
    "basic/Vector_functions.hpp": """template <typename VectorType> void write_vector(const std::string& filename, const VectorType& vec) (defnt)
template <typename VectorType> void sum_into_vector(size_t num_indices, const typename VectorType::GlobalOrdinalType* indices, const typename VectorType::ScalarType* coefs, VectorType& vec) (defnt)
template <typename VectorType> void sum_into_vector(size_t num_indices, const typename VectorType::GlobalOrdinalType* indices, const typename VectorType::ScalarType* coefs, LockingVector<VectorType>& vec) (defnt)
template <typename VectorType> void waxpby(typename VectorType::ScalarType alpha, const VectorType& x, typename VectorType::ScalarType beta, const VectorType& y, VectorType& w) (defnt)
template <typename VectorType> void fused_waxpby(typename VectorType::ScalarType alpha, const VectorType& x, typename VectorType::ScalarType beta, const VectorType& y, VectorType& w, typename VectorType::ScalarType alpha2, const VectorType& x2, typename VectorType::ScalarType beta2, const VectorType& y2, VectorType& w2) (defnt)
template <typename Vector> typename TypeTraits<typename Vector::ScalarType>::magnitude_type dot(const Vector& x, const Vector& y) (defnt)""",
    "basic/WaxpbyOp.hpp": """template <class Scalar> KERNEL_PREFIX operator()(size_t i) const (defnt)
template <class Scalar> KERNEL_PREFIX operator()(size_t i) const (defnt)""",
    "basic/analytic_soln.hpp": """inline Scalar fcn_l(int p, int q) (defnt)
inline Scalar fcn(int n, Scalar u) (defnt)
inline Scalar soln(Scalar x, Scalar y, Scalar z, int max_p, int max_q) (defnt)""",
    "basic/assemble_FE_data.hpp": """template <typename MatrixType, typename VectorType> void assemble_FE_data(const simple_mesh_description<typename MatrixType::GlobalOrdinalType>& mesh, MatrixType& A, VectorType& b, Parameters& params) (defnt)""",
    "basic/box_utils.hpp": """inline void copy_box(const Box& from_box, Box& to_box) (defnt)
template <typename GlobalOrdinal> void get_int_coords(GlobalOrdinal ID, int nx, int ny, int nz, int& x, int& y, int& z) (defnt)
template <typename GlobalOrdinal,typename Scalar> void get_coords(GlobalOrdinal ID, int nx, int ny, int nz, Scalar& x, Scalar& y, Scalar& z) (defnt)
template <typename GlobalOrdinal> GlobalOrdinal get_num_ids(const Box& box) (defnt)
template <typename GlobalOrdinal> GlobalOrdinal get_id(int nx, int ny, int nz, int x, int y, int z) (defnt)
template <typename GlobalOrdinal> void get_ids(int nx, int ny, int nz, const Box& box, GlobalOrdinal* ids) (defnt)
template <typename GlobalOrdinal> void create_map_id_to_row(int global_nx, int global_ny, int global_nz, const Box& box, std::map<GlobalOrdinal,GlobalOrdinal>& id_to_row) (defnt)""",
    "basic/cg_solve.hpp": """template <typename Scalar> void print_vec(const std::vector<Scalar>& vec, const std::string& name) (defnt)
template <typename VectorType> bool breakdown(typename VectorType::ScalarType inner, const VectorType& v, const VectorType& w) (defnt)
template <typename OperatorType, typename VectorType, typename Matvec> void cg_solve(OperatorType& A, const VectorType& b, VectorType& x, Matvec matvec, typename OperatorType::LocalOrdinalType max_iter, typename TypeTraits<typename OperatorType::ScalarType>::magnitude_type& tolerance, typename OperatorType::LocalOrdinalType& num_iters, typename TypeTraits<typename OperatorType::ScalarType>::magnitude_type& normr, timer_type* my_cg_times) (defnt)""",
    "basic/compute_matrix_stats.hpp": """template <typename MatrixType> size_t compute_matrix_stats(const MatrixType& A, int myproc, int numprocs, YAML_Doc& ydoc) (defnt)""",
    "basic/driver.hpp": """template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename ComputeNodeType> void driver(const Box& global_box, Box& my_box, ComputeNodeType& compute_node, Parameters& params, YAML_Doc& ydoc) (defnt)""",
    "basic/exchange_externals.hpp": """template <typename MatrixType, typename VectorType> void exchange_externals(MatrixType& A, VectorType& x) (defnt)
template <typename MatrixType, typename VectorType> void begin_exchange_externals(MatrixType& A, VectorType& x) (defnt)
inline void finish_exchange_externals(int num_neighbors) (defnt)""",
    "basic/generate_matrix_structure.hpp": """template <typename MatrixType> int generate_matrix_structure(const simple_mesh_description<typename MatrixType::GlobalOrdinalType>& mesh, MatrixType& A) (defnt)""",
    "basic/imbalance.hpp": """template <typename GlobalOrdinal> void compute_imbalance(const Box& global_box, const Box& local_box, float& largest_imbalance, float& std_dev, YAML_Doc& doc, bool record_in_doc) (defnt)
std::pair<int,int> decide_how_to_grow(const Box& global_box, const Box& local_box) (defnt)
std::pair<int,int> decide_how_to_shrink(const Box& global_box, const Box& local_box) (defnt)
template <typename GlobalOrdinal> void add_imbalance(const Box& global_box, Box& local_box, float imbalance, YAML_Doc& doc) (defnt)""",
    "basic/main.cpp": """inline void print_box(int myproc, const char* name, const Box& box, const char* name2, const Box& box2) (defnt)
int main(int argc, char** argv) (defnt)
void add_params_to_yaml(YAML_Doc& doc, miniFE::Parameters& params) (defnt)
void add_configuration_to_yaml(YAML_Doc& doc, int numprocs, int numthreads) (defnt)
void add_timestring_to_yaml(YAML_Doc& doc) (defnt)""",
    "basic/make_local_matrix.hpp": """template <typename MatrixType> void make_local_matrix(MatrixType& A) (defnt)""",
    "basic/optional/ThreadPool/src/TPI.c": """static int atomic_fetch_and_decrement( volatile int * value ) (defnt)
char * TPI_Version() (defnt)
int TPI_Lock( int i ) (defnt)
int TPI_Unlock( int i ) (defnt)
static int local_set_lock_count( const int lock_count ) (defnt)
static void local_destroy_locks() (defnt)
static void local_run( Thread * const this_thread , void * reduce ) (defnt)
static int wait_thread( volatile long * const control , const int val ) (defnt)
static void local_barrier_wait( Thread * const this_thread , Thread * const thread ) (defnt)
static void local_barrier( Thread * const this_thread ) (defnt)
static void * local_driver( void * arg ) (defnt)
static void alloc_reduce( int reduce_size ) (defnt)
static int local_start( int work_thread_count , TPI_work_subprogram work_subprogram , const void * work_info , int work_count , int lock_count , TPI_reduce_join reduce_join , TPI_reduce_init reduce_init , int reduce_size , void * reduce_data ) (defnt)
static void local_wait() (defnt)
int TPI_Init( int n ) (defnt)
int TPI_Finalize() (defnt)
static void local_block( TPI_Work * work ) (defnt)
int TPI_Block() (defnt)
int TPI_Unblock() (defnt)
int TPI_Isblocked() (defnt)
int TPI_Lock( int i ) (defnt)
int TPI_Unlock( int i ) (defnt)
static int local_set_lock_count( const int lock_count ) (defnt)
static int local_start( int work_thread_count , TPI_work_subprogram work_subprogram , const void * work_info , int work_count , int lock_count , TPI_reduce_join reduce_join , TPI_reduce_init reduce_init , int reduce_size , void * reduce_data ) (defnt)
static void local_wait() (defnt)
static void local_block( TPI_Work * work ) (defnt)
int TPI_Block() (defnt)
int TPI_Unblock() (defnt)
int TPI_Isblocked() (defnt)
int TPI_Init( int n ) (defnt)
int TPI_Finalize() (defnt)
int TPI_Wait() (defnt)
int TPI_Start( TPI_work_subprogram work_subprogram , const void * work_info , int work_count , int lock_count ) (defnt)
int TPI_Run( TPI_work_subprogram work_subprogram , const void * work_info , int work_count , int lock_count ) (defnt)
int TPI_Run_threads( TPI_work_subprogram work_subprogram , const void * work_info , int lock_count ) (defnt)
int TPI_Start_threads( TPI_work_subprogram work_subprogram , const void * work_info , int lock_count ) (defnt)
int TPI_Run_reduce( TPI_work_subprogram work_subprogram , const void * work_info , int work_count , TPI_reduce_join reduce_join , TPI_reduce_init reduce_init , int reduce_size , void * reduce_data ) (defnt)
int TPI_Run_threads_reduce( TPI_work_subprogram work_subprogram , const void * work_info , TPI_reduce_join reduce_join , TPI_reduce_init reduce_init , int reduce_size , void * reduce_data ) (defnt)
int TPI_Start_reduce( TPI_work_subprogram work_subprogram , const void * work_info , int work_count , TPI_reduce_join reduce_join , TPI_reduce_init reduce_init , int reduce_size , void * reduce_data ) (defnt)
int TPI_Start_threads_reduce( TPI_work_subprogram work_subprogram , const void * work_info , TPI_reduce_join reduce_join , TPI_reduce_init reduce_init , int reduce_size , void * reduce_data ) (defnt)""",
    "basic/optional/ThreadPool/src/TPI.hpp": """inline int Lock( int n ) (defnt)
inline int Unlock( int n ) (defnt)
LockGuard( unsigned i_lock ) (defnt)
~LockGuard() (defnt)
inline int Init( int n ) (defnt)
inline int Finalize() (defnt)
inline double Walltime() (defnt)
template <class Worker> WorkerMethodHelper( Worker & w , Method m ) (defnt)
template <class Worker> static void run( TPI_Work * work ) (defnt)
template <class Worker> inline int Run( Worker & worker, void (Worker::*method)(Work &) , int work_count , int lock_count ) (defnt)""",
    "basic/optional/ThreadPool/src/TPI_Walltime.c": """double TPI_Walltime() (defnt)""",
    "basic/optional/ThreadPool/test/hhpccg/BoxPartitionIB.c": """static void box_partition( int ip , int up , int axis , BoxInput box , int (* const p_box)[3][2] ) (defnt)
void box_partition_rcb( const int np , const int root_box[3][2] , int pbox[][3][2] ) (defnt)
static int box_intersect( BoxInput a , BoxInput b , BoxOutput c ) (defnt)
static void global_to_use_box( BoxInput gbox , BoxInput pbox , const int ghost , BoxOutput interiorBox , BoxOutput useBox ) (defnt)
static int map_global_to_use_box( BoxInput useBox , const int global_x , const int global_y , const int global_z ) (defnt)
int box_map_local( const int local_uses[3][2] , const int map_local_id[] , const int global_x , const int global_y , const int global_z ) (defnt)
static void resize_int( int ** a , int * allocLen , int newLen ) (defnt)
void box_partition_map( const int np , const int my_p , const int gbox[3][2] , const int pbox[][3][2] , const int ghost , int map_use_box[3][2] , int map_local_id[] , int * map_count_interior , int * map_count_owns , int * map_count_uses , int ** map_recv_pc , int ** map_send_pc , int ** map_send_id ) (defnt)
static int box_contain( const int a[3][2] , const int b[3][2] ) (defnt)
static void box_print( FILE * fp , const int a[][2] ) (defnt)
static int box_disjoint( BoxInput a , BoxInput b ) (defnt)
static void test_box( const int box[3][2] , const int np ) (defnt)
static void test_maps( const int root_box[][2] , const int np ) (defnt)
int main( int argc , char * argv[] ) (defnt)""",
    "basic/optional/ThreadPool/test/hhpccg/CGSolver.c": """void cgsolve_set_lhs( const struct distributed_crs_matrix * const matrix , const VECTOR_SCALAR * const x , VECTOR_SCALAR * const b ) (defnt)
void cgsolve_blas( const struct distributed_crs_matrix * matrix , const VECTOR_SCALAR * const b , VECTOR_SCALAR * const x , const VECTOR_SCALAR tolerance , const int max_iter , const int print_iter , int * const iter_count , VECTOR_SCALAR * const norm_resid , double * const solve_dt ) (defnt)
void cgsolve( const struct distributed_crs_matrix * matrix , const VECTOR_SCALAR * const b , VECTOR_SCALAR * const x , const int overlap_comm , const VECTOR_SCALAR tolerance , const int max_iter , const int print_iter , int * const iter_count , VECTOR_SCALAR * const norm_resid , double * const solve_dt ) (defnt)
static void tpi_work_dot_join( TPI_Work * work , const void * src ) (defnt)
static void tpi_work_dot_init( TPI_Work * work ) (defnt)
static void tpi_work_update( TPI_Work * work ) (defnt)
double cgsolver_update( const int length , const VECTOR_SCALAR alpha , const VECTOR_SCALAR * p , const VECTOR_SCALAR * Ap , VECTOR_SCALAR * x , VECTOR_SCALAR * r ) (defnt)""",
    "basic/optional/ThreadPool/test/hhpccg/dcrs_matrix.c": """static double comm_sum( double v ) (defnt)
static double comm_sum( double v ) (defnt)
static void get_off_process_entries( const struct distributed_crs_matrix * const matrix , VECTOR_SCALAR * const vec ) (defnt)
static void dcrs_apply_and_dot_span( const struct distributed_crs_matrix * const matrix , const int span_begin , const int span_end , const VECTOR_SCALAR * const x , VECTOR_SCALAR * const y , double * const result ) (defnt)
static void dcrs_apply_span( const struct distributed_crs_matrix * const matrix , const int span_begin , const int span_end , const VECTOR_SCALAR * const x , VECTOR_SCALAR * const y ) (defnt)
static void work_span( const int count , const int rank , int * jBeg , int * jEnd ) (defnt)
static void tpi_work_dot_join( TPI_Work * work , const void * src ) (defnt)
static void tpi_work_dot_init( TPI_Work * work ) (defnt)
static void tpi_work_dcrs_apply_and_dot( TPI_Work * work ) (defnt)
double dcrs_apply_and_dot( const struct distributed_crs_matrix * matrix , VECTOR_SCALAR * x , VECTOR_SCALAR * y , const int overlap_communication ) (defnt)
static void tpi_work_dcrs_apply( TPI_Work * work ) (defnt)
void dcrs_apply( const struct distributed_crs_matrix * matrix , VECTOR_SCALAR * x , VECTOR_SCALAR * y ) (defnt)""",
    "basic/optional/ThreadPool/test/hhpccg/main.c": """int main( int argc , char ** argv ) (defnt)
static void hpccg_alloc_and_fill( const int np , const int my_p , const int gbox[][2] , const int ghost , struct distributed_crs_matrix * const matrix ) (defnt)""",
    "basic/optional/ThreadPool/test/hhpccg/tpi_vector.c": """void tpi_work_span( TPI_Work * const work , const int n , int * const iBeg , int * const iEnd ) (defnt)
static void tpi_work_fill( TPI_Work * work ) (defnt)
void tpi_fill( int n , VECTOR_SCALAR alpha , VECTOR_SCALAR * x ) (defnt)
static void tpi_work_scale( TPI_Work * work ) (defnt)
void tpi_scale( int n , const VECTOR_SCALAR alpha , VECTOR_SCALAR * x ) (defnt)
static void tpi_work_copy( TPI_Work * work ) (defnt)
void tpi_copy( int n , const VECTOR_SCALAR * x , VECTOR_SCALAR * y ) (defnt)
static void tpi_work_axpby( TPI_Work * work ) (defnt)
void tpi_axpby( int n , VECTOR_SCALAR alpha , const VECTOR_SCALAR * x , VECTOR_SCALAR beta , VECTOR_SCALAR * y ) (defnt)
static void tpi_work_axpy( TPI_Work * work ) (defnt)
void tpi_axpy( int n , VECTOR_SCALAR alpha , const VECTOR_SCALAR * x , VECTOR_SCALAR * y ) (defnt)
static void tpi_work_xpby( TPI_Work * work ) (defnt)
void tpi_xpby( int n , const VECTOR_SCALAR * x , VECTOR_SCALAR beta , VECTOR_SCALAR * y ) (defnt)
static void tpi_work_dot_partial( TPI_Work * work ) (defnt)
static void tpi_work_dot_partial_self( TPI_Work * work ) (defnt)
static void tpi_work_dot_join( TPI_Work * work , const void * src ) (defnt)
static void tpi_work_dot_init( TPI_Work * work ) (defnt)
double tpi_dot( int n , const VECTOR_SCALAR * x , const VECTOR_SCALAR * y ) (defnt)""",
    "basic/optional/ThreadPool/test/hpccg/BoxPartition.c": """static int box_map_local_entry( const int box[][2] , const int ghost , int local_x , int local_y , int local_z ) (defnt)
int box_map_local( const int box_local[][2] , const int ghost , const int box_local_map[] , const int local_x , const int local_y , const int local_z ) (defnt)
static void box_partition( int ip , int up , int axis , const int box[3][2] , int p_box[][3][2] ) (defnt)
static int box_disjoint( const int a[3][2] , const int b[3][2] ) (defnt)
static void resize_int( int ** a , int * allocLen , int newLen ) (defnt)
static void box_partition_maps( const int np , const int my_p , const int pbox[][3][2] , const int ghost , int ** map_local_id , int ** map_recv_pc , int ** map_send_pc , int ** map_send_id ) (defnt)
void box_partition_rcb( const int np , const int my_p , const int root_box[][2] , const int ghost , int (**pbox)[3][2] , int ** map_local_id , int ** map_recv_pc , int ** map_send_pc , int ** map_send_id ) (defnt)
static int box_contain( const int a[3][2] , const int b[3][2] ) (defnt)
static void box_print( FILE * fp , const int a[][2] ) (defnt)
static void test_box( const int box[3][2] , const int np ) (defnt)
static void test_maps( const int root_box[][2] , const int np ) (defnt)
int main( int argc , char * argv[] ) (defnt)""",
    "basic/optional/ThreadPool/test/hpccg/CGSolver.c": """static VECTOR_SCALAR comm_sum( VECTOR_SCALAR v ) (defnt)
static void comm_rhs_vector( const struct cgsolve_data * const data , VECTOR_SCALAR * const vec ) (defnt)
void cgsolve_set_lhs( const struct cgsolve_data * const data , const VECTOR_SCALAR * const x , VECTOR_SCALAR * const b ) (defnt)
void cgsolve( const struct cgsolve_data * const data , const VECTOR_SCALAR * const b , VECTOR_SCALAR * const x , int * const iter_count , VECTOR_SCALAR * const norm_resid , double * const dt_mxv , double * const dt_axpby , double * const dt_dot ) (defnt)""",
    "basic/optional/ThreadPool/test/hpccg/main.c": """static void hpccg_alloc_and_fill( const int np , const int my_p , const int gbox[][2] , const int ghost , struct cgsolve_data * const data ) (defnt)
int main( int argc , char ** argv ) (defnt)""",
    "basic/optional/ThreadPool/test/hpccg/tpi_vector.c": """static void tpi_work_span( TPI_Work * const work , const int n , int * const iBeg , int * const iEnd ) (defnt)
static void tpi_work_fill( TPI_Work * work ) (defnt)
void tpi_fill( int n , VECTOR_SCALAR alpha , VECTOR_SCALAR * x ) (defnt)
static void tpi_work_scale( TPI_Work * work ) (defnt)
void tpi_scale( int n , const VECTOR_SCALAR alpha , VECTOR_SCALAR * x ) (defnt)
static void tpi_work_copy( TPI_Work * work ) (defnt)
void tpi_copy( int n , const VECTOR_SCALAR * x , VECTOR_SCALAR * y ) (defnt)
static void tpi_work_axpby( TPI_Work * work ) (defnt)
void tpi_axpby( int n , VECTOR_SCALAR alpha , const VECTOR_SCALAR * x , VECTOR_SCALAR beta , VECTOR_SCALAR * y ) (defnt)
static void tpi_work_dot_partial( TPI_Work * work ) (defnt)
static void tpi_work_dot_partial_self( TPI_Work * work ) (defnt)
static void tpi_work_dot_join( TPI_Work * work , const void * src ) (defnt)
static void tpi_work_dot_init( TPI_Work * work ) (defnt)
VECTOR_SCALAR tpi_dot( int n , const VECTOR_SCALAR * x , const VECTOR_SCALAR * y ) (defnt)
static void tpi_work_crs_matrix_apply( TPI_Work * work ) (defnt)
void tpi_crs_matrix_apply( const int nRow , const int * A_pc , const int * A_ia , const MATRIX_SCALAR * A_a , const VECTOR_SCALAR * x , VECTOR_SCALAR * y ) (defnt)""",
    "basic/optional/ThreadPool/test/test_c_dnax.c": """static void test_dnax_column( const unsigned num_array , const unsigned stride , const unsigned length , const SCALAR * const coef , SCALAR * const array ) (defnt)
static void test_dnax_row( const unsigned num_array , const unsigned stride , const unsigned length , const SCALAR * const coef , SCALAR * const array ) (defnt)
static void test_dnax_flat_work( TPI_Work * work ) (defnt)
static void test_dnax_column_work( TPI_Work * work ) (defnt)
static void test_dnax_row_work( TPI_Work * work ) (defnt)
static void test_tpi_dnax_driver( const int nthread , const unsigned Mflop_target , const unsigned num_trials , const unsigned num_test , const unsigned num_test_array[] , const unsigned length_array , const unsigned length_chunk ) (defnt)
int test_c_tpi_dnax( int nthread , int ntrial ) (defnt)""",
    "basic/optional/ThreadPool/test/test_mpi_sum.c": """static void my_span( const unsigned count , const unsigned rank , const unsigned size , unsigned * begin , unsigned * length ) (defnt)
static void d2_add_d( double v[] , const double a ) (defnt)
void d4_dot( double v[] , unsigned n , const double * x , const double * y ) (defnt)
double ddot( unsigned n , const double * x , const double * y ) (defnt)
static void reduce_init( TPI_Work * work ) (defnt)
static void reduce_join( TPI_Work * work , const void * arg_src ) (defnt)
static void work_d4_dot_tp( TPI_Work * work ) (defnt)
double d4_dot_tp( COMM comm, unsigned nwork, unsigned n, const double * x, const double * y ) (defnt)
static void task_ddot_tp( TPI_Work * work ) (defnt)
double ddot_tp( COMM comm, unsigned nwork, unsigned n, const double * x, const double * y ) (defnt)
void dfill_rand( unsigned seed , unsigned n , double * x , double mag ) (defnt)
static void task_dfill_rand( TPI_Work * work ) (defnt)
void dfill_rand_tp( unsigned nblock , unsigned seed , unsigned n , double * x , double mag ) (defnt)
static void test_ddot_performance( COMM comm , const int nthreads , const int nblock , const unsigned int num_trials , const unsigned int num_tests , const unsigned int length_array[] /* Global array length for each test */ , const double mag ) (defnt)
static void test_ddot_accuracy( COMM comm , const int nthreads , const int nblock , const unsigned int num_tests , const unsigned int length_array[] /* Global array length for each test */ , const double mag ) (defnt)
static void test_performance( COMM comm , const int test_thread_count , const int test_thread[] ) (defnt)
static void test_accuracy( COMM comm , const int test_thread_count , const int test_thread[] , unsigned test_do ) (defnt)
int main( int argc , char **argv ) (defnt)
static int comm_size( COMM comm ) (defnt)
static int comm_rank( COMM comm ) (defnt)
static void comm_reduce_dmax( COMM comm , double * val ) (defnt)
static void comm_reduce_dsum( COMM comm , double * val ) (defnt)
static void comm_reduce_d4_op( void * argin , void * argout , int * n , MPI_Datatype * d ) (defnt)
static void comm_reduce_d4_sum( COMM comm , double * val ) (defnt)
int main( int argc , char **argv ) (defnt)
static int comm_size( COMM comm ) (defnt)
static int comm_rank( COMM comm ) (defnt)
static void comm_reduce_dmax( COMM comm , double * val ) (defnt)
static void comm_reduce_dsum( COMM comm , double * val ) (defnt)
static void comm_reduce_d4_sum( COMM comm , double * val ) (defnt)""",
"basic/optional/ThreadPool/test/test_pthreads.c": """static void * test_driver( void * arg ) (defnt)
static void test_run( pthread_attr_t * const thread_attr , const int number_threads , const int number_trials , const int number_loops , double * const dt_start_stop , double * const dt_loop ) (defnt)
static double test_mutex_init_destroy( const int number ) (defnt)
static double test_mutex_lock_unlock( const int number ) (defnt)
void test_pthreads_performance( int n_test , int * n_concurrent ) (defnt)""",
    "basic/optional/ThreadPool/test/test_tpi.cpp": """template <unsigned N> ~TEST() (defnt)
template <unsigned N> TEST<N>::TEST() (defnt)
template <unsigned N> void TEST<N>::flag( TPI::Work & work ) (defnt)
template <unsigned N> void TEST<N>::verify() (defnt)
void test_tpi_cpp( int np ) (defnt)
int main( int argc , char ** argv ) (defnt)""",
    "basic/optional/ThreadPool/test/test_tpi_unit.c": """void test_tpi_init( const int ntest , const int nthread[] , const int ntrial ) (defnt)
void test_tpi_block( const int ntest , const int nthread[] , const int ntrial ) (defnt)
void test_tpi_reduce( const int ntest , const int nthread[] , const int ntrial ) (defnt)
void test_tpi_work( const int ntest , const int nthread[] , const int nwork , const int ntrial ) (defnt)
void test_tpi_work_async( const int ntest , const int nthread[] , const int nwork , const int ntrial ) (defnt)
static void test_work( TPI_Work * work ) (defnt)
static void test_reduce_work( TPI_Work * work ) (defnt)
static void test_reduce_init( TPI_Work * work ) (defnt)
static void test_reduce_join( TPI_Work * work , const void * src ) (defnt)
static void test_reduce_via_lock( TPI_Work * work ) (defnt)
static void test_reduce_via_nolock( TPI_Work * work ) (defnt)""",
    "basic/optional/cuda/CudaCall.hpp": """inline void stk_cuda_call(cudaError err , const char* name ) (defnt)""",
    "basic/optional/cuda/CudaMemoryModel.hpp": """CudaMemoryModel() (defnt)
template <class T> inline T * CudaMemoryModel::get_buffer(const T* host_ptr, size_t buf_size) (defnt)
template <class T> inline void CudaMemoryModel::destroy_buffer(T*& device_ptr) (defnt)
template <class T> inline void CudaMemoryModel::copy_to_buffer(const T* host_ptr, size_t buf_size, T* device_ptr) (defnt)
template <class T> inline void CudaMemoryModel::copy_from_buffer(T* host_ptr, size_t buf_size, const T* device_ptr) (defnt)
inline CudaMemoryModel::~CudaMemoryModel() (defnt)""",
    "basic/optional/cuda/CudaNode.cpp": """CUDANode::CUDANode(int device, int numBlocks, int numThreads, int verbose) (defnt)
void CUDANode::expand_blk_mem(size_t size_in_bytes) (defnt)
CUDANode::~CUDANode() (defnt)""",
    "basic/optional/cuda/CudaNode.cuh": """template <class WDP> __global__ void Tkern1D(int length, WDP wd, int stride) (defnt)
template <class WDP> void CUDANode::parallel_for(int length, WDP wd) (defnt)
template <typename SCALAR> void call_dot(DotOp<SCALAR>& wd) (defnt)
template <> void call_dot(DotOp<double>& wd) (defnt)
template <> void call_dot(DotOp<float>& wd) (defnt)
template <class WDP> void CUDANode::parallel_reduce(int length, WDP& wd) (defnt)""",
    "basic/optional/cuda/CudaNode.hpp": """static CUDANode & singleton(int device=0, int numBlocks=-1, int numThreads=256) (defnt)""",
    "basic/optional/cuda/cutil_inline_runtime.h": """inline void __cudaSafeCallNoSync( cudaError err, const char *file, const int line ) (defnt)
inline void __cudaSafeCall( cudaError err, const char *file, const int line ) (defnt)
inline void __cudaSafeThreadSync( const char *file, const int line ) (defnt)
inline void __cutilCheckMsg( const char *errorMessage, const char *file, const int line ) (defnt)""",
    "basic/perform_element_loop.hpp": """template <typename GlobalOrdinal, typename MatrixType, typename VectorType> void perform_element_loop(const simple_mesh_description<GlobalOrdinal>& mesh, const Box& local_elem_box, MatrixType& A, VectorType& b, Parameters& /*params*/) (defnt)""",
    "basic/perform_element_loop_TBB_pipe.hpp": """template <typename GlobalOrdinal,typename Scalar> GetElemNodesCoords(const std::vector<GlobalOrdinal>& elemIDs, const simple_mesh_description<GlobalOrdinal>& mesh, size_t num_elems_at_a_time) (defnt)
template <typename GlobalOrdinal,typename Scalar> ~GetElemNodesCoords() (defnt)
template <typename GlobalOrdinal,typename Scalar> void * operator()(void* item) (defnt)
template <typename GlobalOrdinal,typename Scalar> Compute_FE_Operators() (defnt)
template <typename GlobalOrdinal,typename Scalar> ~Compute_FE_Operators() (defnt)
template <typename GlobalOrdinal,typename Scalar> void * operator()(void* item) (defnt)
template <typename MatrixType, typename VectorType> SumIntoLinearSystem(GlobalOrdinal myFirstRow, GlobalOrdinal myLastRow, MatrixType& mat, VectorType& vec) (defnt)
template <typename MatrixType, typename VectorType> ~SumIntoLinearSystem() (defnt)
template <typename MatrixType, typename VectorType> void * operator()(void* item) (defnt)
template <typename MatrixType, typename VectorType> LockingSumIntoLinearSystem(MatrixType& mat, VectorType& vec) (defnt)
template <typename MatrixType, typename VectorType> ~LockingSumIntoLinearSystem() (defnt)
template <typename MatrixType, typename VectorType> void * operator()(void* item) (defnt)
template <typename GlobalOrdinal, typename MatrixType, typename VectorType> void perform_element_loop(const simple_mesh_description<GlobalOrdinal>& mesh, const Box& local_elem_box, MatrixType& A, VectorType& b, Parameters& params) (defnt)""",
    "basic/perform_element_loop_TBB_pllfor1.hpp": """template <typename GlobalOrdinal,typename Scalar, typename MatrixType, typename VectorType> inline void operator()(int i) (defnt)
template <typename GlobalOrdinal, typename MatrixType, typename VectorType> void perform_element_loop(const simple_mesh_description<GlobalOrdinal>& mesh, const Box& local_elem_box, MatrixType& A, VectorType& b, Parameters& params) (defnt)""",
    "basic/perform_element_loop_TBB_pllfor2.hpp": """template <typename GlobalOrdinal, typename MatrixType, typename VectorType> void perform_element_loop(const simple_mesh_description<GlobalOrdinal>& mesh, const Box& local_elem_box, MatrixType& A, VectorType& b, Parameters& params) (defnt)""",
    "basic/sharedmem.cuh": """template <typename T> __device__ T * getPointer() (defnt)
template <> __device__ int * getPointer() (defnt)
template <> __device__ unsigned int * getPointer() (defnt)
template <> __device__ char * getPointer() (defnt)
template <> __device__ unsigned char * getPointer() (defnt)
template <> __device__ short * getPointer() (defnt)
template <> __device__ unsigned short * getPointer() (defnt)
template <> __device__ long * getPointer() (defnt)
template <> __device__ unsigned long * getPointer() (defnt)
template <> __device__ bool * getPointer() (defnt)
template <> __device__ float * getPointer() (defnt)
template <> __device__ double * getPointer() (defnt)""",
    "basic/simple_mesh_description.hpp": """template <typename GlobalOrdinal> simple_mesh_description(const Box& global_box_in, const Box& local_box_in) (defnt)
template <typename GlobalOrdinal> GlobalOrdinal map_id_to_row(const GlobalOrdinal& id) const (defnt)""",
    "basic/time_kernels.hpp": """template <typename OperatorType, typename VectorType, typename Matvec> void time_kernels(OperatorType& A, const VectorType& b, VectorType& x, Matvec matvec, typename OperatorType::LocalOrdinalType max_iter, typename OperatorType::ScalarType& xdotp, timer_type* my_kern_times) (defnt)""",
    "basic/utest.cpp": """int main(int argc, char** argv) (defnt)""",
    "basic/utest_case.hpp": """std::vector<utest_case*> & get_utest_cases() (defnt)
utest_case() (defnt)
~utest_case() (defnt)""",
    "basic/utest_cases.hpp": """template <typename T> inline int check_get_id(int nx, int ny, int nz, int x, int y, int z, T expected, const char* testname) (defnt)
UTEST_CASE(box_partition) (defnt)
UTEST_CASE(generate_matrix_structure1) (defnt)
UTEST_CASE(generate_matrix_structure2) (defnt)
UTEST_CASE(get_hex8_node_coords_3d) (defnt)
inline void get_test_elem_mat(std::vector<Scalar>& elem_mat) (defnt)
UTEST_CASE(diffusionMatrix) (defnt)
UTEST_CASE(sum_into_row) (defnt)
UTEST_CASE(sum_in_elem_matrix) (defnt)
UTEST_CASE(assemble_FE_data) (defnt)
UTEST_CASE(pll_matvec2) (defnt)
UTEST_CASE(pll_matvec3) (defnt)
UTEST_CASE(ComputeNode_waxpy1) (defnt)
UTEST_CASE(ComputeNode_dot1) (defnt)
UTEST_CASE(ComputeNode_TBB_dot1) (defnt)
UTEST_CASE(ComputeNode_dot2) (defnt)
UTEST_CASE(ser_matvec1) (defnt)
UTEST_CASE(waxpby_perf) (defnt)
UTEST_CASE(matmat3x3_1) (defnt)
UTEST_CASE(matmat3x3_X_3xn_1) (defnt)
UTEST_CASE(matTransMat3x3_X_3xn_1) (defnt)
UTEST_CASE(BoxIterator1) (defnt)
UTEST_CASE(BoxIterator_get_coords) (defnt)""",
    "basic/verify_solution.hpp": """template <typename VectorType> void verify_solution(const simple_mesh_description<typename VectorType::GlobalOrdinalType>& mesh, const VectorType& x) (defnt)""",
    "fem/ElemData.hpp": """template <typename GlobalOrdinal, typename Scalar> ElemData() (defnt)
template <typename GlobalOrdinal, typename Scalar> ~ElemData() (defnt)
template <typename GlobalOrdinal, typename Scalar> ElemDataPtr() (defnt)
template <typename GlobalOrdinal, typename Scalar> ~ElemDataPtr() (defnt)""",
    "fem/Hex8.hpp": """template <typename Scalar> KERNEL_PREFIX shape_fns(const Scalar* x, Scalar* values_at_nodes) (defnt)
template <typename Scalar> KERNEL_PREFIX gradients(const Scalar* x, Scalar* values_per_fn) (defnt)
template <typename Scalar> KERNEL_PREFIX gradients_and_detJ(const Scalar* elemNodeCoords, const Scalar* grad_vals, Scalar& detJ) (defnt)
template <typename Scalar> KERNEL_PREFIX gradients_and_invJ_and_detJ(const Scalar* elemNodeCoords, const Scalar* grad_vals, Scalar* invJ, Scalar& detJ) (defnt)
template <typename Scalar> KERNEL_PREFIX diffusionMatrix_symm(const Scalar* elemNodeCoords, const Scalar* grad_vals, Scalar* elem_mat) (defnt)
template <typename Scalar> KERNEL_PREFIX sourceVector(const Scalar* elemNodeCoords, const Scalar* grad_vals, Scalar* elem_vec) (defnt)""",
    "fem/Hex8_ElemData.hpp": """template <typename Scalar> void compute_gradient_values(Scalar* grad_vals) (defnt)
template <typename GlobalOrdinal,typename Scalar> void compute_element_matrix_and_vector(ElemData<GlobalOrdinal,Scalar>& elem_data) (defnt)
template <typename GlobalOrdinal,typename Scalar> void compute_element_matrix_and_vector(ElemDataPtr<GlobalOrdinal,Scalar>& elem_data) (defnt)""",
    "fem/analytic_soln.hpp": """inline Scalar fcn_l(int p, int q) (defnt)
inline Scalar fcn(int n, Scalar u) (defnt)
inline Scalar soln(Scalar x, Scalar y, Scalar z, int max_p, int max_q) (defnt)""",
    "fem/gauss_pts.hpp": """template <typename Scalar> inline KERNEL_PREFIX gauss_pts(int N, Scalar* pts, Scalar* wts) (defnt)""",
    "fem/matrix_algebra_3x3.hpp": """template <typename Scalar> __host__ __device__ __CUDACC__ fill(Scalar* begin, Scalar* end, const Scalar& val) (defnt)
template <typename Scalar> KERNEL_PREFIX inverse_and_determinant3x3(const Scalar* J, Scalar* invJ, Scalar& detJ) (defnt)
template <typename Scalar> KERNEL_PREFIX matmat3x3(const Scalar* A, const Scalar* B, Scalar* C) (defnt)
template <typename Scalar> KERNEL_PREFIX determinant3x3(const Scalar* J) (defnt)
template <typename Scalar> KERNEL_PREFIX matmat3x3_X_3xn(const Scalar* A, int n, const Scalar* B, Scalar* C) (defnt)
template <typename Scalar> KERNEL_PREFIX matTransMat3x3_X_3xn(const Scalar* A, int n, const Scalar* B, Scalar* C) (defnt)""",
    "fem/verify_solution.hpp": """template <typename VectorType> int verify_solution(const simple_mesh_description<typename VectorType::GlobalOrdinalType>& mesh, const VectorType& x, double tolerance, bool verify_whole_domain = false) (defnt)""",
    "src/CSRMatrix.hpp": """template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal> CSRMatrix() (defnt)
template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal> ~CSRMatrix() (defnt)
template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal> size_t num_nonzeros() const (defnt)
template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal> void reserve_space(unsigned nrows, unsigned ncols_per_row) (defnt)
template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal> void get_row_pointers(GlobalOrdinalType row, size_t& row_length, GlobalOrdinalType*& cols, ScalarType*& coefs) (defnt)""",
    "src/ELLMatrix.hpp": """template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal> ELLMatrix() (defnt)
template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal> ~ELLMatrix() (defnt)
template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal> size_t num_nonzeros() const (defnt)
template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal> void reserve_space(unsigned nrows, unsigned ncols_per_row) (defnt)
template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal> void get_row_pointers(GlobalOrdinalType row, size_t& row_length, GlobalOrdinalType*& cols_ptr, ScalarType*& coefs_ptr) (defnt)""",
    "src/GetNodesCoords.hpp": """template <typename GlobalOrdinal,typename Scalar> inline void operator()(int i) (defnt)""",
    "src/Hex8_box_utils.hpp": """template <typename GlobalOrdinal> void get_hex8_node_ids(int nx, int ny, GlobalOrdinal node0, GlobalOrdinal* elem_node_ids) (defnt)
template <typename Scalar> void get_hex8_node_coords_3d(Scalar x, Scalar y, Scalar z, Scalar hx, Scalar hy, Scalar hz, Scalar* elem_node_coords) (defnt)
template <typename GlobalOrdinal, typename Scalar> void get_elem_nodes_and_coords(const simple_mesh_description<GlobalOrdinal>& mesh, GlobalOrdinal elemID, GlobalOrdinal* node_ords, Scalar* node_coords) (defnt)
template <typename GlobalOrdinal, typename Scalar> void get_elem_nodes_and_coords(const simple_mesh_description<GlobalOrdinal>& mesh, GlobalOrdinal elemID, ElemData<GlobalOrdinal,Scalar>& elem_data) (defnt)""",
    "src/MatrixCopyOp.hpp": """template <typename MatrixType> inline void operator()(int i) (defnt)""",
    "src/MatrixInitOp.hpp": """template <typename GlobalOrdinal> void sort_if_needed(GlobalOrdinal* list, GlobalOrdinal list_len) (defnt)
template <> MatrixInitOp(const std::vector<MINIFE_GLOBAL_ORDINAL>& rows_vec, const std::vector<MINIFE_LOCAL_ORDINAL>& row_offsets_vec, const std::vector<int>& row_coords_vec, int global_nx, int global_ny, int global_nz, MINIFE_GLOBAL_ORDINAL global_n_rows, const miniFE::simple_mesh_description<MINIFE_GLOBAL_ORDINAL>& input_mesh, miniFE::CSRMatrix<MINIFE_SCALAR,MINIFE_LOCAL_ORDINAL,MINIFE_GLOBAL_ORDINAL>& matrix) (defnt)
template <> inline void operator()(int i) (defnt)
template <> MatrixInitOp(const std::vector<MINIFE_GLOBAL_ORDINAL>& rows_vec, const std::vector<MINIFE_LOCAL_ORDINAL>& /*row_offsets_vec*/, const std::vector<int>& row_coords_vec, int global_nx, int global_ny, int global_nz, MINIFE_GLOBAL_ORDINAL global_n_rows, const miniFE::simple_mesh_description<MINIFE_GLOBAL_ORDINAL>& input_mesh, miniFE::ELLMatrix<MINIFE_SCALAR,MINIFE_LOCAL_ORDINAL,MINIFE_GLOBAL_ORDINAL>& matrix) (defnt)
template <> inline void operator()(int i) (defnt)""",
    "src/SparseMatrix_functions.hpp": """template <typename MatrixType> void init_matrix(MatrixType& M, const std::vector<typename MatrixType::GlobalOrdinalType>& rows, const std::vector<typename MatrixType::LocalOrdinalType>& row_offsets, const std::vector<int>& row_coords, int global_nodes_x, int global_nodes_y, int global_nodes_z, typename MatrixType::GlobalOrdinalType global_nrows, const simple_mesh_description<typename MatrixType::GlobalOrdinalType>& mesh) (defnt)
template <typename T, typename U> void sort_with_companions(ptrdiff_t len, T* array, U* companions) (defnt)
template <typename MatrixType> void write_matrix(const std::string& filename, MatrixType& mat) (defnt)
template <typename GlobalOrdinal,typename Scalar> void sum_into_row(int row_len, GlobalOrdinal* row_indices, Scalar* row_coefs, int num_inputs, const GlobalOrdinal* input_indices, const Scalar* input_coefs) (defnt)
template <typename MatrixType> void sum_into_row(typename MatrixType::GlobalOrdinalType row, size_t num_indices, const typename MatrixType::GlobalOrdinalType* col_inds, const typename MatrixType::ScalarType* coefs, MatrixType& mat) (defnt)
template <typename MatrixType> void sum_in_symm_elem_matrix(size_t num, const typename MatrixType::GlobalOrdinalType* indices, const typename MatrixType::ScalarType* coefs, MatrixType& mat) (defnt)
template <typename MatrixType> void sum_in_elem_matrix(size_t num, const typename MatrixType::GlobalOrdinalType* indices, const typename MatrixType::ScalarType* coefs, MatrixType& mat) (defnt)
template <typename GlobalOrdinal, typename Scalar, typename MatrixType, typename VectorType> void sum_into_global_linear_system(ElemData<GlobalOrdinal,Scalar>& elem_data, MatrixType& A, VectorType& b) (defnt)
template <typename MatrixType> void sum_in_elem_matrix(size_t num, const typename MatrixType::GlobalOrdinalType* indices, const typename MatrixType::ScalarType* coefs, LockingMatrix<MatrixType>& mat) (defnt)
template <typename GlobalOrdinal, typename Scalar, typename MatrixType, typename VectorType> void sum_into_global_linear_system(ElemData<GlobalOrdinal,Scalar>& elem_data, LockingMatrix<MatrixType>& A, LockingVector<VectorType>& b) (defnt)
template <typename MatrixType> void add_to_diagonal(typename MatrixType::ScalarType value, MatrixType& mat) (defnt)
template <typename MatrixType> double parallel_memory_overhead_MB(const MatrixType& A) (defnt)
template <typename MatrixType> void rearrange_matrix_local_external(MatrixType& A) (defnt)
template <typename MatrixType> void zero_row_and_put_1_on_diagonal(MatrixType& A, typename MatrixType::GlobalOrdinalType row) (defnt)
template <typename MatrixType, typename VectorType> void impose_dirichlet(typename MatrixType::ScalarType prescribed_value, MatrixType& A, VectorType& b, int global_nx, int global_ny, int global_nz, const std::set<typename MatrixType::GlobalOrdinalType>& bc_rows) (defnt)
template <typename MatrixType> __global__ void matvec_kernel(const MINIFE_LOCAL_ORDINAL rows_size, const typename MatrixType::LocalOrdinalType *Arowoffsets, const typename MatrixType::GlobalOrdinalType *Acols, const typename MatrixType::ScalarType *Acoefs, const typename MatrixType::ScalarType *xcoefs, typename MatrixType::ScalarType *ycoefs) (defnt)
template <typename MatrixType, typename VectorType> void operator()(MatrixType& A, VectorType& x, VectorType& y, const typename MatrixType::LocalOrdinalType *d_Arowoffsets, const typename MatrixType::GlobalOrdinalType *d_Acols, const typename MatrixType::ScalarType *d_Acoefs, const typename MatrixType::ScalarType *d_xcoefs, typename MatrixType::ScalarType *d_ycoefs) (defnt)
template <typename MatrixType, typename VectorType> void operator()(MatrixType& A, VectorType& x, VectorType& y) (defnt)
template <typename MatrixType, typename VectorType> void matvec(MatrixType& A, VectorType& x, VectorType& y, typename MatrixType::LocalOrdinalType *d_Arowoffsets, typename MatrixType::GlobalOrdinalType *d_Acols, typename MatrixType::ScalarType *d_Acoefs, typename MatrixType::ScalarType *d_xcoefs, typename MatrixType::ScalarType *d_ycoefs) (defnt)
template <typename MatrixType, typename VectorType> void operator()(MatrixType& A, VectorType& x, VectorType& y, typename MatrixType::LocalOrdinalType *d_Arowoffsets, typename MatrixType::GlobalOrdinalType *d_Acols, typename MatrixType::ScalarType *d_Acoefs, typename MatrixType::ScalarType *d_xcoefs, typename MatrixType::ScalarType *d_ycoefs) (defnt)""",
    "src/Vector.hpp": """template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal> Vector(GlobalOrdinal startIdx, LocalOrdinal local_sz) (defnt)
template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal> ~Vector() (defnt)""",
    "src/Vector_functions.hpp": """template <typename VectorType> void write_vector(const std::string& filename, const VectorType& vec) (defnt)
template <typename VectorType> void sum_into_vector(size_t num_indices, const typename VectorType::GlobalOrdinalType* indices, const typename VectorType::ScalarType* coefs, VectorType& vec) (defnt)
template <typename VectorType> void sum_into_vector(size_t num_indices, const typename VectorType::GlobalOrdinalType* indices, const typename VectorType::ScalarType* coefs, LockingVector<VectorType>& vec) (defnt)
template <typename VectorType> __global__ void waxby_kernel( const int n, const typename VectorType::ScalarType alpha, const typename VectorType::ScalarType *xcoefs, const typename VectorType::ScalarType beta, const typename VectorType::ScalarType *ycoefs, typename VectorType::ScalarType *wcoefs) (defnt)
template <typename VectorType> __global__ void yaxby_kernel( const int n, const typename VectorType::ScalarType alpha, const typename VectorType::ScalarType *xcoefs, const typename VectorType::ScalarType beta, typename VectorType::ScalarType *ycoefs ) (defnt)
template <typename VectorType> __global__ void wxby_kernel( const int n, const typename VectorType::ScalarType *xcoefs, const typename VectorType::ScalarType beta, const typename VectorType::ScalarType *ycoefs, typename VectorType::ScalarType *wcoefs) (defnt)
template <typename VectorType> __global__ void yxby_kernel( const int n, const typename VectorType::ScalarType *xcoefs, const typename VectorType::ScalarType beta, typename VectorType::ScalarType *ycoefs) (defnt)
template <typename VectorType> __global__ void wx_kernel( const int n, const typename VectorType::ScalarType *xcoefs, typename VectorType::ScalarType *wcoefs) (defnt)
template <typename VectorType> __global__ void dyx_kernel( const int n, const typename VectorType::ScalarType *xcoefs, typename VectorType::ScalarType *ycoefs) (defnt)
template <typename VectorType> __global__ void wax_kernel( const int n, const typename VectorType::ScalarType alpha, const typename VectorType::ScalarType *xcoefs, typename VectorType::ScalarType *wcoefs) (defnt)
template <typename VectorType> __global__ void dyax_kernel( const int n, const typename VectorType::ScalarType alpha, const typename VectorType::ScalarType *xcoefs, typename VectorType::ScalarType *ycoefs) (defnt)
template <typename VectorType> void waxpby(typename VectorType::ScalarType alpha, const VectorType& x, typename VectorType::ScalarType beta, const VectorType& y, VectorType& w, typename VectorType::ScalarType *d_xcoefs, typename VectorType::ScalarType *d_ycoefs, typename VectorType::ScalarType *d_wcoefs) (defnt)
template <typename VectorType> void daxpby(const MINIFE_SCALAR alpha, const VectorType& x, const MINIFE_SCALAR beta, VectorType& y, MINIFE_SCALAR *d_xcoefs, MINIFE_SCALAR *d_ycoefs) (defnt)
template <typename Scalar> __global__ void dot_kernel(const MINIFE_LOCAL_ORDINAL n, const Scalar* x, const Scalar* y, Scalar* d) (defnt)
template <typename Scalar> __global__ void final_reduce(Scalar *d) (defnt)
template <typename Vector> typename TypeTraits<typename Vector::ScalarType>::magnitude_type dot(const Vector& x, const Vector& y, typename Vector::ScalarType *d_xcoefs, typename Vector::ScalarType *d_ycoefs) (defnt)
template <typename Vector> typename TypeTraits<typename Vector::ScalarType>::magnitude_type dot_r2(const Vector& x, const typename Vector::ScalarType *d_xcoefs) (defnt)""",
    "src/YAML_Doc.cpp": """YAML_Doc::YAML_Doc(const std::string& miniApp_Name, const std::string& miniApp_Version, const std::string& destination_Directory, const std::string& destination_FileName) (defnt)
YAML_Doc::~YAML_Doc(void) (defnt)
string YAML_Doc::generateYAML() (defnt)""",
    "src/YAML_Element.cpp": """YAML_Element::YAML_Element(const std::string& key_arg, const std::string& value_arg) (defnt)
YAML_Element::~YAML_Element() (defnt)
YAML_Element * YAML_Element::add(const std::string& key_arg, double value_arg) (defnt)
YAML_Element * YAML_Element::add(const std::string& key_arg, int value_arg) (defnt)
YAML_Element * YAML_Element::add(const std::string& key_arg, long long value_arg) (defnt)
YAML_Element * YAML_Element::add(const std::string& key_arg, size_t value_arg) (defnt)
YAML_Element * YAML_Element::add(const std::string& key_arg, const std::string& value_arg) (defnt)
YAML_Element * YAML_Element::get(const std::string& key_arg) (defnt)
string YAML_Element::printYAML(std::string space) (defnt)
string YAML_Element::convert_double_to_string(double value_arg) (defnt)
string YAML_Element::convert_int_to_string(int value_arg) (defnt)
string YAML_Element::convert_long_long_to_string(long long value_arg) (defnt)
string YAML_Element::convert_size_t_to_string(size_t value_arg) (defnt)""",
    "src/YAML_Element.hpp": """YAML_Element () (defnt)
std::string getKey() (defnt)""",
    "src/assemble_FE_data.hpp": """template <typename MatrixType, typename VectorType> void assemble_FE_data(const simple_mesh_description<typename MatrixType::GlobalOrdinalType>& mesh, MatrixType& A, VectorType& b, Parameters& params) (defnt)""",
    "src/cg_solve.hpp": """template <typename Scalar> void print_vec(const std::vector<Scalar>& vec, const std::string& name) (defnt)
template <typename VectorType> bool breakdown(typename VectorType::ScalarType inner, const VectorType& v, const VectorType& w, typename VectorType::ScalarType *d_v, typename VectorType::ScalarType *d_w) (defnt)
template <typename OperatorType, typename VectorType, typename Matvec> void cg_solve(OperatorType& A, const VectorType& b, VectorType& x, Matvec matvec, typename OperatorType::LocalOrdinalType max_iter, typename TypeTraits<typename OperatorType::ScalarType>::magnitude_type& tolerance, typename OperatorType::LocalOrdinalType& num_iters, typename TypeTraits<typename OperatorType::ScalarType>::magnitude_type& normr, timer_type* my_cg_times) (defnt)""",
    "src/driver.hpp": """template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal> int driver(const Box& global_box, Box& my_box, Parameters& params, YAML_Doc& ydoc) (defnt)""",
    "src/exchange_externals.hpp": """template <typename MatrixType, typename VectorType> void exchange_externals(MatrixType& A, VectorType& x) (defnt)
template <typename MatrixType, typename VectorType> void begin_exchange_externals(MatrixType& A, VectorType& x) (defnt)
inline void finish_exchange_externals(int num_neighbors) (defnt)""",
    "src/generate_matrix_structure.hpp": """template <typename MatrixType> int generate_matrix_structure(const simple_mesh_description<typename MatrixType::GlobalOrdinalType>& mesh, MatrixType& A) (defnt)""",
    "src/main.cpp": """int main(int argc, char** argv) (defnt)
void add_params_to_yaml(YAML_Doc& doc, miniFE::Parameters& params) (defnt)
void add_configuration_to_yaml(YAML_Doc& doc, int numprocs, int numthreads) (defnt)
void add_timestring_to_yaml(YAML_Doc& doc) (defnt)""",
    "src/make_local_matrix.hpp": """template <typename MatrixType> void make_local_matrix(MatrixType& A) (defnt)""",
    "src/omp-SparseMatrix_functions.hpp": """template <typename MatrixType> void init_matrix(MatrixType& M, const std::vector<typename MatrixType::GlobalOrdinalType>& rows, const std::vector<typename MatrixType::LocalOrdinalType>& row_offsets, const std::vector<int>& row_coords, int global_nodes_x, int global_nodes_y, int global_nodes_z, typename MatrixType::GlobalOrdinalType global_nrows, const simple_mesh_description<typename MatrixType::GlobalOrdinalType>& mesh) (defnt)
template <typename T, typename U> void sort_with_companions(ptrdiff_t len, T* array, U* companions) (defnt)
template <typename MatrixType> void write_matrix(const std::string& filename, MatrixType& mat) (defnt)
template <typename GlobalOrdinal,typename Scalar> void sum_into_row(int row_len, GlobalOrdinal* row_indices, Scalar* row_coefs, int num_inputs, const GlobalOrdinal* input_indices, const Scalar* input_coefs) (defnt)
template <typename MatrixType> void sum_into_row(typename MatrixType::GlobalOrdinalType row, size_t num_indices, const typename MatrixType::GlobalOrdinalType* col_inds, const typename MatrixType::ScalarType* coefs, MatrixType& mat) (defnt)
template <typename MatrixType> void sum_in_symm_elem_matrix(size_t num, const typename MatrixType::GlobalOrdinalType* indices, const typename MatrixType::ScalarType* coefs, MatrixType& mat) (defnt)
template <typename MatrixType> void sum_in_elem_matrix(size_t num, const typename MatrixType::GlobalOrdinalType* indices, const typename MatrixType::ScalarType* coefs, MatrixType& mat) (defnt)
template <typename GlobalOrdinal, typename Scalar, typename MatrixType, typename VectorType> void sum_into_global_linear_system(ElemData<GlobalOrdinal,Scalar>& elem_data, MatrixType& A, VectorType& b) (defnt)
template <typename MatrixType> void sum_in_elem_matrix(size_t num, const typename MatrixType::GlobalOrdinalType* indices, const typename MatrixType::ScalarType* coefs, LockingMatrix<MatrixType>& mat) (defnt)
template <typename GlobalOrdinal, typename Scalar, typename MatrixType, typename VectorType> void sum_into_global_linear_system(ElemData<GlobalOrdinal,Scalar>& elem_data, LockingMatrix<MatrixType>& A, LockingVector<VectorType>& b) (defnt)
template <typename MatrixType> void add_to_diagonal(typename MatrixType::ScalarType value, MatrixType& mat) (defnt)
template <typename MatrixType> double parallel_memory_overhead_MB(const MatrixType& A) (defnt)
template <typename MatrixType> void rearrange_matrix_local_external(MatrixType& A) (defnt)
template <typename MatrixType> void zero_row_and_put_1_on_diagonal(MatrixType& A, typename MatrixType::GlobalOrdinalType row) (defnt)
template <typename MatrixType, typename VectorType> void impose_dirichlet(typename MatrixType::ScalarType prescribed_value, MatrixType& A, VectorType& b, int global_nx, int global_ny, int global_nz, const std::set<typename MatrixType::GlobalOrdinalType>& bc_rows) (defnt)
template <typename MatrixType, typename VectorType> void operator()(MatrixType& A, VectorType& x, VectorType& y) (defnt)
template <typename MatrixType, typename VectorType> void operator()(MatrixType& A, VectorType& x, VectorType& y) (defnt)
template <typename MatrixType, typename VectorType> void matvec(MatrixType& A, VectorType& x, VectorType& y) (defnt)
template <typename MatrixType, typename VectorType> void operator()(MatrixType& A, VectorType& x, VectorType& y) (defnt)""",
    "src/omp-Vector_functions.hpp": """template <typename VectorType> void write_vector(const std::string& filename, const VectorType& vec) (defnt)
template <typename VectorType> void sum_into_vector(size_t num_indices, const typename VectorType::GlobalOrdinalType* indices, const typename VectorType::ScalarType* coefs, VectorType& vec) (defnt)
template <typename VectorType> void sum_into_vector(size_t num_indices, const typename VectorType::GlobalOrdinalType* indices, const typename VectorType::ScalarType* coefs, LockingVector<VectorType>& vec) (defnt)
template <typename VectorType> void waxpby(typename VectorType::ScalarType alpha, const VectorType& x, typename VectorType::ScalarType beta, const VectorType& y, VectorType& w) (defnt)
template <typename VectorType> void daxpby(const MINIFE_SCALAR alpha, const VectorType& x, const MINIFE_SCALAR beta, VectorType& y) (defnt)
template <typename Vector> typename TypeTraits<typename Vector::ScalarType>::magnitude_type dot(const Vector& x, const Vector& y) (defnt)
template <typename Vector> typename TypeTraits<typename Vector::ScalarType>::magnitude_type dot_r2(const Vector& x) (defnt)""",
    "src/perform_element_loop.hpp": """template <typename GlobalOrdinal, typename MatrixType, typename VectorType> void perform_element_loop(const simple_mesh_description<GlobalOrdinal>& mesh, const Box& local_elem_box, MatrixType& A, VectorType& b, Parameters& /*params*/) (defnt)""",
    "src/simple_mesh_description.hpp": """template <typename GlobalOrdinal> simple_mesh_description(const Box& global_box_in, const Box& local_box_in) (defnt)
template <typename GlobalOrdinal> GlobalOrdinal map_id_to_row(const GlobalOrdinal& id) const (defnt)
template <typename GlobalOrdinal> GlobalOrdinal max_row_in_map() const (defnt)""",
    "src/time_kernels.hpp": """template <typename OperatorType, typename VectorType, typename Matvec> void time_kernels(OperatorType& A, const VectorType& b, VectorType& x, Matvec matvec, typename OperatorType::LocalOrdinalType max_iter, typename OperatorType::ScalarType& xdotp, timer_type* my_kern_times) (defnt)""",
    "utils/Box.hpp": """__host__ __device__ int * operator[](int xyz) (defnt)
__host__ __device__ int * operator[](int xyz) const (defnt)""",
"utils/BoxIterator.hpp": """~BoxIterator() (defnt)
static BoxIterator begin(const Box& box) (defnt)
static BoxIterator end(const Box& box) (defnt)
BoxIterator & operator=(const BoxIterator& src) (defnt)
BoxIterator & operator++() (defnt)
BoxIterator operator++(int) (defnt)
bool operator==(const BoxIterator& rhs) const (defnt)
bool operator!=(const BoxIterator& rhs) const (defnt)
BoxIterator(const Box& box, bool at_end = false) (defnt)""",
"utils/BoxPartition.cpp": """static int box_map_local_entry( const Box& box , const int ghost , int local_x , int local_y , int local_z ) (defnt)
int box_map_local( const Box& box_local, const int ghost , const int box_local_map[] , const int local_x , const int local_y , const int local_z ) (defnt)
void box_partition( int ip , int up , int axis , const Box& box, Box* p_box ) (defnt)
static int box_disjoint( const Box& a , const Box& b) (defnt)
static void resize_int( int ** a , int * allocLen , int newLen ) (defnt)
static void box_partition_maps( const int np , const int my_p , const Box* pbox, const int ghost , int ** map_local_id , int ** map_recv_pc , int ** map_send_pc , int ** map_send_id ) (defnt)
void box_partition_rcb( const int np , const int my_p , const Box& root_box, const int ghost , Box** pbox, int ** map_local_id , int ** map_recv_pc , int ** map_send_pc , int ** map_send_id ) (defnt)
static int box_contain( const Box& a , const Box& b ) (defnt)
static void box_print( FILE * fp , const Box& a ) (defnt)
static void test_box( const Box& box , const int np ) (defnt)
static void test_maps( const Box& root_box , const int np ) (defnt)
int main( int argc , char * argv[] ) (defnt)""",
    "utils/Parameters.hpp": """Parameters() (defnt)""",
    "utils/TypeTraits.hpp": """template <> static char * name() (defnt)
template <> static MPI_Datatype mpi_type() (defnt)
template <> static char * name() (defnt)
template <> static MPI_Datatype mpi_type() (defnt)
template <> static char * name() (defnt)
template <> static MPI_Datatype mpi_type() (defnt)
template <> static char * name() (defnt)
template <> static MPI_Datatype mpi_type() (defnt)
template <> static char * name() (defnt)
template <> static MPI_Datatype mpi_type() (defnt)
template <> static char * name() (defnt)
template <> static MPI_Datatype mpi_type() (defnt)
template <> static char * name() (defnt)
template <> static MPI_Datatype mpi_type() (defnt)
template <> static char * name() (defnt)
template <> static MPI_Datatype mpi_type() (defnt)""",
    "utils/box_utils.hpp": """inline void copy_box(const Box& from_box, Box& to_box) (defnt)
template <typename GlobalOrdinal> void get_int_coords(GlobalOrdinal ID, int nx, int ny, int nz, int& x, int& y, int& z) (defnt)
template <typename GlobalOrdinal,typename Scalar> void get_coords(GlobalOrdinal ID, int nx, int ny, int nz, Scalar& x, Scalar& y, Scalar& z) (defnt)
template <typename GlobalOrdinal> GlobalOrdinal get_num_ids(const Box& box) (defnt)
template <typename GlobalOrdinal> __host__ __device__ __CUDACC__ get_id(int nx, int ny, int nz, int x, int y, int z) (defnt)
template <typename GlobalOrdinal> void get_ids(int nx, int ny, int nz, const Box& box, std::vector<GlobalOrdinal>& ids, bool include_ghost_layer=false) (defnt)
template <typename GlobalOrdinal> void get_ghost_ids(int nx, int ny, int nz, const Box& box, std::vector<GlobalOrdinal>& ids) (defnt)
inline void print_box(int myproc, const char* name, const Box& box, const char* name2, const Box& box2) (defnt)
bool is_neighbor(const Box& box1, const Box& box2) (defnt)
template <typename GlobalOrdinal> void create_map_id_to_row(int global_nx, int global_ny, int global_nz, const Box& box, std::map<GlobalOrdinal,GlobalOrdinal>& id_to_row) (defnt)""",
    "utils/compute_matrix_stats.hpp": """template <typename MatrixType> size_t compute_matrix_stats(const MatrixType& A, int myproc, int numprocs, YAML_Doc& ydoc) (defnt)""",
    "utils/imbalance.hpp": """template <typename GlobalOrdinal> void compute_imbalance(const Box& global_box, const Box& local_box, float& largest_imbalance, float& std_dev, YAML_Doc& doc, bool record_in_doc) (defnt)
std::pair<int,int> decide_how_to_grow(const Box& global_box, const Box& local_box) (defnt)
std::pair<int,int> decide_how_to_shrink(const Box& global_box, const Box& local_box) (defnt)
template <typename GlobalOrdinal> void add_imbalance(const Box& global_box, Box& local_box, float imbalance, YAML_Doc& doc) (defnt)""",
    "utils/mytimer.cpp": """timer_type mytimer() (defnt)
timer_type mytimer(void) (defnt)
timer_type mytimer(void) (defnt)
timer_type mytimer(void) (defnt)
timer_type mytimer(void) (defnt)""",
    "utils/outstream.hpp": """inline std::ostream & outstream(int np=1, int p=0) (defnt)""",
    "utils/param_utils.cpp": """void read_args_into_string(int argc, char** argv, std::string& arg_string) (defnt)
void read_file_into_string(const std::string& filename, std::string& file_contents) (defnt)""",
    "utils/param_utils.hpp": """template <typename T> T parse_parameter(const std::string& arg_string, const std::string& param_name, const T& default_value) (defnt)""",
    "utils/utils.cpp": """void get_parameters(int argc, char** argv, Parameters& params) (defnt)
void broadcast_parameters(Parameters& params) (defnt)
void initialize_mpi(int argc, char** argv, int& numprocs, int& myproc) (defnt)
void finalize_mpi() (defnt)""",
    "utils/utils.hpp": """template <typename Scalar> Scalar percentage_difference(Scalar value, Scalar average) (defnt)
template <typename GlobalOrdinal> void get_global_min_max(GlobalOrdinal local_n, GlobalOrdinal& global_n, GlobalOrdinal& min_n, int& min_proc, GlobalOrdinal& max_n, int& max_proc) (defnt)
template <typename Scalar> Scalar compute_std_dev_as_percentage(Scalar local_nrows, Scalar avg_nrows) (defnt)
template <typename GlobalOrdinal> GlobalOrdinal find_row_for_id(GlobalOrdinal id, const std::map<GlobalOrdinal,GlobalOrdinal>& ids_to_rows) (defnt)""",
}


EXPECTED_FUNCTION_DECLARATIONS = {
    "basic/optional/ThreadPool/test/test_tpi.cpp": """template <unsigned N> void flag( TPI::Work & ) (decl)
template <unsigned N> void verify() (decl)""",
    "basic/optional/cuda/CudaNode.hpp": """void expand_blk_mem(size_t size_in_bytes) (decl)""",
    "basic/utest_case.hpp": """bool run() (decl)""",
    "src/YAML_Doc.hpp": """std::string generateYAML() (decl)""",
    "src/YAML_Element.hpp": """std::string printYAML(std::string space) (decl)
std::string convert_double_to_string(double value_arg) (decl)
std::string convert_int_to_string(int value_arg) (decl)
std::string convert_long_long_to_string(long long value_arg) (decl)
std::string convert_size_t_to_string(size_t value_arg) (decl)""",
}


EXPECTED_INCLUDE_TREES = {
    "basic/BoxPartition.cpp": """basic/BoxPartition.cpp
  #include <stdio.h> (DNE)
  #include <stdlib.h> (DNE)
  #include <Box.hpp>
  #include <BoxPartition.hpp>
    #include <Box.hpp>

""",
    "basic/main.cpp": """basic/main.cpp
  #include <iostream> (DNE)
  #include <ctime> (DNE)
  #include <cstdlib> (DNE)
  #include <vector> (DNE)
  #include <miniFE_version.h> (DNE)
  #include <outstream.hpp> (DNE)
  #include <mpi.h> (DNE)
  #include <ComputeNodeType.hpp>
    #include <tbb/task_scheduler_init.h> (DNE)
    #include <TBBNode.hpp>
      #include <tbb/blocked_range.h> (DNE)
      #include <tbb/parallel_for.h> (DNE)
      #include <tbb/parallel_reduce.h> (DNE)
      #include <tbb/task_scheduler_init.h> (DNE)
      #include <stdlib.h> (DNE)
      #include <NoOpMemoryModel.hpp>
      #include <iostream> (DNE)
    #include <TPI.h> (DNE)
    #include <TPINode.hpp>
      #include <TPI.h> (DNE)
      #include <NoOpMemoryModel.hpp>
      #include <iostream> (DNE)
    #include <CudaNode.hpp> (DNE)
    #include <SerialComputeNode.hpp>
      #include <NoOpMemoryModel.hpp>
  #include <Box.hpp>
  #include <BoxPartition.hpp>
    #include <Box.hpp>
  #include <box_utils.hpp>
    #include <vector> (DNE)
    #include <map> (DNE)
    #include <mpi.h> (DNE)
    #include <TypeTraits.hpp>
      #include <complex> (DNE)
      #include <mpi.h> (DNE)
    #include <Box.hpp>
  #include <Parameters.hpp> (DNE)
  #include <utils.hpp> (DNE)
  #include <driver.hpp>
    #include <cstddef> (DNE)
    #include <cmath> (DNE)
    #include <cstdlib> (DNE)
    #include <iostream> (DNE)
    #include <sstream> (DNE)
    #include <iomanip> (DNE)
    #include <box_utils.hpp>
      #include <vector> (DNE)
      #include <map> (DNE)
      #include <mpi.h> (DNE)
      #include <TypeTraits.hpp>
        #include <complex> (DNE)
        #include <mpi.h> (DNE)
      #include <Box.hpp>
    #include <Vector.hpp>
      #include <vector> (DNE)
      #include <MemInitOp.hpp>
    #include <CSRMatrix.hpp>
      #include <cstddef> (DNE)
      #include <vector> (DNE)
      #include <algorithm> (DNE)
      #include <mpi.h> (DNE)
    #include <ELLMatrix.hpp>
      #include <cstddef> (DNE)
      #include <vector> (DNE)
      #include <algorithm> (DNE)
      #include <mpi.h> (DNE)
    #include <CSRMatrix.hpp>
      #include <cstddef> (DNE)
      #include <vector> (DNE)
      #include <algorithm> (DNE)
      #include <mpi.h> (DNE)
    #include <simple_mesh_description.hpp>
      #include <utils.hpp> (DNE)
      #include <set> (DNE)
      #include <map> (DNE)
    #include <SparseMatrix_functions.hpp>
      #include <cstddef> (DNE)
      #include <vector> (DNE)
      #include <set> (DNE)
      #include <algorithm> (DNE)
      #include <sstream> (DNE)
      #include <fstream> (DNE)
      #include <Vector.hpp>
        #include <vector> (DNE)
        #include <MemInitOp.hpp>
      #include <Vector_functions.hpp>
        #include <vector> (DNE)
        #include <sstream> (DNE)
        #include <fstream> (DNE)
        #include <mpi.h> (DNE)
        #include <LockingVector.hpp>
          #include <vector> (DNE)
          #include <Lock.hpp>
            #include <iostream> (DNE)
            #include <tbb/atomic.h> (DNE)
        #include <TypeTraits.hpp>
          #include <complex> (DNE)
          #include <mpi.h> (DNE)
        #include <Vector.hpp>
          #include <vector> (DNE)
          #include <MemInitOp.hpp>
        #include <WaxpbyOp.hpp>
        #include <DotOp.hpp>
      #include <ElemData.hpp> (DNE)
      #include <FusedMatvecDotOp.hpp>
      #include <MatvecOp.hpp>
        #include <CSRMatrix.hpp>
          #include <cstddef> (DNE)
          #include <vector> (DNE)
          #include <algorithm> (DNE)
          #include <mpi.h> (DNE)
        #include <ELLMatrix.hpp>
          #include <cstddef> (DNE)
          #include <vector> (DNE)
          #include <algorithm> (DNE)
          #include <mpi.h> (DNE)
        #include <ComputeNodeType.hpp>
          #include <tbb/task_scheduler_init.h> (DNE)
          #include <TBBNode.hpp>
            #include <tbb/blocked_range.h> (DNE)
            #include <tbb/parallel_for.h> (DNE)
            #include <tbb/parallel_reduce.h> (DNE)
            #include <tbb/task_scheduler_init.h> (DNE)
            #include <stdlib.h> (DNE)
            #include <NoOpMemoryModel.hpp>
            #include <iostream> (DNE)
          #include <TPI.h> (DNE)
          #include <TPINode.hpp>
            #include <TPI.h> (DNE)
            #include <NoOpMemoryModel.hpp>
            #include <iostream> (DNE)
          #include <CudaNode.hpp> (DNE)
          #include <SerialComputeNode.hpp>
            #include <NoOpMemoryModel.hpp>
      #include <MatrixInitOp.hpp>
        #include <simple_mesh_description.hpp>
          #include <utils.hpp> (DNE)
          #include <set> (DNE)
          #include <map> (DNE)
        #include <box_utils.hpp>
          #include <vector> (DNE)
          #include <map> (DNE)
          #include <mpi.h> (DNE)
          #include <TypeTraits.hpp>
            #include <complex> (DNE)
            #include <mpi.h> (DNE)
          #include <Box.hpp>
        #include <ComputeNodeType.hpp>
          #include <tbb/task_scheduler_init.h> (DNE)
          #include <TBBNode.hpp>
            #include <tbb/blocked_range.h> (DNE)
            #include <tbb/parallel_for.h> (DNE)
            #include <tbb/parallel_reduce.h> (DNE)
            #include <tbb/task_scheduler_init.h> (DNE)
            #include <stdlib.h> (DNE)
            #include <NoOpMemoryModel.hpp>
            #include <iostream> (DNE)
          #include <TPI.h> (DNE)
          #include <TPINode.hpp>
            #include <TPI.h> (DNE)
            #include <NoOpMemoryModel.hpp>
            #include <iostream> (DNE)
          #include <CudaNode.hpp> (DNE)
          #include <SerialComputeNode.hpp>
            #include <NoOpMemoryModel.hpp>
        #include <CSRMatrix.hpp>
          #include <cstddef> (DNE)
          #include <vector> (DNE)
          #include <algorithm> (DNE)
          #include <mpi.h> (DNE)
        #include <ELLMatrix.hpp>
          #include <cstddef> (DNE)
          #include <vector> (DNE)
          #include <algorithm> (DNE)
          #include <mpi.h> (DNE)
        #include <algorithm> (DNE)
      #include <MatrixCopyOp.hpp>
      #include <exchange_externals.hpp>
        #include <cstdlib> (DNE)
        #include <iostream> (DNE)
        #include <mpi.h> (DNE)
        #include <outstream.hpp> (DNE)
        #include <TypeTraits.hpp>
          #include <complex> (DNE)
          #include <mpi.h> (DNE)
      #include <mytimer.hpp> (DNE)
      #include <LockingMatrix.hpp>
        #include <vector> (DNE)
        #include <Lock.hpp>
          #include <iostream> (DNE)
          #include <tbb/atomic.h> (DNE)
      #include <mpi.h> (DNE)
    #include <generate_matrix_structure.hpp>
      #include <sstream> (DNE)
      #include <stdexcept> (DNE)
      #include <map> (DNE)
      #include <algorithm> (DNE)
      #include <simple_mesh_description.hpp>
        #include <utils.hpp> (DNE)
        #include <set> (DNE)
        #include <map> (DNE)
      #include <SparseMatrix_functions.hpp>
        #include <cstddef> (DNE)
        #include <vector> (DNE)
        #include <set> (DNE)
        #include <algorithm> (DNE)
        #include <sstream> (DNE)
        #include <fstream> (DNE)
        #include <Vector.hpp>
          #include <vector> (DNE)
          #include <MemInitOp.hpp>
        #include <Vector_functions.hpp>
          #include <vector> (DNE)
          #include <sstream> (DNE)
          #include <fstream> (DNE)
          #include <mpi.h> (DNE)
          #include <LockingVector.hpp>
            #include <vector> (DNE)
            #include <Lock.hpp>
              #include <iostream> (DNE)
              #include <tbb/atomic.h> (DNE)
          #include <TypeTraits.hpp>
            #include <complex> (DNE)
            #include <mpi.h> (DNE)
          #include <Vector.hpp>
            #include <vector> (DNE)
            #include <MemInitOp.hpp>
          #include <WaxpbyOp.hpp>
          #include <DotOp.hpp>
        #include <ElemData.hpp> (DNE)
        #include <FusedMatvecDotOp.hpp>
        #include <MatvecOp.hpp>
          #include <CSRMatrix.hpp>
            #include <cstddef> (DNE)
            #include <vector> (DNE)
            #include <algorithm> (DNE)
            #include <mpi.h> (DNE)
          #include <ELLMatrix.hpp>
            #include <cstddef> (DNE)
            #include <vector> (DNE)
            #include <algorithm> (DNE)
            #include <mpi.h> (DNE)
          #include <ComputeNodeType.hpp>
            #include <tbb/task_scheduler_init.h> (DNE)
            #include <TBBNode.hpp>
              #include <tbb/blocked_range.h> (DNE)
              #include <tbb/parallel_for.h> (DNE)
              #include <tbb/parallel_reduce.h> (DNE)
              #include <tbb/task_scheduler_init.h> (DNE)
              #include <stdlib.h> (DNE)
              #include <NoOpMemoryModel.hpp>
              #include <iostream> (DNE)
            #include <TPI.h> (DNE)
            #include <TPINode.hpp>
              #include <TPI.h> (DNE)
              #include <NoOpMemoryModel.hpp>
              #include <iostream> (DNE)
            #include <CudaNode.hpp> (DNE)
            #include <SerialComputeNode.hpp>
              #include <NoOpMemoryModel.hpp>
        #include <MatrixInitOp.hpp>
          #include <simple_mesh_description.hpp>
            #include <utils.hpp> (DNE)
            #include <set> (DNE)
            #include <map> (DNE)
          #include <box_utils.hpp>
            #include <vector> (DNE)
            #include <map> (DNE)
            #include <mpi.h> (DNE)
            #include <TypeTraits.hpp>
              #include <complex> (DNE)
              #include <mpi.h> (DNE)
            #include <Box.hpp>
          #include <ComputeNodeType.hpp>
            #include <tbb/task_scheduler_init.h> (DNE)
            #include <TBBNode.hpp>
              #include <tbb/blocked_range.h> (DNE)
              #include <tbb/parallel_for.h> (DNE)
              #include <tbb/parallel_reduce.h> (DNE)
              #include <tbb/task_scheduler_init.h> (DNE)
              #include <stdlib.h> (DNE)
              #include <NoOpMemoryModel.hpp>
              #include <iostream> (DNE)
            #include <TPI.h> (DNE)
            #include <TPINode.hpp>
              #include <TPI.h> (DNE)
              #include <NoOpMemoryModel.hpp>
              #include <iostream> (DNE)
            #include <CudaNode.hpp> (DNE)
            #include <SerialComputeNode.hpp>
              #include <NoOpMemoryModel.hpp>
          #include <CSRMatrix.hpp>
            #include <cstddef> (DNE)
            #include <vector> (DNE)
            #include <algorithm> (DNE)
            #include <mpi.h> (DNE)
          #include <ELLMatrix.hpp>
            #include <cstddef> (DNE)
            #include <vector> (DNE)
            #include <algorithm> (DNE)
            #include <mpi.h> (DNE)
          #include <algorithm> (DNE)
        #include <MatrixCopyOp.hpp>
        #include <exchange_externals.hpp>
          #include <cstdlib> (DNE)
          #include <iostream> (DNE)
          #include <mpi.h> (DNE)
          #include <outstream.hpp> (DNE)
          #include <TypeTraits.hpp>
            #include <complex> (DNE)
            #include <mpi.h> (DNE)
        #include <mytimer.hpp> (DNE)
        #include <LockingMatrix.hpp>
          #include <vector> (DNE)
          #include <Lock.hpp>
            #include <iostream> (DNE)
            #include <tbb/atomic.h> (DNE)
        #include <mpi.h> (DNE)
      #include <box_utils.hpp>
        #include <vector> (DNE)
        #include <map> (DNE)
        #include <mpi.h> (DNE)
        #include <TypeTraits.hpp>
          #include <complex> (DNE)
          #include <mpi.h> (DNE)
        #include <Box.hpp>
      #include <utils.hpp> (DNE)
      #include <mpi.h> (DNE)
    #include <assemble_FE_data.hpp>
      #include <box_utils.hpp>
        #include <vector> (DNE)
        #include <map> (DNE)
        #include <mpi.h> (DNE)
        #include <TypeTraits.hpp>
          #include <complex> (DNE)
          #include <mpi.h> (DNE)
        #include <Box.hpp>
      #include <simple_mesh_description.hpp>
        #include <utils.hpp> (DNE)
        #include <set> (DNE)
        #include <map> (DNE)
      #include <perform_element_loop_TBB_pllfor1.hpp>
        #include <LockingMatrix.hpp>
          #include <vector> (DNE)
          #include <Lock.hpp>
            #include <iostream> (DNE)
            #include <tbb/atomic.h> (DNE)
        #include <LockingVector.hpp>
          #include <vector> (DNE)
          #include <Lock.hpp>
            #include <iostream> (DNE)
            #include <tbb/atomic.h> (DNE)
        #include <BoxIterator.hpp>
        #include <simple_mesh_description.hpp>
          #include <utils.hpp> (DNE)
          #include <set> (DNE)
          #include <map> (DNE)
        #include <SparseMatrix_functions.hpp>
          #include <cstddef> (DNE)
          #include <vector> (DNE)
          #include <set> (DNE)
          #include <algorithm> (DNE)
          #include <sstream> (DNE)
          #include <fstream> (DNE)
          #include <Vector.hpp>
            #include <vector> (DNE)
            #include <MemInitOp.hpp>
          #include <Vector_functions.hpp>
            #include <vector> (DNE)
            #include <sstream> (DNE)
            #include <fstream> (DNE)
            #include <mpi.h> (DNE)
            #include <LockingVector.hpp>
              #include <vector> (DNE)
              #include <Lock.hpp>
                #include <iostream> (DNE)
                #include <tbb/atomic.h> (DNE)
            #include <TypeTraits.hpp>
              #include <complex> (DNE)
              #include <mpi.h> (DNE)
            #include <Vector.hpp>
              #include <vector> (DNE)
              #include <MemInitOp.hpp>
            #include <WaxpbyOp.hpp>
            #include <DotOp.hpp>
          #include <ElemData.hpp> (DNE)
          #include <FusedMatvecDotOp.hpp>
          #include <MatvecOp.hpp>
            #include <CSRMatrix.hpp>
              #include <cstddef> (DNE)
              #include <vector> (DNE)
              #include <algorithm> (DNE)
              #include <mpi.h> (DNE)
            #include <ELLMatrix.hpp>
              #include <cstddef> (DNE)
              #include <vector> (DNE)
              #include <algorithm> (DNE)
              #include <mpi.h> (DNE)
            #include <ComputeNodeType.hpp>
              #include <tbb/task_scheduler_init.h> (DNE)
              #include <TBBNode.hpp>
                #include <tbb/blocked_range.h> (DNE)
                #include <tbb/parallel_for.h> (DNE)
                #include <tbb/parallel_reduce.h> (DNE)
                #include <tbb/task_scheduler_init.h> (DNE)
                #include <stdlib.h> (DNE)
                #include <NoOpMemoryModel.hpp>
                #include <iostream> (DNE)
              #include <TPI.h> (DNE)
              #include <TPINode.hpp>
                #include <TPI.h> (DNE)
                #include <NoOpMemoryModel.hpp>
                #include <iostream> (DNE)
              #include <CudaNode.hpp> (DNE)
              #include <SerialComputeNode.hpp>
                #include <NoOpMemoryModel.hpp>
          #include <MatrixInitOp.hpp>
            #include <simple_mesh_description.hpp>
              #include <utils.hpp> (DNE)
              #include <set> (DNE)
              #include <map> (DNE)
            #include <box_utils.hpp>
              #include <vector> (DNE)
              #include <map> (DNE)
              #include <mpi.h> (DNE)
              #include <TypeTraits.hpp>
                #include <complex> (DNE)
                #include <mpi.h> (DNE)
              #include <Box.hpp>
            #include <ComputeNodeType.hpp>
              #include <tbb/task_scheduler_init.h> (DNE)
              #include <TBBNode.hpp>
                #include <tbb/blocked_range.h> (DNE)
                #include <tbb/parallel_for.h> (DNE)
                #include <tbb/parallel_reduce.h> (DNE)
                #include <tbb/task_scheduler_init.h> (DNE)
                #include <stdlib.h> (DNE)
                #include <NoOpMemoryModel.hpp>
                #include <iostream> (DNE)
              #include <TPI.h> (DNE)
              #include <TPINode.hpp>
                #include <TPI.h> (DNE)
                #include <NoOpMemoryModel.hpp>
                #include <iostream> (DNE)
              #include <CudaNode.hpp> (DNE)
              #include <SerialComputeNode.hpp>
                #include <NoOpMemoryModel.hpp>
            #include <CSRMatrix.hpp>
              #include <cstddef> (DNE)
              #include <vector> (DNE)
              #include <algorithm> (DNE)
              #include <mpi.h> (DNE)
            #include <ELLMatrix.hpp>
              #include <cstddef> (DNE)
              #include <vector> (DNE)
              #include <algorithm> (DNE)
              #include <mpi.h> (DNE)
            #include <algorithm> (DNE)
          #include <MatrixCopyOp.hpp>
          #include <exchange_externals.hpp>
            #include <cstdlib> (DNE)
            #include <iostream> (DNE)
            #include <mpi.h> (DNE)
            #include <outstream.hpp> (DNE)
            #include <TypeTraits.hpp>
              #include <complex> (DNE)
              #include <mpi.h> (DNE)
          #include <mytimer.hpp> (DNE)
          #include <LockingMatrix.hpp>
            #include <vector> (DNE)
            #include <Lock.hpp>
              #include <iostream> (DNE)
              #include <tbb/atomic.h> (DNE)
          #include <mpi.h> (DNE)
        #include <Hex8_box_utils.hpp>
          #include <stdexcept> (DNE)
          #include <box_utils.hpp>
            #include <vector> (DNE)
            #include <map> (DNE)
            #include <mpi.h> (DNE)
            #include <TypeTraits.hpp>
              #include <complex> (DNE)
              #include <mpi.h> (DNE)
            #include <Box.hpp>
          #include <ElemData.hpp> (DNE)
          #include <simple_mesh_description.hpp>
            #include <utils.hpp> (DNE)
            #include <set> (DNE)
            #include <map> (DNE)
          #include <Hex8.hpp> (DNE)
        #include <Hex8_ElemData.hpp> (DNE)
        #include <mytimer.hpp> (DNE)
      #include <perform_element_loop.hpp>
        #include <BoxIterator.hpp>
        #include <simple_mesh_description.hpp>
          #include <utils.hpp> (DNE)
          #include <set> (DNE)
          #include <map> (DNE)
        #include <SparseMatrix_functions.hpp>
          #include <cstddef> (DNE)
          #include <vector> (DNE)
          #include <set> (DNE)
          #include <algorithm> (DNE)
          #include <sstream> (DNE)
          #include <fstream> (DNE)
          #include <Vector.hpp>
            #include <vector> (DNE)
            #include <MemInitOp.hpp>
          #include <Vector_functions.hpp>
            #include <vector> (DNE)
            #include <sstream> (DNE)
            #include <fstream> (DNE)
            #include <mpi.h> (DNE)
            #include <LockingVector.hpp>
              #include <vector> (DNE)
              #include <Lock.hpp>
                #include <iostream> (DNE)
                #include <tbb/atomic.h> (DNE)
            #include <TypeTraits.hpp>
              #include <complex> (DNE)
              #include <mpi.h> (DNE)
            #include <Vector.hpp>
              #include <vector> (DNE)
              #include <MemInitOp.hpp>
            #include <WaxpbyOp.hpp>
            #include <DotOp.hpp>
          #include <ElemData.hpp> (DNE)
          #include <FusedMatvecDotOp.hpp>
          #include <MatvecOp.hpp>
            #include <CSRMatrix.hpp>
              #include <cstddef> (DNE)
              #include <vector> (DNE)
              #include <algorithm> (DNE)
              #include <mpi.h> (DNE)
            #include <ELLMatrix.hpp>
              #include <cstddef> (DNE)
              #include <vector> (DNE)
              #include <algorithm> (DNE)
              #include <mpi.h> (DNE)
            #include <ComputeNodeType.hpp>
              #include <tbb/task_scheduler_init.h> (DNE)
              #include <TBBNode.hpp>
                #include <tbb/blocked_range.h> (DNE)
                #include <tbb/parallel_for.h> (DNE)
                #include <tbb/parallel_reduce.h> (DNE)
                #include <tbb/task_scheduler_init.h> (DNE)
                #include <stdlib.h> (DNE)
                #include <NoOpMemoryModel.hpp>
                #include <iostream> (DNE)
              #include <TPI.h> (DNE)
              #include <TPINode.hpp>
                #include <TPI.h> (DNE)
                #include <NoOpMemoryModel.hpp>
                #include <iostream> (DNE)
              #include <CudaNode.hpp> (DNE)
              #include <SerialComputeNode.hpp>
                #include <NoOpMemoryModel.hpp>
          #include <MatrixInitOp.hpp>
            #include <simple_mesh_description.hpp>
              #include <utils.hpp> (DNE)
              #include <set> (DNE)
              #include <map> (DNE)
            #include <box_utils.hpp>
              #include <vector> (DNE)
              #include <map> (DNE)
              #include <mpi.h> (DNE)
              #include <TypeTraits.hpp>
                #include <complex> (DNE)
                #include <mpi.h> (DNE)
              #include <Box.hpp>
            #include <ComputeNodeType.hpp>
              #include <tbb/task_scheduler_init.h> (DNE)
              #include <TBBNode.hpp>
                #include <tbb/blocked_range.h> (DNE)
                #include <tbb/parallel_for.h> (DNE)
                #include <tbb/parallel_reduce.h> (DNE)
                #include <tbb/task_scheduler_init.h> (DNE)
                #include <stdlib.h> (DNE)
                #include <NoOpMemoryModel.hpp>
                #include <iostream> (DNE)
              #include <TPI.h> (DNE)
              #include <TPINode.hpp>
                #include <TPI.h> (DNE)
                #include <NoOpMemoryModel.hpp>
                #include <iostream> (DNE)
              #include <CudaNode.hpp> (DNE)
              #include <SerialComputeNode.hpp>
                #include <NoOpMemoryModel.hpp>
            #include <CSRMatrix.hpp>
              #include <cstddef> (DNE)
              #include <vector> (DNE)
              #include <algorithm> (DNE)
              #include <mpi.h> (DNE)
            #include <ELLMatrix.hpp>
              #include <cstddef> (DNE)
              #include <vector> (DNE)
              #include <algorithm> (DNE)
              #include <mpi.h> (DNE)
            #include <algorithm> (DNE)
          #include <MatrixCopyOp.hpp>
          #include <exchange_externals.hpp>
            #include <cstdlib> (DNE)
            #include <iostream> (DNE)
            #include <mpi.h> (DNE)
            #include <outstream.hpp> (DNE)
            #include <TypeTraits.hpp>
              #include <complex> (DNE)
              #include <mpi.h> (DNE)
          #include <mytimer.hpp> (DNE)
          #include <LockingMatrix.hpp>
            #include <vector> (DNE)
            #include <Lock.hpp>
              #include <iostream> (DNE)
              #include <tbb/atomic.h> (DNE)
          #include <mpi.h> (DNE)
        #include <box_utils.hpp>
          #include <vector> (DNE)
          #include <map> (DNE)
          #include <mpi.h> (DNE)
          #include <TypeTraits.hpp>
            #include <complex> (DNE)
            #include <mpi.h> (DNE)
          #include <Box.hpp>
        #include <Hex8_box_utils.hpp>
          #include <stdexcept> (DNE)
          #include <box_utils.hpp>
            #include <vector> (DNE)
            #include <map> (DNE)
            #include <mpi.h> (DNE)
            #include <TypeTraits.hpp>
              #include <complex> (DNE)
              #include <mpi.h> (DNE)
            #include <Box.hpp>
          #include <ElemData.hpp> (DNE)
          #include <simple_mesh_description.hpp>
            #include <utils.hpp> (DNE)
            #include <set> (DNE)
            #include <map> (DNE)
          #include <Hex8.hpp> (DNE)
        #include <Hex8_ElemData.hpp> (DNE)
    #include <verify_solution.hpp>
      #include <sstream> (DNE)
      #include <stdexcept> (DNE)
      #include <map> (DNE)
      #include <algorithm> (DNE)
      #include <simple_mesh_description.hpp>
        #include <utils.hpp> (DNE)
        #include <set> (DNE)
        #include <map> (DNE)
      #include <analytic_soln.hpp>
        #include <cmath> (DNE)
      #include <box_utils.hpp>
        #include <vector> (DNE)
        #include <map> (DNE)
        #include <mpi.h> (DNE)
        #include <TypeTraits.hpp>
          #include <complex> (DNE)
          #include <mpi.h> (DNE)
        #include <Box.hpp>
      #include <utils.hpp> (DNE)
      #include <mpi.h> (DNE)
    #include <compute_matrix_stats.hpp>
      #include <cstddef> (DNE)
      #include <cmath> (DNE)
      #include <cstdlib> (DNE)
      #include <iostream> (DNE)
      #include <sstream> (DNE)
      #include <iomanip> (DNE)
      #include <outstream.hpp> (DNE)
      #include <utils.hpp> (DNE)
      #include <YAML_Doc.hpp> (DNE)
    #include <make_local_matrix.hpp>
      #include <map> (DNE)
      #include <mpi.h> (DNE)
    #include <imbalance.hpp>
      #include <cmath> (DNE)
      #include <mpi.h> (DNE)
      #include <box_utils.hpp>
        #include <vector> (DNE)
        #include <map> (DNE)
        #include <mpi.h> (DNE)
        #include <TypeTraits.hpp>
          #include <complex> (DNE)
          #include <mpi.h> (DNE)
        #include <Box.hpp>
      #include <utils.hpp> (DNE)
      #include <YAML_Doc.hpp> (DNE)
    #include <cg_solve.hpp>
      #include <cmath> (DNE)
      #include <limits> (DNE)
      #include <Vector_functions.hpp>
        #include <vector> (DNE)
        #include <sstream> (DNE)
        #include <fstream> (DNE)
        #include <mpi.h> (DNE)
        #include <LockingVector.hpp>
          #include <vector> (DNE)
          #include <Lock.hpp>
            #include <iostream> (DNE)
            #include <tbb/atomic.h> (DNE)
        #include <TypeTraits.hpp>
          #include <complex> (DNE)
          #include <mpi.h> (DNE)
        #include <Vector.hpp>
          #include <vector> (DNE)
          #include <MemInitOp.hpp>
        #include <WaxpbyOp.hpp>
        #include <DotOp.hpp>
      #include <mytimer.hpp> (DNE)
      #include <outstream.hpp> (DNE)
    #include <time_kernels.hpp>
      #include <cmath> (DNE)
      #include <Vector_functions.hpp>
        #include <vector> (DNE)
        #include <sstream> (DNE)
        #include <fstream> (DNE)
        #include <mpi.h> (DNE)
        #include <LockingVector.hpp>
          #include <vector> (DNE)
          #include <Lock.hpp>
            #include <iostream> (DNE)
            #include <tbb/atomic.h> (DNE)
        #include <TypeTraits.hpp>
          #include <complex> (DNE)
          #include <mpi.h> (DNE)
        #include <Vector.hpp>
          #include <vector> (DNE)
          #include <MemInitOp.hpp>
        #include <WaxpbyOp.hpp>
        #include <DotOp.hpp>
      #include <mytimer.hpp> (DNE)
      #include <cuda.h> (DNE)
    #include <outstream.hpp> (DNE)
    #include <utils.hpp> (DNE)
    #include <mytimer.hpp> (DNE)
    #include <YAML_Doc.hpp> (DNE)
    #include <mpi.h> (DNE)
  #include <YAML_Doc.hpp> (DNE)
  #include <miniFE_info.hpp> (DNE)
  #include <miniFE_no_info.hpp> (DNE)

""",
    "basic/optional/ThreadPool/test/hhpccg/BoxPartitionIB.c": """basic/optional/ThreadPool/test/hhpccg/BoxPartitionIB.c
  #include <stdio.h> (DNE)
  #include <stdlib.h> (DNE)
  #include <BoxPartitionIB.h>

""",
    "basic/optional/ThreadPool/test/hhpccg/main.c": """basic/optional/ThreadPool/test/hhpccg/main.c
  #include <math.h> (DNE)
  #include <stdio.h> (DNE)
  #include <stdlib.h> (DNE)
  #include <string.h> (DNE)
  #include <ThreadPool_config.h> (DNE)
  #include <TPI.h> (DNE)
  #include <BoxPartitionIB.h>
  #include <dcrs_matrix.h>
    #include <tpi_vector.h>
  #include <CGSolver.h>
    #include <tpi_vector.h>
    #include <dcrs_matrix.h>
      #include <tpi_vector.h>
  #include <mpi.h> (DNE)

""",
    "basic/optional/ThreadPool/test/hpccg/BoxPartition.c": """basic/optional/ThreadPool/test/hpccg/BoxPartition.c
  #include <stdio.h> (DNE)
  #include <stdlib.h> (DNE)
  #include <BoxPartition.h>

""",
    "basic/optional/ThreadPool/test/hpccg/main.c": """basic/optional/ThreadPool/test/hpccg/main.c
  #include <math.h> (DNE)
  #include <stdio.h> (DNE)
  #include <stdlib.h> (DNE)
  #include <string.h> (DNE)
  #include <ThreadPool_config.h> (DNE)
  #include <TPI.h> (DNE)
  #include <BoxPartition.h>
  #include <CGSolver.h>
    #include <tpi_vector.h>
      #include <ThreadPool_config.h> (DNE)
  #include <mpi.h> (DNE)

""",
    "basic/optional/ThreadPool/test/test_c_dnax.c": """basic/optional/ThreadPool/test/test_c_dnax.c
  #include <math.h> (DNE)
  #include <stddef.h> (DNE)
  #include <stdlib.h> (DNE)
  #include <stdio.h> (DNE)
  #include <TPI.h> (DNE)
  #include <mpi.h> (DNE)

""",
    "basic/optional/ThreadPool/test/test_mpi_sum.c": """basic/optional/ThreadPool/test/test_mpi_sum.c
  #include <math.h> (DNE)
  #include <stdlib.h> (DNE)
  #include <stdio.h> (DNE)
  #include <TPI.h> (DNE)
  #include <ThreadPool_config.h> (DNE)
  #include <mpi.h> (DNE)

""",
    "basic/optional/ThreadPool/test/test_tpi.cpp": """basic/optional/ThreadPool/test/test_tpi.cpp
  #include <stdexcept> (DNE)
  #include <iostream> (DNE)
  #include <TPI.hpp> (DNE)

""",
    "basic/optional/ThreadPool/test/test_tpi_unit.c": """basic/optional/ThreadPool/test/test_tpi_unit.c
  #include <stdlib.h> (DNE)
  #include <stdio.h> (DNE)
  #include <math.h> (DNE)
  #include <TPI.h> (DNE)
  #include <mpi.h> (DNE)

""",
    "basic/utest.cpp": """basic/utest.cpp
  #include <iostream> (DNE)
  #include <mpi.h> (DNE)
  #include <utest_case.hpp>
    #include <vector> (DNE)
  #include <utest_cases.hpp>
    #include <iostream> (DNE)
    #include <cmath> (DNE)
    #include <BoxPartition.hpp>
      #include <Box.hpp>
    #include <box_utils.hpp>
      #include <vector> (DNE)
      #include <map> (DNE)
      #include <mpi.h> (DNE)
      #include <TypeTraits.hpp>
        #include <complex> (DNE)
        #include <mpi.h> (DNE)
      #include <Box.hpp>
    #include <simple_mesh_description.hpp>
      #include <utils.hpp> (DNE)
      #include <set> (DNE)
      #include <map> (DNE)
    #include <generate_matrix_structure.hpp>
      #include <sstream> (DNE)
      #include <stdexcept> (DNE)
      #include <map> (DNE)
      #include <algorithm> (DNE)
      #include <simple_mesh_description.hpp>
        #include <utils.hpp> (DNE)
        #include <set> (DNE)
        #include <map> (DNE)
      #include <SparseMatrix_functions.hpp>
        #include <cstddef> (DNE)
        #include <vector> (DNE)
        #include <set> (DNE)
        #include <algorithm> (DNE)
        #include <sstream> (DNE)
        #include <fstream> (DNE)
        #include <Vector.hpp>
          #include <vector> (DNE)
          #include <MemInitOp.hpp>
        #include <Vector_functions.hpp>
          #include <vector> (DNE)
          #include <sstream> (DNE)
          #include <fstream> (DNE)
          #include <mpi.h> (DNE)
          #include <LockingVector.hpp>
            #include <vector> (DNE)
            #include <Lock.hpp>
              #include <iostream> (DNE)
              #include <tbb/atomic.h> (DNE)
          #include <TypeTraits.hpp>
            #include <complex> (DNE)
            #include <mpi.h> (DNE)
          #include <Vector.hpp>
            #include <vector> (DNE)
            #include <MemInitOp.hpp>
          #include <WaxpbyOp.hpp>
          #include <DotOp.hpp>
        #include <ElemData.hpp> (DNE)
        #include <FusedMatvecDotOp.hpp>
        #include <MatvecOp.hpp>
          #include <CSRMatrix.hpp>
            #include <cstddef> (DNE)
            #include <vector> (DNE)
            #include <algorithm> (DNE)
            #include <mpi.h> (DNE)
          #include <ELLMatrix.hpp>
            #include <cstddef> (DNE)
            #include <vector> (DNE)
            #include <algorithm> (DNE)
            #include <mpi.h> (DNE)
          #include <ComputeNodeType.hpp>
            #include <tbb/task_scheduler_init.h> (DNE)
            #include <TBBNode.hpp>
              #include <tbb/blocked_range.h> (DNE)
              #include <tbb/parallel_for.h> (DNE)
              #include <tbb/parallel_reduce.h> (DNE)
              #include <tbb/task_scheduler_init.h> (DNE)
              #include <stdlib.h> (DNE)
              #include <NoOpMemoryModel.hpp>
              #include <iostream> (DNE)
            #include <TPI.h> (DNE)
            #include <TPINode.hpp>
              #include <TPI.h> (DNE)
              #include <NoOpMemoryModel.hpp>
              #include <iostream> (DNE)
            #include <CudaNode.hpp> (DNE)
            #include <SerialComputeNode.hpp>
              #include <NoOpMemoryModel.hpp>
        #include <MatrixInitOp.hpp>
          #include <simple_mesh_description.hpp>
            #include <utils.hpp> (DNE)
            #include <set> (DNE)
            #include <map> (DNE)
          #include <box_utils.hpp>
            #include <vector> (DNE)
            #include <map> (DNE)
            #include <mpi.h> (DNE)
            #include <TypeTraits.hpp>
              #include <complex> (DNE)
              #include <mpi.h> (DNE)
            #include <Box.hpp>
          #include <ComputeNodeType.hpp>
            #include <tbb/task_scheduler_init.h> (DNE)
            #include <TBBNode.hpp>
              #include <tbb/blocked_range.h> (DNE)
              #include <tbb/parallel_for.h> (DNE)
              #include <tbb/parallel_reduce.h> (DNE)
              #include <tbb/task_scheduler_init.h> (DNE)
              #include <stdlib.h> (DNE)
              #include <NoOpMemoryModel.hpp>
              #include <iostream> (DNE)
            #include <TPI.h> (DNE)
            #include <TPINode.hpp>
              #include <TPI.h> (DNE)
              #include <NoOpMemoryModel.hpp>
              #include <iostream> (DNE)
            #include <CudaNode.hpp> (DNE)
            #include <SerialComputeNode.hpp>
              #include <NoOpMemoryModel.hpp>
          #include <CSRMatrix.hpp>
            #include <cstddef> (DNE)
            #include <vector> (DNE)
            #include <algorithm> (DNE)
            #include <mpi.h> (DNE)
          #include <ELLMatrix.hpp>
            #include <cstddef> (DNE)
            #include <vector> (DNE)
            #include <algorithm> (DNE)
            #include <mpi.h> (DNE)
          #include <algorithm> (DNE)
        #include <MatrixCopyOp.hpp>
        #include <exchange_externals.hpp>
          #include <cstdlib> (DNE)
          #include <iostream> (DNE)
          #include <mpi.h> (DNE)
          #include <outstream.hpp> (DNE)
          #include <TypeTraits.hpp>
            #include <complex> (DNE)
            #include <mpi.h> (DNE)
        #include <mytimer.hpp> (DNE)
        #include <LockingMatrix.hpp>
          #include <vector> (DNE)
          #include <Lock.hpp>
            #include <iostream> (DNE)
            #include <tbb/atomic.h> (DNE)
        #include <mpi.h> (DNE)
      #include <box_utils.hpp>
        #include <vector> (DNE)
        #include <map> (DNE)
        #include <mpi.h> (DNE)
        #include <TypeTraits.hpp>
          #include <complex> (DNE)
          #include <mpi.h> (DNE)
        #include <Box.hpp>
      #include <utils.hpp> (DNE)
      #include <mpi.h> (DNE)
    #include <Hex8.hpp> (DNE)
    #include <Hex8_box_utils.hpp>
      #include <stdexcept> (DNE)
      #include <box_utils.hpp>
        #include <vector> (DNE)
        #include <map> (DNE)
        #include <mpi.h> (DNE)
        #include <TypeTraits.hpp>
          #include <complex> (DNE)
          #include <mpi.h> (DNE)
        #include <Box.hpp>
      #include <ElemData.hpp> (DNE)
      #include <simple_mesh_description.hpp>
        #include <utils.hpp> (DNE)
        #include <set> (DNE)
        #include <map> (DNE)
      #include <Hex8.hpp> (DNE)
    #include <assemble_FE_data.hpp>
      #include <box_utils.hpp>
        #include <vector> (DNE)
        #include <map> (DNE)
        #include <mpi.h> (DNE)
        #include <TypeTraits.hpp>
          #include <complex> (DNE)
          #include <mpi.h> (DNE)
        #include <Box.hpp>
      #include <simple_mesh_description.hpp>
        #include <utils.hpp> (DNE)
        #include <set> (DNE)
        #include <map> (DNE)
      #include <perform_element_loop_TBB_pllfor1.hpp>
        #include <LockingMatrix.hpp>
          #include <vector> (DNE)
          #include <Lock.hpp>
            #include <iostream> (DNE)
            #include <tbb/atomic.h> (DNE)
        #include <LockingVector.hpp>
          #include <vector> (DNE)
          #include <Lock.hpp>
            #include <iostream> (DNE)
            #include <tbb/atomic.h> (DNE)
        #include <BoxIterator.hpp>
        #include <simple_mesh_description.hpp>
          #include <utils.hpp> (DNE)
          #include <set> (DNE)
          #include <map> (DNE)
        #include <SparseMatrix_functions.hpp>
          #include <cstddef> (DNE)
          #include <vector> (DNE)
          #include <set> (DNE)
          #include <algorithm> (DNE)
          #include <sstream> (DNE)
          #include <fstream> (DNE)
          #include <Vector.hpp>
            #include <vector> (DNE)
            #include <MemInitOp.hpp>
          #include <Vector_functions.hpp>
            #include <vector> (DNE)
            #include <sstream> (DNE)
            #include <fstream> (DNE)
            #include <mpi.h> (DNE)
            #include <LockingVector.hpp>
              #include <vector> (DNE)
              #include <Lock.hpp>
                #include <iostream> (DNE)
                #include <tbb/atomic.h> (DNE)
            #include <TypeTraits.hpp>
              #include <complex> (DNE)
              #include <mpi.h> (DNE)
            #include <Vector.hpp>
              #include <vector> (DNE)
              #include <MemInitOp.hpp>
            #include <WaxpbyOp.hpp>
            #include <DotOp.hpp>
          #include <ElemData.hpp> (DNE)
          #include <FusedMatvecDotOp.hpp>
          #include <MatvecOp.hpp>
            #include <CSRMatrix.hpp>
              #include <cstddef> (DNE)
              #include <vector> (DNE)
              #include <algorithm> (DNE)
              #include <mpi.h> (DNE)
            #include <ELLMatrix.hpp>
              #include <cstddef> (DNE)
              #include <vector> (DNE)
              #include <algorithm> (DNE)
              #include <mpi.h> (DNE)
            #include <ComputeNodeType.hpp>
              #include <tbb/task_scheduler_init.h> (DNE)
              #include <TBBNode.hpp>
                #include <tbb/blocked_range.h> (DNE)
                #include <tbb/parallel_for.h> (DNE)
                #include <tbb/parallel_reduce.h> (DNE)
                #include <tbb/task_scheduler_init.h> (DNE)
                #include <stdlib.h> (DNE)
                #include <NoOpMemoryModel.hpp>
                #include <iostream> (DNE)
              #include <TPI.h> (DNE)
              #include <TPINode.hpp>
                #include <TPI.h> (DNE)
                #include <NoOpMemoryModel.hpp>
                #include <iostream> (DNE)
              #include <CudaNode.hpp> (DNE)
              #include <SerialComputeNode.hpp>
                #include <NoOpMemoryModel.hpp>
          #include <MatrixInitOp.hpp>
            #include <simple_mesh_description.hpp>
              #include <utils.hpp> (DNE)
              #include <set> (DNE)
              #include <map> (DNE)
            #include <box_utils.hpp>
              #include <vector> (DNE)
              #include <map> (DNE)
              #include <mpi.h> (DNE)
              #include <TypeTraits.hpp>
                #include <complex> (DNE)
                #include <mpi.h> (DNE)
              #include <Box.hpp>
            #include <ComputeNodeType.hpp>
              #include <tbb/task_scheduler_init.h> (DNE)
              #include <TBBNode.hpp>
                #include <tbb/blocked_range.h> (DNE)
                #include <tbb/parallel_for.h> (DNE)
                #include <tbb/parallel_reduce.h> (DNE)
                #include <tbb/task_scheduler_init.h> (DNE)
                #include <stdlib.h> (DNE)
                #include <NoOpMemoryModel.hpp>
                #include <iostream> (DNE)
              #include <TPI.h> (DNE)
              #include <TPINode.hpp>
                #include <TPI.h> (DNE)
                #include <NoOpMemoryModel.hpp>
                #include <iostream> (DNE)
              #include <CudaNode.hpp> (DNE)
              #include <SerialComputeNode.hpp>
                #include <NoOpMemoryModel.hpp>
            #include <CSRMatrix.hpp>
              #include <cstddef> (DNE)
              #include <vector> (DNE)
              #include <algorithm> (DNE)
              #include <mpi.h> (DNE)
            #include <ELLMatrix.hpp>
              #include <cstddef> (DNE)
              #include <vector> (DNE)
              #include <algorithm> (DNE)
              #include <mpi.h> (DNE)
            #include <algorithm> (DNE)
          #include <MatrixCopyOp.hpp>
          #include <exchange_externals.hpp>
            #include <cstdlib> (DNE)
            #include <iostream> (DNE)
            #include <mpi.h> (DNE)
            #include <outstream.hpp> (DNE)
            #include <TypeTraits.hpp>
              #include <complex> (DNE)
              #include <mpi.h> (DNE)
          #include <mytimer.hpp> (DNE)
          #include <LockingMatrix.hpp>
            #include <vector> (DNE)
            #include <Lock.hpp>
              #include <iostream> (DNE)
              #include <tbb/atomic.h> (DNE)
          #include <mpi.h> (DNE)
        #include <Hex8_box_utils.hpp>
          #include <stdexcept> (DNE)
          #include <box_utils.hpp>
            #include <vector> (DNE)
            #include <map> (DNE)
            #include <mpi.h> (DNE)
            #include <TypeTraits.hpp>
              #include <complex> (DNE)
              #include <mpi.h> (DNE)
            #include <Box.hpp>
          #include <ElemData.hpp> (DNE)
          #include <simple_mesh_description.hpp>
            #include <utils.hpp> (DNE)
            #include <set> (DNE)
            #include <map> (DNE)
          #include <Hex8.hpp> (DNE)
        #include <Hex8_ElemData.hpp> (DNE)
        #include <mytimer.hpp> (DNE)
      #include <perform_element_loop.hpp>
        #include <BoxIterator.hpp>
        #include <simple_mesh_description.hpp>
          #include <utils.hpp> (DNE)
          #include <set> (DNE)
          #include <map> (DNE)
        #include <SparseMatrix_functions.hpp>
          #include <cstddef> (DNE)
          #include <vector> (DNE)
          #include <set> (DNE)
          #include <algorithm> (DNE)
          #include <sstream> (DNE)
          #include <fstream> (DNE)
          #include <Vector.hpp>
            #include <vector> (DNE)
            #include <MemInitOp.hpp>
          #include <Vector_functions.hpp>
            #include <vector> (DNE)
            #include <sstream> (DNE)
            #include <fstream> (DNE)
            #include <mpi.h> (DNE)
            #include <LockingVector.hpp>
              #include <vector> (DNE)
              #include <Lock.hpp>
                #include <iostream> (DNE)
                #include <tbb/atomic.h> (DNE)
            #include <TypeTraits.hpp>
              #include <complex> (DNE)
              #include <mpi.h> (DNE)
            #include <Vector.hpp>
              #include <vector> (DNE)
              #include <MemInitOp.hpp>
            #include <WaxpbyOp.hpp>
            #include <DotOp.hpp>
          #include <ElemData.hpp> (DNE)
          #include <FusedMatvecDotOp.hpp>
          #include <MatvecOp.hpp>
            #include <CSRMatrix.hpp>
              #include <cstddef> (DNE)
              #include <vector> (DNE)
              #include <algorithm> (DNE)
              #include <mpi.h> (DNE)
            #include <ELLMatrix.hpp>
              #include <cstddef> (DNE)
              #include <vector> (DNE)
              #include <algorithm> (DNE)
              #include <mpi.h> (DNE)
            #include <ComputeNodeType.hpp>
              #include <tbb/task_scheduler_init.h> (DNE)
              #include <TBBNode.hpp>
                #include <tbb/blocked_range.h> (DNE)
                #include <tbb/parallel_for.h> (DNE)
                #include <tbb/parallel_reduce.h> (DNE)
                #include <tbb/task_scheduler_init.h> (DNE)
                #include <stdlib.h> (DNE)
                #include <NoOpMemoryModel.hpp>
                #include <iostream> (DNE)
              #include <TPI.h> (DNE)
              #include <TPINode.hpp>
                #include <TPI.h> (DNE)
                #include <NoOpMemoryModel.hpp>
                #include <iostream> (DNE)
              #include <CudaNode.hpp> (DNE)
              #include <SerialComputeNode.hpp>
                #include <NoOpMemoryModel.hpp>
          #include <MatrixInitOp.hpp>
            #include <simple_mesh_description.hpp>
              #include <utils.hpp> (DNE)
              #include <set> (DNE)
              #include <map> (DNE)
            #include <box_utils.hpp>
              #include <vector> (DNE)
              #include <map> (DNE)
              #include <mpi.h> (DNE)
              #include <TypeTraits.hpp>
                #include <complex> (DNE)
                #include <mpi.h> (DNE)
              #include <Box.hpp>
            #include <ComputeNodeType.hpp>
              #include <tbb/task_scheduler_init.h> (DNE)
              #include <TBBNode.hpp>
                #include <tbb/blocked_range.h> (DNE)
                #include <tbb/parallel_for.h> (DNE)
                #include <tbb/parallel_reduce.h> (DNE)
                #include <tbb/task_scheduler_init.h> (DNE)
                #include <stdlib.h> (DNE)
                #include <NoOpMemoryModel.hpp>
                #include <iostream> (DNE)
              #include <TPI.h> (DNE)
              #include <TPINode.hpp>
                #include <TPI.h> (DNE)
                #include <NoOpMemoryModel.hpp>
                #include <iostream> (DNE)
              #include <CudaNode.hpp> (DNE)
              #include <SerialComputeNode.hpp>
                #include <NoOpMemoryModel.hpp>
            #include <CSRMatrix.hpp>
              #include <cstddef> (DNE)
              #include <vector> (DNE)
              #include <algorithm> (DNE)
              #include <mpi.h> (DNE)
            #include <ELLMatrix.hpp>
              #include <cstddef> (DNE)
              #include <vector> (DNE)
              #include <algorithm> (DNE)
              #include <mpi.h> (DNE)
            #include <algorithm> (DNE)
          #include <MatrixCopyOp.hpp>
          #include <exchange_externals.hpp>
            #include <cstdlib> (DNE)
            #include <iostream> (DNE)
            #include <mpi.h> (DNE)
            #include <outstream.hpp> (DNE)
            #include <TypeTraits.hpp>
              #include <complex> (DNE)
              #include <mpi.h> (DNE)
          #include <mytimer.hpp> (DNE)
          #include <LockingMatrix.hpp>
            #include <vector> (DNE)
            #include <Lock.hpp>
              #include <iostream> (DNE)
              #include <tbb/atomic.h> (DNE)
          #include <mpi.h> (DNE)
        #include <box_utils.hpp>
          #include <vector> (DNE)
          #include <map> (DNE)
          #include <mpi.h> (DNE)
          #include <TypeTraits.hpp>
            #include <complex> (DNE)
            #include <mpi.h> (DNE)
          #include <Box.hpp>
        #include <Hex8_box_utils.hpp>
          #include <stdexcept> (DNE)
          #include <box_utils.hpp>
            #include <vector> (DNE)
            #include <map> (DNE)
            #include <mpi.h> (DNE)
            #include <TypeTraits.hpp>
              #include <complex> (DNE)
              #include <mpi.h> (DNE)
            #include <Box.hpp>
          #include <ElemData.hpp> (DNE)
          #include <simple_mesh_description.hpp>
            #include <utils.hpp> (DNE)
            #include <set> (DNE)
            #include <map> (DNE)
          #include <Hex8.hpp> (DNE)
        #include <Hex8_ElemData.hpp> (DNE)
    #include <Parameters.hpp> (DNE)
    #include <make_local_matrix.hpp>
      #include <map> (DNE)
      #include <mpi.h> (DNE)
    #include <exchange_externals.hpp>
      #include <cstdlib> (DNE)
      #include <iostream> (DNE)
      #include <mpi.h> (DNE)
      #include <outstream.hpp> (DNE)
      #include <TypeTraits.hpp>
        #include <complex> (DNE)
        #include <mpi.h> (DNE)
    #include <Vector_functions.hpp>
      #include <vector> (DNE)
      #include <sstream> (DNE)
      #include <fstream> (DNE)
      #include <mpi.h> (DNE)
      #include <LockingVector.hpp>
        #include <vector> (DNE)
        #include <Lock.hpp>
          #include <iostream> (DNE)
          #include <tbb/atomic.h> (DNE)
      #include <TypeTraits.hpp>
        #include <complex> (DNE)
        #include <mpi.h> (DNE)
      #include <Vector.hpp>
        #include <vector> (DNE)
        #include <MemInitOp.hpp>
      #include <WaxpbyOp.hpp>
      #include <DotOp.hpp>
    #include <BoxIterator.hpp>
    #include <mytimer.hpp> (DNE)
    #include <SerialComputeNode.hpp>
      #include <NoOpMemoryModel.hpp>
    #include <TPI.h> (DNE)
    #include <TPINode.hpp>
      #include <TPI.h> (DNE)
      #include <NoOpMemoryModel.hpp>
      #include <iostream> (DNE)
    #include <tbb/task_scheduler_init.h> (DNE)
    #include <TBBNode.hpp>
      #include <tbb/blocked_range.h> (DNE)
      #include <tbb/parallel_for.h> (DNE)
      #include <tbb/parallel_reduce.h> (DNE)
      #include <tbb/task_scheduler_init.h> (DNE)
      #include <stdlib.h> (DNE)
      #include <NoOpMemoryModel.hpp>
      #include <iostream> (DNE)
    #include <CudaNode.hpp> (DNE)
    #include <utest_case.hpp>
      #include <vector> (DNE)

""",
    "src/main.cpp": """src/main.cpp
  #include <iostream> (DNE)
  #include <ctime> (DNE)
  #include <cstdlib> (DNE)
  #include <vector> (DNE)
  #include <miniFE_version.h> (DNE)
  #include <outstream.hpp> (DNE)
  #include <omp.h> (DNE)
  #include <sys/time.h> (DNE)
  #include <sys/resource.h> (DNE)
  #include <mpi.h> (DNE)
  #include <Box.hpp> (DNE)
  #include <BoxPartition.hpp> (DNE)
  #include <box_utils.hpp> (DNE)
  #include <Parameters.hpp> (DNE)
  #include <utils.hpp> (DNE)
  #include <driver.hpp>
    #include <cstddef> (DNE)
    #include <cmath> (DNE)
    #include <cstdlib> (DNE)
    #include <iostream> (DNE)
    #include <sstream> (DNE)
    #include <iomanip> (DNE)
    #include <box_utils.hpp> (DNE)
    #include <Vector.hpp>
      #include <vector> (DNE)
    #include <CSRMatrix.hpp>
      #include <cstddef> (DNE)
      #include <vector> (DNE)
      #include <algorithm> (DNE)
      #include <mpi.h> (DNE)
    #include <ELLMatrix.hpp>
      #include <cstddef> (DNE)
      #include <vector> (DNE)
      #include <algorithm> (DNE)
      #include <mpi.h> (DNE)
    #include <CSRMatrix.hpp>
      #include <cstddef> (DNE)
      #include <vector> (DNE)
      #include <algorithm> (DNE)
      #include <mpi.h> (DNE)
    #include <simple_mesh_description.hpp>
      #include <utils.hpp> (DNE)
      #include <set> (DNE)
      #include <map> (DNE)
    #include <SparseMatrix_functions.hpp>
      #include <cstddef> (DNE)
      #include <vector> (DNE)
      #include <set> (DNE)
      #include <algorithm> (DNE)
      #include <sstream> (DNE)
      #include <fstream> (DNE)
      #include <Vector.hpp>
        #include <vector> (DNE)
      #include <Vector_functions.hpp>
        #include <vector> (DNE)
        #include <sstream> (DNE)
        #include <fstream> (DNE)
        #include <cuda.h> (DNE)
        #include <mpi.h> (DNE)
        #include <LockingVector.hpp> (DNE)
        #include <TypeTraits.hpp> (DNE)
        #include <Vector.hpp>
          #include <vector> (DNE)
      #include <ElemData.hpp> (DNE)
      #include <MatrixInitOp.hpp>
        #include <simple_mesh_description.hpp>
          #include <utils.hpp> (DNE)
          #include <set> (DNE)
          #include <map> (DNE)
        #include <box_utils.hpp> (DNE)
        #include <CSRMatrix.hpp>
          #include <cstddef> (DNE)
          #include <vector> (DNE)
          #include <algorithm> (DNE)
          #include <mpi.h> (DNE)
        #include <ELLMatrix.hpp>
          #include <cstddef> (DNE)
          #include <vector> (DNE)
          #include <algorithm> (DNE)
          #include <mpi.h> (DNE)
        #include <algorithm> (DNE)
      #include <MatrixCopyOp.hpp>
      #include <exchange_externals.hpp>
        #include <cstdlib> (DNE)
        #include <iostream> (DNE)
        #include <mpi.h> (DNE)
        #include <outstream.hpp> (DNE)
        #include <TypeTraits.hpp> (DNE)
      #include <mytimer.hpp> (DNE)
      #include <LockingMatrix.hpp> (DNE)
      #include <mpi.h> (DNE)
    #include <generate_matrix_structure.hpp>
      #include <sstream> (DNE)
      #include <stdexcept> (DNE)
      #include <map> (DNE)
      #include <algorithm> (DNE)
      #include <simple_mesh_description.hpp>
        #include <utils.hpp> (DNE)
        #include <set> (DNE)
        #include <map> (DNE)
      #include <SparseMatrix_functions.hpp>
        #include <cstddef> (DNE)
        #include <vector> (DNE)
        #include <set> (DNE)
        #include <algorithm> (DNE)
        #include <sstream> (DNE)
        #include <fstream> (DNE)
        #include <Vector.hpp>
          #include <vector> (DNE)
        #include <Vector_functions.hpp>
          #include <vector> (DNE)
          #include <sstream> (DNE)
          #include <fstream> (DNE)
          #include <cuda.h> (DNE)
          #include <mpi.h> (DNE)
          #include <LockingVector.hpp> (DNE)
          #include <TypeTraits.hpp> (DNE)
          #include <Vector.hpp>
            #include <vector> (DNE)
        #include <ElemData.hpp> (DNE)
        #include <MatrixInitOp.hpp>
          #include <simple_mesh_description.hpp>
            #include <utils.hpp> (DNE)
            #include <set> (DNE)
            #include <map> (DNE)
          #include <box_utils.hpp> (DNE)
          #include <CSRMatrix.hpp>
            #include <cstddef> (DNE)
            #include <vector> (DNE)
            #include <algorithm> (DNE)
            #include <mpi.h> (DNE)
          #include <ELLMatrix.hpp>
            #include <cstddef> (DNE)
            #include <vector> (DNE)
            #include <algorithm> (DNE)
            #include <mpi.h> (DNE)
          #include <algorithm> (DNE)
        #include <MatrixCopyOp.hpp>
        #include <exchange_externals.hpp>
          #include <cstdlib> (DNE)
          #include <iostream> (DNE)
          #include <mpi.h> (DNE)
          #include <outstream.hpp> (DNE)
          #include <TypeTraits.hpp> (DNE)
        #include <mytimer.hpp> (DNE)
        #include <LockingMatrix.hpp> (DNE)
        #include <mpi.h> (DNE)
      #include <box_utils.hpp> (DNE)
      #include <utils.hpp> (DNE)
      #include <mpi.h> (DNE)
    #include <assemble_FE_data.hpp>
      #include <box_utils.hpp> (DNE)
      #include <simple_mesh_description.hpp>
        #include <utils.hpp> (DNE)
        #include <set> (DNE)
        #include <map> (DNE)
      #include <perform_element_loop.hpp>
        #include <BoxIterator.hpp> (DNE)
        #include <simple_mesh_description.hpp>
          #include <utils.hpp> (DNE)
          #include <set> (DNE)
          #include <map> (DNE)
        #include <SparseMatrix_functions.hpp>
          #include <cstddef> (DNE)
          #include <vector> (DNE)
          #include <set> (DNE)
          #include <algorithm> (DNE)
          #include <sstream> (DNE)
          #include <fstream> (DNE)
          #include <Vector.hpp>
            #include <vector> (DNE)
          #include <Vector_functions.hpp>
            #include <vector> (DNE)
            #include <sstream> (DNE)
            #include <fstream> (DNE)
            #include <cuda.h> (DNE)
            #include <mpi.h> (DNE)
            #include <LockingVector.hpp> (DNE)
            #include <TypeTraits.hpp> (DNE)
            #include <Vector.hpp>
              #include <vector> (DNE)
          #include <ElemData.hpp> (DNE)
          #include <MatrixInitOp.hpp>
            #include <simple_mesh_description.hpp>
              #include <utils.hpp> (DNE)
              #include <set> (DNE)
              #include <map> (DNE)
            #include <box_utils.hpp> (DNE)
            #include <CSRMatrix.hpp>
              #include <cstddef> (DNE)
              #include <vector> (DNE)
              #include <algorithm> (DNE)
              #include <mpi.h> (DNE)
            #include <ELLMatrix.hpp>
              #include <cstddef> (DNE)
              #include <vector> (DNE)
              #include <algorithm> (DNE)
              #include <mpi.h> (DNE)
            #include <algorithm> (DNE)
          #include <MatrixCopyOp.hpp>
          #include <exchange_externals.hpp>
            #include <cstdlib> (DNE)
            #include <iostream> (DNE)
            #include <mpi.h> (DNE)
            #include <outstream.hpp> (DNE)
            #include <TypeTraits.hpp> (DNE)
          #include <mytimer.hpp> (DNE)
          #include <LockingMatrix.hpp> (DNE)
          #include <mpi.h> (DNE)
        #include <box_utils.hpp> (DNE)
        #include <Hex8_box_utils.hpp>
          #include <stdexcept> (DNE)
          #include <box_utils.hpp> (DNE)
          #include <ElemData.hpp> (DNE)
          #include <simple_mesh_description.hpp>
            #include <utils.hpp> (DNE)
            #include <set> (DNE)
            #include <map> (DNE)
          #include <Hex8.hpp> (DNE)
        #include <Hex8_ElemData.hpp> (DNE)
        #include <omp.h> (DNE)
    #include <verify_solution.hpp> (DNE)
    #include <compute_matrix_stats.hpp> (DNE)
    #include <make_local_matrix.hpp>
      #include <assert.h> (DNE)
      #include <utils.hpp> (DNE)
      #include <map> (DNE)
      #include <mpi.h> (DNE)
    #include <imbalance.hpp> (DNE)
    #include <cg_solve.hpp>
      #include <cmath> (DNE)
      #include <limits> (DNE)
      #include <Vector_functions.hpp>
        #include <vector> (DNE)
        #include <sstream> (DNE)
        #include <fstream> (DNE)
        #include <cuda.h> (DNE)
        #include <mpi.h> (DNE)
        #include <LockingVector.hpp> (DNE)
        #include <TypeTraits.hpp> (DNE)
        #include <Vector.hpp>
          #include <vector> (DNE)
      #include <mytimer.hpp> (DNE)
      #include <outstream.hpp> (DNE)
    #include <time_kernels.hpp>
      #include <cmath> (DNE)
      #include <Vector_functions.hpp>
        #include <vector> (DNE)
        #include <sstream> (DNE)
        #include <fstream> (DNE)
        #include <cuda.h> (DNE)
        #include <mpi.h> (DNE)
        #include <LockingVector.hpp> (DNE)
        #include <TypeTraits.hpp> (DNE)
        #include <Vector.hpp>
          #include <vector> (DNE)
      #include <mytimer.hpp> (DNE)
      #include <cuda.h> (DNE)
    #include <outstream.hpp> (DNE)
    #include <utils.hpp> (DNE)
    #include <mytimer.hpp> (DNE)
    #include <YAML_Doc.hpp>
      #include <string> (DNE)
      #include <vector> (DNE)
      #include "YAML_Element.hpp"
        #include <string> (DNE)
        #include <vector> (DNE)
    #include <mpi.h> (DNE)
  #include <YAML_Doc.hpp>
    #include <string> (DNE)
    #include <vector> (DNE)
    #include "YAML_Element.hpp"
      #include <string> (DNE)
      #include <vector> (DNE)
  #include <miniFE_info.hpp> (DNE)
  #include <miniFE_no_info.hpp> (DNE)

""",
    "utils/BoxPartition.cpp": """utils/BoxPartition.cpp
  #include <stdio.h> (DNE)
  #include <stdlib.h> (DNE)
  #include <Box.hpp>
  #include <BoxPartition.hpp>
    #include <Box.hpp>

""",
}

EXPECTED_KERNELS = [
    {
        "file": "basic/optional/cuda/CudaNode.cuh",
        "kernel": "Tkern1D",
        "line": 19,
    },
    {
        "file": "basic/sharedmem.cuh",
        "kernel": "getPointer",
        "line": 40,
    },
    {
        "file": "basic/sharedmem.cuh",
        "kernel": "getPointer",
        "line": 45,
    },
    {
        "file": "basic/sharedmem.cuh",
        "kernel": "getPointer",
        "line": 57,
    },
    {
        "file": "src/SparseMatrix_functions.hpp",
        "kernel": "matvec_kernel",
        "line": 482,
    },
    {
        "file": "src/Vector_functions.hpp",
        "kernel": "waxby_kernel",
        "line": 133,
    },
    {
        "file": "src/Vector_functions.hpp",
        "kernel": "yaxby_kernel",
        "line": 146,
    },
    {
        "file": "src/Vector_functions.hpp",
        "kernel": "wxby_kernel",
        "line": 158,
    },
    {
        "file": "src/Vector_functions.hpp",
        "kernel": "yxby_kernel",
        "line": 170,
    },
    {
        "file": "src/Vector_functions.hpp",
        "kernel": "wx_kernel",
        "line": 181,
    },
    {
        "file": "src/Vector_functions.hpp",
        "kernel": "dyx_kernel",
        "line": 192,
    },
    {
        "file": "src/Vector_functions.hpp",
        "kernel": "wax_kernel",
        "line": 203,
    },
    {
        "file": "src/Vector_functions.hpp",
        "kernel": "dyax_kernel",
        "line": 215,
    },
    {
        "file": "src/Vector_functions.hpp",
        "kernel": "dot_kernel",
        "line": 310,
    },
    {
        "file": "src/Vector_functions.hpp",
        "kernel": "final_reduce",
        "line": 334,
    },
]
