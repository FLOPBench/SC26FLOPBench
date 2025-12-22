EXPECTED_TREE = (
    "miniFE-cuda/\n"
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
    "src/driver.hpp": """template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal> int driver(const Box& global_box, Box& my_box, Parameters& params, YAML_Doc& ydoc) (defnt)""",
    "src/main.cpp": """int main(int argc, char** argv) (defnt)
void add_params_to_yaml(YAML_Doc& doc, miniFE::Parameters& params) (defnt)
void add_configuration_to_yaml(YAML_Doc& doc, int numprocs, int numthreads) (defnt)
void add_timestring_to_yaml(YAML_Doc& doc) (defnt)""",
}

EXPECTED_FUNCTION_DECLARATIONS = {}

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
