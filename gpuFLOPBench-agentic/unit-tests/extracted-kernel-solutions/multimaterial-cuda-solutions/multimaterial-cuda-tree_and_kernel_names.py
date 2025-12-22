EXPECTED_TREE = (
    "multimaterial-cuda/\n"
    "  compact.cu\n"
    "  full_matrix.cu\n"
    "  Makefile\n"
    "  multimat.cu\n"
    "  volfrac.dat.tgz"
)

EXPECTED_MAIN_FILES = ["multimat.cu"]

EXPECTED_KERNELS = [
    {"file": "compact.cu", "kernel": "ccc_loop1", "line": 18},
    {"file": "compact.cu", "kernel": "ccc_loop1_2", "line": 60},
    {"file": "compact.cu", "kernel": "ccc_loop2", "line": 80},
    {"file": "compact.cu", "kernel": "ccc_loop2_2", "line": 128},
    {"file": "compact.cu", "kernel": "ccc_loop3", "line": 144},
]

EXPECTED_FUNCTION_DEFINITIONS = {
    "compact.cu": """char *cp_to_device(char *from, size_t size) (defnt)
void cp_to_host(char *to, char*from, size_t size) (defnt)
__global__ void ccc_loop1_2( const double * __restrict__ rho_compact_list, const double * __restrict__ Vf_compact_list, const double * __restrict__ V, double * __restrict__ rho_ave_compact, const int * __restrict__ mmc_index, const int mmc_cells, const int * __restrict__ mmc_i, const int * __restrict__ mmc_j, int sizex, int sizey) (defnt)
__global__ void ccc_loop2( const int * __restrict__ imaterial, const int * __restrict__ matids, const int * __restrict__ nextfrac, const double * __restrict__ rho_compact, const double * __restrict__ rho_compact_list, const double * __restrict__ t_compact, const double * __restrict__ t_compact_list, const double * __restrict__ Vf_compact_list, const double * __restrict__ n, double * __restrict__ p_compact, double * __restrict__ p_compact_list, int sizex, int sizey, int * __restrict__ mmc_index) (defnt)
__global__ void ccc_loop2_2( const int * __restrict__ matids, const double * __restrict__ rho_compact_list, const double * __restrict__ t_compact_list, const double * __restrict__ Vf_compact_list, const double * __restrict__ n, double * __restrict__ p_compact_list, int * __restrict__ mmc_index, int mmc_cells) (defnt)
__global__ void ccc_loop3( const int * __restrict__ imaterial, const int * __restrict__ nextfrac, const int * __restrict__ matids, const double * __restrict__ rho_compact, const double * __restrict__ rho_compact_list, double * __restrict__ rho_mat_ave_compact, double * __restrict__ rho_mat_ave_compact_list, const double * __restrict__ x, const double * __restrict__ y, int sizex, int sizey, int * __restrict__ mmc_index) (defnt)
void compact_cell_centric(full_data cc, compact_data ccc, int argc, char** argv) (defnt)
bool compact_check_results(full_data cc, compact_data ccc) (defnt)""",
    "full_matrix.cu": """void full_matrix_cell_centric(full_data cc) (defnt)
void full_matrix_material_centric(full_data cc, full_data mc) (defnt)
bool full_matrix_check_results(full_data cc, full_data mc) (defnt)""",
    "multimat.cu": """void initialise_field_rand(full_data cc, double prob2, double prob3, double prob4) (defnt)
void initialise_field_static(full_data cc) (defnt)
void initialise_field_file(full_data cc) (defnt)
int main(int argc, char** argv) (defnt)""",
}

EXPECTED_FUNCTION_DECLARATIONS = {}
