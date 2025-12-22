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
    "compact.cu": """void cp_to_host (defnt)
__global__ void ccc_loop1_2 (defnt)
__global__ void ccc_loop2 (defnt)
__global__ void ccc_loop2_2 (defnt)
__global__ void ccc_loop3 (defnt)
void compact_cell_centric (defnt)
bool compact_check_results (defnt)""",
    "full_matrix.cu": """void full_matrix_cell_centric (defnt)
void full_matrix_material_centric (defnt)
bool full_matrix_check_results (defnt)""",
    "multimat.cu": """void initialise_field_rand (defnt)
void initialise_field_static (defnt)
void initialise_field_file (defnt)
int main (defnt)""",
}

EXPECTED_FUNCTION_DECLARATIONS = {}
