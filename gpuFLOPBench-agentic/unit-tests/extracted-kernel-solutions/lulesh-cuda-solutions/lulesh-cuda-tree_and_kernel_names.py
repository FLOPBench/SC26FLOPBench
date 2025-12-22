EXPECTED_TREE = (
    "lulesh-cuda/\n"
    "  lulesh-init.cu\n"
    "  lulesh-util.cu\n"
    "  lulesh-viz.cu\n"
    "  lulesh.cu\n"
    "  lulesh.h\n"
    "  Makefile"
)

EXPECTED_MAIN_FILES = ["lulesh.cu"]

EXPECTED_KERNELS = [
    {"file": "lulesh.cu", "kernel": "fill_sig", "line": 686},
    {"file": "lulesh.cu", "kernel": "integrateStress", "line": 699},
    {"file": "lulesh.cu", "kernel": "acc_final_force", "line": 773},
    {"file": "lulesh.cu", "kernel": "hgc", "line": 804},
    {"file": "lulesh.cu", "kernel": "fb", "line": 891},
    {"file": "lulesh.cu", "kernel": "collect_final_force", "line": 1098},
    {"file": "lulesh.cu", "kernel": "accelerationForNode", "line": 1129},
    {
        "file": "lulesh.cu",
        "kernel": "applyAccelerationBoundaryConditionsForNodes",
        "line": 1147,
    },
    {"file": "lulesh.cu", "kernel": "calcVelocityForNodes", "line": 1167},
    {"file": "lulesh.cu", "kernel": "calcPositionForNodes", "line": 1197},
    {"file": "lulesh.cu", "kernel": "calcKinematicsForElems", "line": 1214},
    {"file": "lulesh.cu", "kernel": "calcStrainRates", "line": 1327},
    {
        "file": "lulesh.cu",
        "kernel": "calcMonotonicQGradientsForElems",
        "line": 1356,
    },
    {"file": "lulesh.cu", "kernel": "calcMonotonicQForElems", "line": 1514},
    {"file": "lulesh.cu", "kernel": "applyMaterialPropertiesForElems", "line": 1686},
]

EXPECTED_FUNCTION_DEFINITIONS = {
    "lulesh-init.cu": """Domain::Domain (defnt)
void Domain::BuildMesh (defnt)
void Domain::SetupThreadSupportStructures (defnt)
void Domain::SetupCommBuffers (defnt)
void Domain::CreateRegionIndexSets (defnt)
regElemSize (defnt)
regElemSize (defnt)
void Domain::SetupSymmetryPlanes (defnt)
void Domain::SetupElementConnectivities (defnt)
void Domain::SetupBoundaryConditions (defnt)
void InitMeshDecomp (defnt)""",
    "lulesh-util.cu": """int StrToInt (defnt)
void PrintCommandLineOptions (defnt)
void ParseError (defnt)
void ParseCommandLineOptions (defnt)
void VerifyAndWriteFinalOutput (defnt)""",
    "lulesh-viz.cu": """void DumpToVisit (defnt)
void DumpDomainToVisit (defnt)
void DumpMultiblockObjects (defnt)
void DumpToVisit (defnt)""",
    "lulesh.cu": """__device__ void SumElemFaceNormal (defnt)
__device__ void CalcElemShapeFunctionDerivatives (defnt)
__device__ void CalcElemNodeNormals (defnt)
__device__ void SumElemStressesToNodeForces (defnt)
__device__ void VoluDer (defnt)
__device__ void CalcElemVolumeDerivative (defnt)
__device__ Real_t calcElemVolume (defnt)
__device__ Real_t CalcElemVolume (defnt)
__device__ Real_t AreaFace (defnt)
__device__ Real_t CalcElemCharacteristicLength (defnt)
__device__ void CalcElemVelocityGradient (defnt)
__global__ void fill_sig (defnt)
__global__ void integrateStress (defnt)
__global__ void acc_final_force (defnt)
__global__ void hgc (defnt)
__global__ void fb (defnt)
__global__ void collect_final_force (defnt)
__global__ void accelerationForNode (defnt)
__global__ void applyAccelerationBoundaryConditionsForNodes (defnt)
__global__ void calcVelocityForNodes (defnt)
__global__ void calcPositionForNodes (defnt)
__global__ void calcKinematicsForElems (defnt)
__global__ void calcStrainRates (defnt)
__global__ void calcMonotonicQGradientsForElems (defnt)
__global__ void calcMonotonicQForElems (defnt)
__global__ void applyMaterialPropertiesForElems (defnt)
template <typename T> void Release (defnt)
void TimeIncrement (defnt)
void CalcCourantConstraintForElems (defnt)
void CalcHydroConstraintForElems (defnt)
void CalcTimeConstraintsForElems (defnt)
int main (defnt)""",
    "lulesh.h": """real4 SQRT (defnt)
real8 SQRT (defnt)
real10 SQRT (defnt)
real4 CBRT (defnt)
real8 CBRT (defnt)
real10 CBRT (defnt)
real4 FABS (defnt)
real8 FABS (defnt)
real10 FABS (defnt)
void Domain::AllocateNodePersistent (defnt)
void Domain::AllocateElemPersistent (defnt)
void Domain::AllocateGradients (defnt)
void Domain::DeallocateGradients (defnt)
void Domain::AllocateStrains (defnt)
void Domain::DeallocateStrains (defnt)
Index_t Domain::symmX (defnt)
Index_t Domain::symmY (defnt)
Index_t Domain::symmZ (defnt)
bool Domain::symmXempty (defnt)
bool Domain::symmYempty (defnt)
bool Domain::symmZempty (defnt)
Index_t Domain::nodeElemCount (defnt)
Real_t Domain::u_cut (defnt)
Real_t Domain::e_cut (defnt)
Real_t Domain::p_cut (defnt)
Real_t Domain::q_cut (defnt)
Real_t Domain::v_cut (defnt)
Real_t Domain::hgcoef (defnt)
Real_t Domain::qstop (defnt)
Real_t Domain::monoq_max_slope (defnt)
Real_t Domain::monoq_limiter_mult (defnt)
Real_t Domain::ss4o3 (defnt)
Real_t Domain::qlc_monoq (defnt)
Real_t Domain::qqc_monoq (defnt)
Real_t Domain::qqc (defnt)
Real_t Domain::eosvmax (defnt)
Real_t Domain::eosvmin (defnt)
Real_t Domain::pmin (defnt)
Real_t Domain::emin (defnt)
Real_t Domain::dvovmax (defnt)
Real_t Domain::refdens (defnt)""",
}

EXPECTED_FUNCTION_DECLARATIONS = {
    "lulesh.h": """void Domain::BuildMesh (decl)
void Domain::SetupThreadSupportStructures (decl)
void Domain::CreateRegionIndexSets (decl)
void Domain::SetupCommBuffers (decl)
void Domain::SetupSymmetryPlanes (decl)
void Domain::SetupElementConnectivities (decl)
void Domain::SetupBoundaryConditions (decl)""",
}
