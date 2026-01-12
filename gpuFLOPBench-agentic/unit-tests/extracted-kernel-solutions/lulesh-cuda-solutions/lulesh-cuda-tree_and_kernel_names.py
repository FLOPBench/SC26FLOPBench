EXPECTED_TREE = (
    "/\n"
    "  lulesh-init.cu\n"
    "  lulesh-util.cu\n"
    "  lulesh-viz.cu\n"
    "  lulesh.cu\n"
    "  lulesh.h\n"
    "  Makefile"
)

EXPECTED_MAIN_FILES = ["lulesh.cu"]

EXPECTED_INCLUDE_TREES = {
    "lulesh.cu": """lulesh.cu
  #include <math.h> (DNE)
  #include <stdio.h> (DNE)
  #include <stdlib.h> (DNE)
  #include <string.h> (DNE)
  #include <ctype.h> (DNE)
  #include <time.h> (DNE)
  #include <sys/time.h> (DNE)
  #include <unistd.h> (DNE)
  #include <climits> (DNE)
  #include <iostream> (DNE)
  #include <sstream> (DNE)
  #include <limits> (DNE)
  #include <fstream> (DNE)
  #include <string> (DNE)
  #include <random> (DNE)
  #include <cassert> (DNE)
  #include "lulesh.h"
    #include <math.h> (DNE)
    #include <vector> (DNE)
    #include <cuda.h> (DNE)

""",
}

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
    "lulesh-init.cu": """Domain::Domain(Int_t numRanks, Index_t colLoc, Index_t rowLoc, Index_t planeLoc, Index_t nx, int tp, int nr, int balance, Int_t cost) (defnt)
void Domain::BuildMesh(Int_t nx, Int_t edgeNodes, Int_t edgeElems) (defnt)
void Domain::SetupThreadSupportStructures() (defnt)
void Domain::SetupCommBuffers(Int_t edgeNodes) (defnt)
void Domain::CreateRegionIndexSets(Int_t nr, Int_t balance) (defnt)
regElemSize(i) (defnt)
regElemSize(i) (defnt)
void Domain::SetupSymmetryPlanes(Int_t edgeNodes) (defnt)
void Domain::SetupElementConnectivities(Int_t edgeElems) (defnt)
void Domain::SetupBoundaryConditions(Int_t edgeElems) (defnt)
void InitMeshDecomp(Int_t numRanks, Int_t myRank, Int_t *col, Int_t *row, Int_t *plane, Int_t *side) (defnt)""",
"lulesh-util.cu": """int StrToInt(const char *token, int *retVal) (defnt)
static void PrintCommandLineOptions(char *execname, int myRank) (defnt)
static void ParseError(const char *message, int myRank) (defnt)
void ParseCommandLineOptions(int argc, char *argv[], int myRank, struct cmdLineOpts *opts) (defnt)
void VerifyAndWriteFinalOutput(Real_t elapsed_time, Domain& locDom, Int_t nx, Int_t numRanks) (defnt)""",
"lulesh-viz.cu": """void DumpToVisit(Domain& domain, int numFiles, int myRank, int numRanks) (defnt)
static void DumpDomainToVisit(DBfile *db, Domain& domain, int myRank) (defnt)
void DumpMultiblockObjects(DBfile *db, char basename[], int numRanks) (defnt)
void DumpToVisit(Domain& domain, int numFiles, int myRank, int numRanks) (defnt)""",
    "lulesh.cu": """__device__ static inline void SumElemFaceNormal(Real_t *normalX0, Real_t *normalY0, Real_t *normalZ0, Real_t *normalX1, Real_t *normalY1, Real_t *normalZ1, Real_t *normalX2, Real_t *normalY2, Real_t *normalZ2, Real_t *normalX3, Real_t *normalY3, Real_t *normalZ3, const Real_t x0, const Real_t y0, const Real_t z0, const Real_t x1, const Real_t y1, const Real_t z1, const Real_t x2, const Real_t y2, const Real_t z2, const Real_t x3, const Real_t y3, const Real_t z3) (defnt)
__device__ static inline void CalcElemShapeFunctionDerivatives( Real_t const x[], Real_t const y[], Real_t const z[], Real_t b[][8], Real_t* const volume ) (defnt)
__device__ static inline void CalcElemNodeNormals(Real_t pfx[8], Real_t pfy[8], Real_t pfz[8], const Real_t x[8], const Real_t y[8], const Real_t z[8]) (defnt)
__device__ static inline void SumElemStressesToNodeForces( const Real_t B[][8], const Real_t stress_xx, const Real_t stress_yy, const Real_t stress_zz, Real_t fx[], Real_t fy[], Real_t fz[] ) (defnt)
__device__ static inline void VoluDer(const Real_t x0, const Real_t x1, const Real_t x2, const Real_t x3, const Real_t x4, const Real_t x5, const Real_t y0, const Real_t y1, const Real_t y2, const Real_t y3, const Real_t y4, const Real_t y5, const Real_t z0, const Real_t z1, const Real_t z2, const Real_t z3, const Real_t z4, const Real_t z5, Real_t* dvdx, Real_t* dvdy, Real_t* dvdz) (defnt)
__device__ static inline void CalcElemVolumeDerivative(Real_t dvdx[8], Real_t dvdy[8], Real_t dvdz[8], const Real_t x[8], const Real_t y[8], const Real_t z[8]) (defnt)
__host__ __device__ static inline Real_t calcElemVolume( const Real_t x0, const Real_t x1, const Real_t x2, const Real_t x3, const Real_t x4, const Real_t x5, const Real_t x6, const Real_t x7, const Real_t y0, const Real_t y1, const Real_t y2, const Real_t y3, const Real_t y4, const Real_t y5, const Real_t y6, const Real_t y7, const Real_t z0, const Real_t z1, const Real_t z2, const Real_t z3, const Real_t z4, const Real_t z5, const Real_t z6, const Real_t z7 ) (defnt)
__host__ __device__ Real_t CalcElemVolume( const Real_t x[8], const Real_t y[8], const Real_t z[8] ) (defnt)
__device__ static inline Real_t AreaFace( const Real_t x0, const Real_t x1, const Real_t x2, const Real_t x3, const Real_t y0, const Real_t y1, const Real_t y2, const Real_t y3, const Real_t z0, const Real_t z1, const Real_t z2, const Real_t z3) (defnt)
__device__ static inline Real_t CalcElemCharacteristicLength( const Real_t x[8], const Real_t y[8], const Real_t z[8], const Real_t volume) (defnt)
__device__ static inline void CalcElemVelocityGradient( const Real_t* const xvel, const Real_t* const yvel, const Real_t* const zvel, const Real_t b[][8], const Real_t detJ, Real_t* const d ) (defnt)
__global__ void fill_sig( Real_t *__restrict__ sigxx, Real_t *__restrict__ sigyy, Real_t *__restrict__ sigzz, const Real_t *__restrict__ p, const Real_t *__restrict__ q, const Index_t numElem ) (defnt)
__global__ void integrateStress ( Real_t *__restrict__ fx_elem, Real_t *__restrict__ fy_elem, Real_t *__restrict__ fz_elem, const Real_t *__restrict__ x, const Real_t *__restrict__ y, const Real_t *__restrict__ z, const Index_t *__restrict__ nodelist, const Real_t *__restrict__ sigxx, const Real_t *__restrict__ sigyy, const Real_t *__restrict__ sigzz, Real_t *__restrict__ determ, const Index_t numElem) (defnt)
__global__ void acc_final_force ( const Real_t *__restrict__ fx_elem, const Real_t *__restrict__ fy_elem, const Real_t *__restrict__ fz_elem, Real_t *__restrict__ fx, Real_t *__restrict__ fy, Real_t *__restrict__ fz, const Index_t *__restrict__ nodeElemStart, const Index_t *__restrict__ nodeElemCornerList, const Index_t numNode) (defnt)
__global__ void hgc ( Real_t *__restrict__ dvdx, Real_t *__restrict__ dvdy, Real_t *__restrict__ dvdz, Real_t *__restrict__ x8n, Real_t *__restrict__ y8n, Real_t *__restrict__ z8n, Real_t *__restrict__ determ, const Real_t *__restrict__ x, const Real_t *__restrict__ y, const Real_t *__restrict__ z, const Index_t *__restrict__ nodelist, const Real_t *__restrict__ volo, const Real_t *__restrict__ v, int *__restrict__ vol_error, const Index_t numElem ) (defnt)
__global__ void fb ( const Real_t *__restrict__ dvdx, const Real_t *__restrict__ dvdy, const Real_t *__restrict__ dvdz, const Real_t *__restrict__ x8n, const Real_t *__restrict__ y8n, const Real_t *__restrict__ z8n, const Real_t *__restrict__ determ, const Real_t *__restrict__ xd, const Real_t *__restrict__ yd, const Real_t *__restrict__ zd, const Real_t *__restrict__ ss, const Real_t *__restrict__ elemMass, const Index_t *__restrict__ nodelist, const Real_t *__restrict__ gamma, Real_t *__restrict__ fx_elem, Real_t *__restrict__ fy_elem, Real_t *__restrict__ fz_elem, Real_t hgcoef, const Index_t numElem ) (defnt)
__global__ void collect_final_force ( const Real_t *__restrict__ fx_elem, const Real_t *__restrict__ fy_elem, const Real_t *__restrict__ fz_elem, Real_t *__restrict__ fx, Real_t *__restrict__ fy, Real_t *__restrict__ fz, const Index_t *__restrict__ nodeElemStart, const Index_t *__restrict__ nodeElemCornerList, const Index_t numNode ) (defnt)
__global__ void accelerationForNode ( const Real_t *__restrict__ fx, const Real_t *__restrict__ fy, const Real_t *__restrict__ fz, const Real_t *__restrict__ nodalMass, Real_t *__restrict__ xdd, Real_t *__restrict__ ydd, Real_t *__restrict__ zdd, const Index_t numNode) (defnt)
__global__ void applyAccelerationBoundaryConditionsForNodes ( const Index_t *__restrict__ symmX, const Index_t *__restrict__ symmY, const Index_t *__restrict__ symmZ, Real_t *__restrict__ xdd, Real_t *__restrict__ ydd, Real_t *__restrict__ zdd, const Index_t s1, const Index_t s2, const Index_t s3, const Index_t numNodeBC ) (defnt)
__global__ void calcVelocityForNodes ( Real_t *__restrict__ xd, Real_t *__restrict__ yd, Real_t *__restrict__ zd, const Real_t *__restrict__ xdd, const Real_t *__restrict__ ydd, const Real_t *__restrict__ zdd, const Real_t deltaTime, const Real_t u_cut, const Index_t numNode ) (defnt)
__global__ void calcPositionForNodes ( Real_t *__restrict__ x, Real_t *__restrict__ y, Real_t *__restrict__ z, const Real_t *__restrict__ xd, const Real_t *__restrict__ yd, const Real_t *__restrict__ zd, const Real_t deltaTime, const Index_t numNode) (defnt)
__global__ void calcKinematicsForElems ( const Real_t *__restrict__ xd, const Real_t *__restrict__ yd, const Real_t *__restrict__ zd, const Real_t *__restrict__ x, const Real_t *__restrict__ y, const Real_t *__restrict__ z, const Index_t *__restrict__ nodeList, const Real_t *__restrict__ volo, const Real_t *__restrict__ v, Real_t *__restrict__ delv, Real_t *__restrict__ arealg, Real_t *__restrict__ dxx, Real_t *__restrict__ dyy, Real_t *__restrict__ dzz, Real_t *__restrict__ vnew, const Real_t deltaTime, const Index_t numElem ) (defnt)
__global__ void calcStrainRates( Real_t *__restrict__ dxx, Real_t *__restrict__ dyy, Real_t *__restrict__ dzz, const Real_t *__restrict__ vnew, Real_t *__restrict__ vdov, int *__restrict__ vol_error, const Index_t numElem ) (defnt)
__global__ void calcMonotonicQGradientsForElems ( const Real_t *__restrict__ xd, const Real_t *__restrict__ yd, const Real_t *__restrict__ zd, const Real_t *__restrict__ x, const Real_t *__restrict__ y, const Real_t *__restrict__ z, const Index_t *__restrict__ nodelist, const Real_t *__restrict__ volo, Real_t *__restrict__ delv_eta, Real_t *__restrict__ delx_eta, Real_t *__restrict__ delv_zeta, Real_t *__restrict__ delx_zeta, Real_t *__restrict__ delv_xi, Real_t *__restrict__ delx_xi, const Real_t *__restrict__ vnew, const Index_t numElem ) (defnt)
__global__ void calcMonotonicQForElems ( const Index_t *__restrict__ elemBC, const Real_t *__restrict__ elemMass, Real_t *__restrict__ ql, Real_t *__restrict__ qq, const Real_t *__restrict__ vdov, const Real_t *__restrict__ volo, const Real_t *__restrict__ delv_eta, const Real_t *__restrict__ delx_eta, const Real_t *__restrict__ delv_zeta, const Real_t *__restrict__ delx_zeta, const Real_t *__restrict__ delv_xi, const Real_t *__restrict__ delx_xi, const Index_t *__restrict__ lxim, const Index_t *__restrict__ lxip, const Index_t *__restrict__ lzetam, const Index_t *__restrict__ lzetap, const Index_t *__restrict__ letap, const Index_t *__restrict__ letam, const Real_t *__restrict__ vnew, const Real_t monoq_limiter_mult, const Real_t monoq_max_slope, const Real_t qlc_monoq, const Real_t qqc_monoq, const Index_t numElem ) (defnt)
__global__ void applyMaterialPropertiesForElems( const Real_t *__restrict__ ql, const Real_t *__restrict__ qq, const Real_t *__restrict__ delv, const Index_t *__restrict__ elemRep, const Index_t *__restrict__ elemElem, Real_t *__restrict__ q, Real_t *__restrict__ p, Real_t *__restrict__ e, Real_t *__restrict__ ss, Real_t *__restrict__ v, Real_t *__restrict__ vnewc, const Real_t e_cut, const Real_t p_cut, const Real_t ss4o3, const Real_t q_cut, const Real_t v_cut, const Real_t eosvmax, const Real_t eosvmin, const Real_t pmin, const Real_t emin, const Real_t rho0, const Index_t numElem ) (defnt)
template <typename T> T *Allocate(size_t size) (defnt)
template <typename T> void Release(T **ptr) (defnt)
static inline void TimeIncrement(Domain& domain) (defnt)
static inline void CalcCourantConstraintForElems(Domain &domain, Index_t length, Index_t *regElemlist, Real_t qqc, Real_t& dtcourant) (defnt)
static inline void CalcHydroConstraintForElems(Domain &domain, Index_t length, Index_t *regElemlist, Real_t dvovmax, Real_t& dthydro) (defnt)
static inline void CalcTimeConstraintsForElems(Domain& domain) (defnt)
int main(int argc, char *argv[]) (defnt)""",
    "lulesh.h": """inline real4 SQRT(real4 arg) (defnt)
inline real8 SQRT(real8 arg) (defnt)
inline real10 SQRT(real10 arg) (defnt)
inline real4 CBRT(real4 arg) (defnt)
inline real8 CBRT(real8 arg) (defnt)
inline real10 CBRT(real10 arg) (defnt)
inline real4 FABS(real4 arg) (defnt)
inline real8 FABS(real8 arg) (defnt)
inline real10 FABS(real10 arg) (defnt)
void AllocateNodePersistent(Int_t numNode) (defnt)
void AllocateElemPersistent(Int_t numElem) (defnt)
void AllocateGradients(Int_t numElem, Int_t allElem) (defnt)
void DeallocateGradients() (defnt)
void AllocateStrains(Int_t numElem) (defnt)
void DeallocateStrains() (defnt)
Real_t & x(Index_t idx) (defnt)
Real_t & y(Index_t idx) (defnt)
Real_t & z(Index_t idx) (defnt)
Real_t & xd(Index_t idx) (defnt)
Real_t & yd(Index_t idx) (defnt)
Real_t & zd(Index_t idx) (defnt)
Real_t & xdd(Index_t idx) (defnt)
Real_t & ydd(Index_t idx) (defnt)
Real_t & zdd(Index_t idx) (defnt)
Real_t & fx(Index_t idx) (defnt)
Real_t & fy(Index_t idx) (defnt)
Real_t & fz(Index_t idx) (defnt)
Real_t & nodalMass(Index_t idx) (defnt)
Index_t symmX(Index_t idx) (defnt)
Index_t symmY(Index_t idx) (defnt)
Index_t symmZ(Index_t idx) (defnt)
bool symmXempty() (defnt)
bool symmYempty() (defnt)
bool symmZempty() (defnt)
Index_t & regElemSize(Index_t idx) (defnt)
Index_t & regNumList(Index_t idx) (defnt)
Index_t * regNumList() (defnt)
Index_t * regElemlist(Int_t r) (defnt)
Index_t & regElemlist(Int_t r, Index_t idx) (defnt)
Index_t * nodelist(Index_t idx) (defnt)
Index_t & lxim(Index_t idx) (defnt)
Index_t & lxip(Index_t idx) (defnt)
Index_t & letam(Index_t idx) (defnt)
Index_t & letap(Index_t idx) (defnt)
Index_t & lzetam(Index_t idx) (defnt)
Index_t & lzetap(Index_t idx) (defnt)
Int_t & elemBC(Index_t idx) (defnt)
Real_t & dxx(Index_t idx) (defnt)
Real_t & dyy(Index_t idx) (defnt)
Real_t & dzz(Index_t idx) (defnt)
Real_t & delv_xi(Index_t idx) (defnt)
Real_t & delv_eta(Index_t idx) (defnt)
Real_t & delv_zeta(Index_t idx) (defnt)
Real_t & delx_xi(Index_t idx) (defnt)
Real_t & delx_eta(Index_t idx) (defnt)
Real_t & delx_zeta(Index_t idx) (defnt)
Real_t & e(Index_t idx) (defnt)
Real_t & p(Index_t idx) (defnt)
Real_t & q(Index_t idx) (defnt)
Real_t & ql(Index_t idx) (defnt)
Real_t & qq(Index_t idx) (defnt)
Real_t & v(Index_t idx) (defnt)
Real_t & delv(Index_t idx) (defnt)
Real_t & volo(Index_t idx) (defnt)
Real_t & vdov(Index_t idx) (defnt)
Real_t & arealg(Index_t idx) (defnt)
Real_t & ss(Index_t idx) (defnt)
Real_t & elemMass(Index_t idx) (defnt)
Index_t & elemRep(Index_t idx) (defnt)
Index_t & elemElem(Index_t idx) (defnt)
Index_t nodeElemCount(Index_t idx) (defnt)
Index_t *nodeElemCornerList(Index_t idx) (defnt)
Real_t u_cut() const (defnt)
Real_t e_cut() const (defnt)
Real_t p_cut() const (defnt)
Real_t q_cut() const (defnt)
Real_t v_cut() const (defnt)
Real_t hgcoef() const (defnt)
Real_t qstop() const (defnt)
Real_t monoq_max_slope() const (defnt)
Real_t monoq_limiter_mult() const (defnt)
Real_t ss4o3() const (defnt)
Real_t qlc_monoq() const (defnt)
Real_t qqc_monoq() const (defnt)
Real_t qqc() const (defnt)
Real_t eosvmax() const (defnt)
Real_t eosvmin() const (defnt)
Real_t pmin() const (defnt)
Real_t emin() const (defnt)
Real_t dvovmax() const (defnt)
Real_t refdens() const (defnt)
Real_t & time() (defnt)
Real_t & deltatime() (defnt)
Real_t & deltatimemultlb() (defnt)
Real_t & deltatimemultub() (defnt)
Real_t & stoptime() (defnt)
Real_t & dtcourant() (defnt)
Real_t & dthydro() (defnt)
Real_t & dtmax() (defnt)
Real_t & dtfixed() (defnt)
Int_t & cycle() (defnt)
Index_t & numRanks() (defnt)
Index_t & colLoc() (defnt)
Index_t & rowLoc() (defnt)
Index_t & planeLoc() (defnt)
Index_t & tp() (defnt)
Index_t & sizeX() (defnt)
Index_t & sizeY() (defnt)
Index_t & sizeZ() (defnt)
Index_t & numReg() (defnt)
Int_t & cost() (defnt)
Index_t & numElem() (defnt)
Index_t & numNode() (defnt)
Index_t & maxPlaneSize() (defnt)
Index_t & maxEdgeSize() (defnt)""",
}

EXPECTED_FUNCTION_DECLARATIONS = {
    "lulesh.h": """void BuildMesh(Int_t nx, Int_t edgeNodes, Int_t edgeElems) (decl)
void SetupThreadSupportStructures() (decl)
void CreateRegionIndexSets(Int_t nreg, Int_t balance) (decl)
void SetupCommBuffers(Int_t edgeNodes) (decl)
void SetupSymmetryPlanes(Int_t edgeNodes) (decl)
void SetupElementConnectivities(Int_t edgeElems) (decl)
void SetupBoundaryConditions(Int_t edgeElems) (decl)""",
}
