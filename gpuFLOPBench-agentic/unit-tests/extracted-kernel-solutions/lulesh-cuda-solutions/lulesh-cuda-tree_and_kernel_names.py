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

EXPECTED_FUNCTION_LIST_JSON = '''
[
    {
        "file": "lulesh.cu",
        "functions": [
            {
                "kind": "defnt",
                "lines": 35,
                "name": "SumElemFaceNormal",
                "offset": 183,
                "qualifiers": [
                    "__device__",
                    "static",
                    "inline"
                ],
                "return_type": "void",
                "signature": "SumElemFaceNormal(Real_t *normalX0, Real_t *normalY0, Real_t *normalZ0, Real_t *normalX1, Real_t *normalY1, Real_t *normalZ1, Real_t *normalX2, Real_t *normalY2, Real_t *normalZ2, Real_t *normalX3, Real_t *normalY3, Real_t *normalZ3, const Real_t x0, const Real_t y0, const Real_t z0, const Real_t x1, const Real_t y1, const Real_t z1, const Real_t x2, const Real_t y2, const Real_t z2, const Real_t x3, const Real_t y3, const Real_t z3)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 88,
                "name": "CalcElemShapeFunctionDerivatives",
                "offset": 220,
                "qualifiers": [
                    "__device__",
                    "static",
                    "inline"
                ],
                "return_type": "void",
                "signature": "CalcElemShapeFunctionDerivatives( Real_t const x[], Real_t const y[], Real_t const z[], Real_t b[][8], Real_t* const volume )",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 56,
                "name": "CalcElemNodeNormals",
                "offset": 309,
                "qualifiers": [
                    "__device__",
                    "static",
                    "inline"
                ],
                "return_type": "void",
                "signature": "CalcElemNodeNormals(Real_t pfx[8], Real_t pfy[8], Real_t pfz[8], const Real_t x[8], const Real_t y[8], const Real_t z[8])",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 13,
                "name": "SumElemStressesToNodeForces",
                "offset": 367,
                "qualifiers": [
                    "__device__",
                    "static",
                    "inline"
                ],
                "return_type": "void",
                "signature": "SumElemStressesToNodeForces( const Real_t B[][8], const Real_t stress_xx, const Real_t stress_yy, const Real_t stress_zz, Real_t fx[], Real_t fy[], Real_t fz[] )",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 29,
                "name": "VoluDer",
                "offset": 385,
                "qualifiers": [
                    "__device__",
                    "static",
                    "inline"
                ],
                "return_type": "void",
                "signature": "VoluDer(const Real_t x0, const Real_t x1, const Real_t x2, const Real_t x3, const Real_t x4, const Real_t x5, const Real_t y0, const Real_t y1, const Real_t y2, const Real_t y3, const Real_t y4, const Real_t y5, const Real_t z0, const Real_t z1, const Real_t z2, const Real_t z3, const Real_t z4, const Real_t z5, Real_t* dvdx, Real_t* dvdy, Real_t* dvdz)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 41,
                "name": "CalcElemVolumeDerivative",
                "offset": 417,
                "qualifiers": [
                    "__device__",
                    "static",
                    "inline"
                ],
                "return_type": "void",
                "signature": "CalcElemVolumeDerivative(Real_t dvdx[8], Real_t dvdy[8], Real_t dvdz[8], const Real_t x[8], const Real_t y[8], const Real_t z[8])",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 84,
                "name": "calcElemVolume",
                "offset": 460,
                "qualifiers": [
                    "__host__",
                    "__device__",
                    "static",
                    "inline"
                ],
                "return_type": "Real_t",
                "signature": "calcElemVolume( const Real_t x0, const Real_t x1, const Real_t x2, const Real_t x3, const Real_t x4, const Real_t x5, const Real_t x6, const Real_t x7, const Real_t y0, const Real_t y1, const Real_t y2, const Real_t y3, const Real_t y4, const Real_t y5, const Real_t y6, const Real_t y7, const Real_t z0, const Real_t z1, const Real_t z2, const Real_t z3, const Real_t z4, const Real_t z5, const Real_t z6, const Real_t z7 )",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 8,
                "name": "CalcElemVolume",
                "offset": 545,
                "qualifiers": [
                    "__host__",
                    "__device__"
                ],
                "return_type": "Real_t",
                "signature": "CalcElemVolume( const Real_t x[8], const Real_t y[8], const Real_t z[8] )",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 21,
                "name": "AreaFace",
                "offset": 554,
                "qualifiers": [
                    "__device__",
                    "static",
                    "inline"
                ],
                "return_type": "Real_t",
                "signature": "AreaFace( const Real_t x0, const Real_t x1, const Real_t x2, const Real_t x3, const Real_t y0, const Real_t y1, const Real_t y2, const Real_t y3, const Real_t z0, const Real_t z1, const Real_t z2, const Real_t z3)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 42,
                "name": "CalcElemCharacteristicLength",
                "offset": 579,
                "qualifiers": [
                    "__device__",
                    "static",
                    "inline"
                ],
                "return_type": "Real_t",
                "signature": "CalcElemCharacteristicLength( const Real_t x[8], const Real_t y[8], const Real_t z[8], const Real_t volume)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 62,
                "name": "CalcElemVelocityGradient",
                "offset": 623,
                "qualifiers": [
                    "__device__",
                    "static",
                    "inline"
                ],
                "return_type": "void",
                "signature": "CalcElemVelocityGradient( const Real_t* const xvel, const Real_t* const yvel, const Real_t* const zvel, const Real_t b[][8], const Real_t detJ, Real_t* const d )",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 12,
                "name": "fill_sig",
                "offset": 686,
                "qualifiers": [
                    "__global__"
                ],
                "return_type": "void",
                "signature": "fill_sig( Real_t *__restrict__ sigxx, Real_t *__restrict__ sigyy, Real_t *__restrict__ sigzz, const Real_t *__restrict__ p, const Real_t *__restrict__ q, const Index_t numElem )",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 73,
                "name": "integrateStress",
                "offset": 699,
                "qualifiers": [
                    "__global__"
                ],
                "return_type": "void",
                "signature": "integrateStress ( Real_t *__restrict__ fx_elem, Real_t *__restrict__ fy_elem, Real_t *__restrict__ fz_elem, const Real_t *__restrict__ x, const Real_t *__restrict__ y, const Real_t *__restrict__ z, const Index_t *__restrict__ nodelist, const Real_t *__restrict__ sigxx, const Real_t *__restrict__ sigyy, const Real_t *__restrict__ sigzz, Real_t *__restrict__ determ, const Index_t numElem)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 30,
                "name": "acc_final_force",
                "offset": 773,
                "qualifiers": [
                    "__global__"
                ],
                "return_type": "void",
                "signature": "acc_final_force ( const Real_t *__restrict__ fx_elem, const Real_t *__restrict__ fy_elem, const Real_t *__restrict__ fz_elem, Real_t *__restrict__ fx, Real_t *__restrict__ fy, Real_t *__restrict__ fz, const Index_t *__restrict__ nodeElemStart, const Index_t *__restrict__ nodeElemCornerList, const Index_t numNode)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 86,
                "name": "hgc",
                "offset": 804,
                "qualifiers": [
                    "__global__"
                ],
                "return_type": "void",
                "signature": "hgc ( Real_t *__restrict__ dvdx, Real_t *__restrict__ dvdy, Real_t *__restrict__ dvdz, Real_t *__restrict__ x8n, Real_t *__restrict__ y8n, Real_t *__restrict__ z8n, Real_t *__restrict__ determ, const Real_t *__restrict__ x, const Real_t *__restrict__ y, const Real_t *__restrict__ z, const Index_t *__restrict__ nodelist, const Real_t *__restrict__ volo, const Real_t *__restrict__ v, int *__restrict__ vol_error, const Index_t numElem )",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 206,
                "name": "fb",
                "offset": 891,
                "qualifiers": [
                    "__global__"
                ],
                "return_type": "void",
                "signature": "fb ( const Real_t *__restrict__ dvdx, const Real_t *__restrict__ dvdy, const Real_t *__restrict__ dvdz, const Real_t *__restrict__ x8n, const Real_t *__restrict__ y8n, const Real_t *__restrict__ z8n, const Real_t *__restrict__ determ, const Real_t *__restrict__ xd, const Real_t *__restrict__ yd, const Real_t *__restrict__ zd, const Real_t *__restrict__ ss, const Real_t *__restrict__ elemMass, const Index_t *__restrict__ nodelist, const Real_t *__restrict__ gamma, Real_t *__restrict__ fx_elem, Real_t *__restrict__ fy_elem, Real_t *__restrict__ fz_elem, Real_t hgcoef, const Index_t numElem )",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 30,
                "name": "collect_final_force",
                "offset": 1098,
                "qualifiers": [
                    "__global__"
                ],
                "return_type": "void",
                "signature": "collect_final_force ( const Real_t *__restrict__ fx_elem, const Real_t *__restrict__ fy_elem, const Real_t *__restrict__ fz_elem, Real_t *__restrict__ fx, Real_t *__restrict__ fy, Real_t *__restrict__ fz, const Index_t *__restrict__ nodeElemStart, const Index_t *__restrict__ nodeElemCornerList, const Index_t numNode )",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 17,
                "name": "accelerationForNode",
                "offset": 1129,
                "qualifiers": [
                    "__global__"
                ],
                "return_type": "void",
                "signature": "accelerationForNode ( const Real_t *__restrict__ fx, const Real_t *__restrict__ fy, const Real_t *__restrict__ fz, const Real_t *__restrict__ nodalMass, Real_t *__restrict__ xdd, Real_t *__restrict__ ydd, Real_t *__restrict__ zdd, const Index_t numNode)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 19,
                "name": "applyAccelerationBoundaryConditionsForNodes",
                "offset": 1147,
                "qualifiers": [
                    "__global__"
                ],
                "return_type": "void",
                "signature": "applyAccelerationBoundaryConditionsForNodes ( const Index_t *__restrict__ symmX, const Index_t *__restrict__ symmY, const Index_t *__restrict__ symmZ, Real_t *__restrict__ xdd, Real_t *__restrict__ ydd, Real_t *__restrict__ zdd, const Index_t s1, const Index_t s2, const Index_t s3, const Index_t numNodeBC )",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 29,
                "name": "calcVelocityForNodes",
                "offset": 1167,
                "qualifiers": [
                    "__global__"
                ],
                "return_type": "void",
                "signature": "calcVelocityForNodes ( Real_t *__restrict__ xd, Real_t *__restrict__ yd, Real_t *__restrict__ zd, const Real_t *__restrict__ xdd, const Real_t *__restrict__ ydd, const Real_t *__restrict__ zdd, const Real_t deltaTime, const Real_t u_cut, const Index_t numNode )",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 16,
                "name": "calcPositionForNodes",
                "offset": 1197,
                "qualifiers": [
                    "__global__"
                ],
                "return_type": "void",
                "signature": "calcPositionForNodes ( Real_t *__restrict__ x, Real_t *__restrict__ y, Real_t *__restrict__ z, const Real_t *__restrict__ xd, const Real_t *__restrict__ yd, const Real_t *__restrict__ zd, const Real_t deltaTime, const Index_t numNode)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 112,
                "name": "calcKinematicsForElems",
                "offset": 1214,
                "qualifiers": [
                    "__global__"
                ],
                "return_type": "void",
                "signature": "calcKinematicsForElems ( const Real_t *__restrict__ xd, const Real_t *__restrict__ yd, const Real_t *__restrict__ zd, const Real_t *__restrict__ x, const Real_t *__restrict__ y, const Real_t *__restrict__ z, const Index_t *__restrict__ nodeList, const Real_t *__restrict__ volo, const Real_t *__restrict__ v, Real_t *__restrict__ delv, Real_t *__restrict__ arealg, Real_t *__restrict__ dxx, Real_t *__restrict__ dyy, Real_t *__restrict__ dzz, Real_t *__restrict__ vnew, const Real_t deltaTime, const Index_t numElem )",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 28,
                "name": "calcStrainRates",
                "offset": 1327,
                "qualifiers": [
                    "__global__"
                ],
                "return_type": "void",
                "signature": "calcStrainRates( Real_t *__restrict__ dxx, Real_t *__restrict__ dyy, Real_t *__restrict__ dzz, const Real_t *__restrict__ vnew, Real_t *__restrict__ vdov, int *__restrict__ vol_error, const Index_t numElem )",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 157,
                "name": "calcMonotonicQGradientsForElems",
                "offset": 1356,
                "qualifiers": [
                    "__global__"
                ],
                "return_type": "void",
                "signature": "calcMonotonicQGradientsForElems ( const Real_t *__restrict__ xd, const Real_t *__restrict__ yd, const Real_t *__restrict__ zd, const Real_t *__restrict__ x, const Real_t *__restrict__ y, const Real_t *__restrict__ z, const Index_t *__restrict__ nodelist, const Real_t *__restrict__ volo, Real_t *__restrict__ delv_eta, Real_t *__restrict__ delx_eta, Real_t *__restrict__ delv_zeta, Real_t *__restrict__ delx_zeta, Real_t *__restrict__ delv_xi, Real_t *__restrict__ delx_xi, const Real_t *__restrict__ vnew, const Index_t numElem )",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 171,
                "name": "calcMonotonicQForElems",
                "offset": 1514,
                "qualifiers": [
                    "__global__"
                ],
                "return_type": "void",
                "signature": "calcMonotonicQForElems ( const Index_t *__restrict__ elemBC, const Real_t *__restrict__ elemMass, Real_t *__restrict__ ql, Real_t *__restrict__ qq, const Real_t *__restrict__ vdov, const Real_t *__restrict__ volo, const Real_t *__restrict__ delv_eta, const Real_t *__restrict__ delx_eta, const Real_t *__restrict__ delv_zeta, const Real_t *__restrict__ delx_zeta, const Real_t *__restrict__ delv_xi, const Real_t *__restrict__ delx_xi, const Index_t *__restrict__ lxim, const Index_t *__restrict__ lxip, const Index_t *__restrict__ lzetam, const Index_t *__restrict__ lzetap, const Index_t *__restrict__ letap, const Index_t *__restrict__ letam, const Real_t *__restrict__ vnew, const Real_t monoq_limiter_mult, const Real_t monoq_max_slope, const Real_t qlc_monoq, const Real_t qqc_monoq, const Index_t numElem )",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 226,
                "name": "applyMaterialPropertiesForElems",
                "offset": 1686,
                "qualifiers": [
                    "__global__"
                ],
                "return_type": "void",
                "signature": "applyMaterialPropertiesForElems( const Real_t *__restrict__ ql, const Real_t *__restrict__ qq, const Real_t *__restrict__ delv, const Index_t *__restrict__ elemRep, const Index_t *__restrict__ elemElem, Real_t *__restrict__ q, Real_t *__restrict__ p, Real_t *__restrict__ e, Real_t *__restrict__ ss, Real_t *__restrict__ v, Real_t *__restrict__ vnewc, const Real_t e_cut, const Real_t p_cut, const Real_t ss4o3, const Real_t q_cut, const Real_t v_cut, const Real_t eosvmax, const Real_t eosvmin, const Real_t pmin, const Real_t emin, const Real_t rho0, const Index_t numElem )",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 5,
                "name": "Allocate",
                "offset": 1919,
                "qualifiers": [],
                "return_type": "T*",
                "signature": "Allocate(size_t size)",
                "templates": [
                    "template <typename T>"
                ]
            },
            {
                "kind": "defnt",
                "lines": 8,
                "name": "Release",
                "offset": 1925,
                "qualifiers": [],
                "return_type": "void",
                "signature": "Release(T **ptr)",
                "templates": [
                    "template <typename T>"
                ]
            },
            {
                "kind": "defnt",
                "lines": 51,
                "name": "TimeIncrement",
                "offset": 1938,
                "qualifiers": [
                    "static",
                    "inline"
                ],
                "return_type": "void",
                "signature": "TimeIncrement(Domain& domain)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 64,
                "name": "CalcCourantConstraintForElems",
                "offset": 1996,
                "qualifiers": [
                    "static",
                    "inline"
                ],
                "return_type": "void",
                "signature": "CalcCourantConstraintForElems(Domain &domain, Index_t length, Index_t *regElemlist, Real_t qqc, Real_t& dtcourant)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 53,
                "name": "CalcHydroConstraintForElems",
                "offset": 2063,
                "qualifiers": [
                    "static",
                    "inline"
                ],
                "return_type": "void",
                "signature": "CalcHydroConstraintForElems(Domain &domain, Index_t length, Index_t *regElemlist, Real_t dvovmax, Real_t& dthydro)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 21,
                "name": "CalcTimeConstraintsForElems",
                "offset": 2119,
                "qualifiers": [
                    "static",
                    "inline"
                ],
                "return_type": "void",
                "signature": "CalcTimeConstraintsForElems(Domain& domain)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 980,
                "name": "main",
                "offset": 2145,
                "qualifiers": [],
                "return_type": "int",
                "signature": "main(int argc, char *argv[])",
                "templates": []
            }
        ]
    },
    {
        "file": "lulesh.h",
        "functions": [
            {
                "kind": "defnt",
                "lines": 5,
                "name": "CBRT",
                "offset": 28,
                "qualifiers": [
                    "inline"
                ],
                "return_type": "real4",
                "signature": "CBRT(real4 arg)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 6,
                "name": "CBRT",
                "offset": 28,
                "qualifiers": [
                    "inline"
                ],
                "return_type": "real8",
                "signature": "CBRT(real8 arg)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 7,
                "name": "CBRT",
                "offset": 28,
                "qualifiers": [
                    "inline"
                ],
                "return_type": "real10",
                "signature": "CBRT(real10 arg)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 9,
                "name": "FABS",
                "offset": 28,
                "qualifiers": [
                    "inline"
                ],
                "return_type": "real4",
                "signature": "FABS(real4 arg)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 10,
                "name": "FABS",
                "offset": 28,
                "qualifiers": [
                    "inline"
                ],
                "return_type": "real8",
                "signature": "FABS(real8 arg)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 11,
                "name": "FABS",
                "offset": 28,
                "qualifiers": [
                    "inline"
                ],
                "return_type": "real10",
                "signature": "FABS(real10 arg)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "SQRT",
                "offset": 28,
                "qualifiers": [
                    "inline"
                ],
                "return_type": "real4",
                "signature": "SQRT(real4 arg)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 2,
                "name": "SQRT",
                "offset": 28,
                "qualifiers": [
                    "inline"
                ],
                "return_type": "real8",
                "signature": "SQRT(real8 arg)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 3,
                "name": "SQRT",
                "offset": 28,
                "qualifiers": [
                    "inline"
                ],
                "return_type": "real10",
                "signature": "SQRT(real10 arg)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 20,
                "name": "Domain::AllocateNodePersistent",
                "offset": 115,
                "qualifiers": [],
                "return_type": "void",
                "signature": "AllocateNodePersistent(Int_t numNode)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 37,
                "name": "Domain::AllocateElemPersistent",
                "offset": 136,
                "qualifiers": [],
                "return_type": "void",
                "signature": "AllocateElemPersistent(Int_t numElem)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 12,
                "name": "Domain::AllocateGradients",
                "offset": 174,
                "qualifiers": [],
                "return_type": "void",
                "signature": "AllocateGradients(Int_t numElem, Int_t allElem)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 10,
                "name": "Domain::DeallocateGradients",
                "offset": 187,
                "qualifiers": [],
                "return_type": "void",
                "signature": "DeallocateGradients()",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 6,
                "name": "Domain::AllocateStrains",
                "offset": 198,
                "qualifiers": [],
                "return_type": "void",
                "signature": "AllocateStrains(Int_t numElem)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 6,
                "name": "Domain::DeallocateStrains",
                "offset": 205,
                "qualifiers": [],
                "return_type": "void",
                "signature": "DeallocateStrains()",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::x",
                "offset": 219,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "x(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::y",
                "offset": 220,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "y(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::z",
                "offset": 221,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "z(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::xd",
                "offset": 224,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "xd(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::yd",
                "offset": 225,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "yd(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::zd",
                "offset": 226,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "zd(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::xdd",
                "offset": 229,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "xdd(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::ydd",
                "offset": 230,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "ydd(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::zdd",
                "offset": 231,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "zdd(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::fx",
                "offset": 234,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "fx(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::fy",
                "offset": 235,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "fy(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::fz",
                "offset": 236,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "fz(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::nodalMass",
                "offset": 239,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "nodalMass(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::symmX",
                "offset": 242,
                "qualifiers": [],
                "return_type": "Index_t",
                "signature": "symmX(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::symmY",
                "offset": 243,
                "qualifiers": [],
                "return_type": "Index_t",
                "signature": "symmY(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::symmZ",
                "offset": 244,
                "qualifiers": [],
                "return_type": "Index_t",
                "signature": "symmZ(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::symmXempty",
                "offset": 245,
                "qualifiers": [],
                "return_type": "bool",
                "signature": "symmXempty()",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::symmYempty",
                "offset": 246,
                "qualifiers": [],
                "return_type": "bool",
                "signature": "symmYempty()",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::symmZempty",
                "offset": 247,
                "qualifiers": [],
                "return_type": "bool",
                "signature": "symmZempty()",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::regElemSize",
                "offset": 252,
                "qualifiers": [],
                "return_type": "Index_t &",
                "signature": "regElemSize(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::regNumList",
                "offset": 253,
                "qualifiers": [],
                "return_type": "Index_t &",
                "signature": "regNumList(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::regNumList",
                "offset": 254,
                "qualifiers": [],
                "return_type": "Index_t*",
                "signature": "regNumList()",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::regElemlist",
                "offset": 255,
                "qualifiers": [],
                "return_type": "Index_t*",
                "signature": "regElemlist(Int_t r)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::regElemlist",
                "offset": 256,
                "qualifiers": [],
                "return_type": "Index_t &",
                "signature": "regElemlist(Int_t r, Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::nodelist",
                "offset": 258,
                "qualifiers": [],
                "return_type": "Index_t*",
                "signature": "nodelist(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::lxim",
                "offset": 261,
                "qualifiers": [],
                "return_type": "Index_t &",
                "signature": "lxim(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::lxip",
                "offset": 262,
                "qualifiers": [],
                "return_type": "Index_t &",
                "signature": "lxip(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::letam",
                "offset": 263,
                "qualifiers": [],
                "return_type": "Index_t &",
                "signature": "letam(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::letap",
                "offset": 264,
                "qualifiers": [],
                "return_type": "Index_t &",
                "signature": "letap(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::lzetam",
                "offset": 265,
                "qualifiers": [],
                "return_type": "Index_t &",
                "signature": "lzetam(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::lzetap",
                "offset": 266,
                "qualifiers": [],
                "return_type": "Index_t &",
                "signature": "lzetap(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::elemBC",
                "offset": 269,
                "qualifiers": [],
                "return_type": "Int_t &",
                "signature": "elemBC(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::dxx",
                "offset": 272,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "dxx(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::dyy",
                "offset": 273,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "dyy(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::dzz",
                "offset": 274,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "dzz(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::delv_xi",
                "offset": 277,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "delv_xi(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::delv_eta",
                "offset": 278,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "delv_eta(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::delv_zeta",
                "offset": 279,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "delv_zeta(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::delx_xi",
                "offset": 282,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "delx_xi(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::delx_eta",
                "offset": 283,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "delx_eta(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::delx_zeta",
                "offset": 284,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "delx_zeta(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::e",
                "offset": 287,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "e(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::p",
                "offset": 290,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "p(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::q",
                "offset": 293,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "q(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::ql",
                "offset": 296,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "ql(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::qq",
                "offset": 298,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "qq(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::v",
                "offset": 301,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "v(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::delv",
                "offset": 302,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "delv(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::volo",
                "offset": 305,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "volo(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::vdov",
                "offset": 308,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "vdov(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::arealg",
                "offset": 311,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "arealg(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::ss",
                "offset": 314,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "ss(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::elemMass",
                "offset": 317,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "elemMass(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::elemRep",
                "offset": 320,
                "qualifiers": [],
                "return_type": "Index_t &",
                "signature": "elemRep(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::elemElem",
                "offset": 323,
                "qualifiers": [],
                "return_type": "Index_t &",
                "signature": "elemElem(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 2,
                "name": "Domain::nodeElemCount",
                "offset": 325,
                "qualifiers": [],
                "return_type": "Index_t",
                "signature": "nodeElemCount(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 2,
                "name": "Domain::nodeElemCornerList",
                "offset": 328,
                "qualifiers": [],
                "return_type": "Index_t*",
                "signature": "nodeElemCornerList(Index_t idx)",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::u_cut",
                "offset": 334,
                "qualifiers": [],
                "return_type": "Real_t",
                "signature": "u_cut() const",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::e_cut",
                "offset": 335,
                "qualifiers": [],
                "return_type": "Real_t",
                "signature": "e_cut() const",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::p_cut",
                "offset": 336,
                "qualifiers": [],
                "return_type": "Real_t",
                "signature": "p_cut() const",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::q_cut",
                "offset": 337,
                "qualifiers": [],
                "return_type": "Real_t",
                "signature": "q_cut() const",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::v_cut",
                "offset": 338,
                "qualifiers": [],
                "return_type": "Real_t",
                "signature": "v_cut() const",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::hgcoef",
                "offset": 341,
                "qualifiers": [],
                "return_type": "Real_t",
                "signature": "hgcoef() const",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::qstop",
                "offset": 342,
                "qualifiers": [],
                "return_type": "Real_t",
                "signature": "qstop() const",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::monoq_max_slope",
                "offset": 343,
                "qualifiers": [],
                "return_type": "Real_t",
                "signature": "monoq_max_slope() const",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::monoq_limiter_mult",
                "offset": 344,
                "qualifiers": [],
                "return_type": "Real_t",
                "signature": "monoq_limiter_mult() const",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::ss4o3",
                "offset": 345,
                "qualifiers": [],
                "return_type": "Real_t",
                "signature": "ss4o3() const",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::qlc_monoq",
                "offset": 346,
                "qualifiers": [],
                "return_type": "Real_t",
                "signature": "qlc_monoq() const",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::qqc_monoq",
                "offset": 347,
                "qualifiers": [],
                "return_type": "Real_t",
                "signature": "qqc_monoq() const",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::qqc",
                "offset": 348,
                "qualifiers": [],
                "return_type": "Real_t",
                "signature": "qqc() const",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::eosvmax",
                "offset": 350,
                "qualifiers": [],
                "return_type": "Real_t",
                "signature": "eosvmax() const",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::eosvmin",
                "offset": 351,
                "qualifiers": [],
                "return_type": "Real_t",
                "signature": "eosvmin() const",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::pmin",
                "offset": 352,
                "qualifiers": [],
                "return_type": "Real_t",
                "signature": "pmin() const",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::emin",
                "offset": 353,
                "qualifiers": [],
                "return_type": "Real_t",
                "signature": "emin() const",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::dvovmax",
                "offset": 354,
                "qualifiers": [],
                "return_type": "Real_t",
                "signature": "dvovmax() const",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::refdens",
                "offset": 355,
                "qualifiers": [],
                "return_type": "Real_t",
                "signature": "refdens() const",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::time",
                "offset": 358,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "time()",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::deltatime",
                "offset": 359,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "deltatime()",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::deltatimemultlb",
                "offset": 360,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "deltatimemultlb()",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::deltatimemultub",
                "offset": 361,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "deltatimemultub()",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::stoptime",
                "offset": 362,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "stoptime()",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::dtcourant",
                "offset": 363,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "dtcourant()",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::dthydro",
                "offset": 364,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "dthydro()",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::dtmax",
                "offset": 365,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "dtmax()",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::dtfixed",
                "offset": 366,
                "qualifiers": [],
                "return_type": "Real_t &",
                "signature": "dtfixed()",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::cycle",
                "offset": 368,
                "qualifiers": [],
                "return_type": "Int_t &",
                "signature": "cycle()",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::numRanks",
                "offset": 369,
                "qualifiers": [],
                "return_type": "Index_t &",
                "signature": "numRanks()",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::colLoc",
                "offset": 371,
                "qualifiers": [],
                "return_type": "Index_t &",
                "signature": "colLoc()",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::rowLoc",
                "offset": 372,
                "qualifiers": [],
                "return_type": "Index_t &",
                "signature": "rowLoc()",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::planeLoc",
                "offset": 373,
                "qualifiers": [],
                "return_type": "Index_t &",
                "signature": "planeLoc()",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::tp",
                "offset": 374,
                "qualifiers": [],
                "return_type": "Index_t &",
                "signature": "tp()",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::sizeX",
                "offset": 376,
                "qualifiers": [],
                "return_type": "Index_t &",
                "signature": "sizeX()",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::sizeY",
                "offset": 377,
                "qualifiers": [],
                "return_type": "Index_t &",
                "signature": "sizeY()",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::sizeZ",
                "offset": 378,
                "qualifiers": [],
                "return_type": "Index_t &",
                "signature": "sizeZ()",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::numReg",
                "offset": 379,
                "qualifiers": [],
                "return_type": "Index_t &",
                "signature": "numReg()",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::cost",
                "offset": 380,
                "qualifiers": [],
                "return_type": "Int_t &",
                "signature": "cost()",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::numElem",
                "offset": 381,
                "qualifiers": [],
                "return_type": "Index_t &",
                "signature": "numElem()",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::numNode",
                "offset": 382,
                "qualifiers": [],
                "return_type": "Index_t &",
                "signature": "numNode()",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::maxPlaneSize",
                "offset": 384,
                "qualifiers": [],
                "return_type": "Index_t &",
                "signature": "maxPlaneSize()",
                "templates": []
            },
            {
                "kind": "defnt",
                "lines": 1,
                "name": "Domain::maxEdgeSize",
                "offset": 385,
                "qualifiers": [],
                "return_type": "Index_t &",
                "signature": "maxEdgeSize()",
                "templates": []
            },
            {
                "kind": "decl",
                "lines": 1,
                "name": "Domain::BuildMesh",
                "offset": 389,
                "qualifiers": [],
                "return_type": "void",
                "signature": "BuildMesh(Int_t nx, Int_t edgeNodes, Int_t edgeElems)",
                "templates": []
            },
            {
                "kind": "decl",
                "lines": 1,
                "name": "Domain::SetupThreadSupportStructures",
                "offset": 390,
                "qualifiers": [],
                "return_type": "void",
                "signature": "SetupThreadSupportStructures()",
                "templates": []
            },
            {
                "kind": "decl",
                "lines": 1,
                "name": "Domain::CreateRegionIndexSets",
                "offset": 391,
                "qualifiers": [],
                "return_type": "void",
                "signature": "CreateRegionIndexSets(Int_t nreg, Int_t balance)",
                "templates": []
            },
            {
                "kind": "decl",
                "lines": 1,
                "name": "Domain::SetupCommBuffers",
                "offset": 392,
                "qualifiers": [],
                "return_type": "void",
                "signature": "SetupCommBuffers(Int_t edgeNodes)",
                "templates": []
            },
            {
                "kind": "decl",
                "lines": 1,
                "name": "Domain::SetupSymmetryPlanes",
                "offset": 393,
                "qualifiers": [],
                "return_type": "void",
                "signature": "SetupSymmetryPlanes(Int_t edgeNodes)",
                "templates": []
            },
            {
                "kind": "decl",
                "lines": 1,
                "name": "Domain::SetupElementConnectivities",
                "offset": 394,
                "qualifiers": [],
                "return_type": "void",
                "signature": "SetupElementConnectivities(Int_t edgeElems)",
                "templates": []
            },
            {
                "kind": "decl",
                "lines": 1,
                "name": "Domain::SetupBoundaryConditions",
                "offset": 395,
                "qualifiers": [],
                "return_type": "void",
                "signature": "SetupBoundaryConditions(Int_t edgeElems)",
                "templates": []
            }
        ]
    }
]
'''




EXPECTED_TEMPLATED_FUNCTION_DEFINITIONS_JSON = '''
[
    {
        "file": "lulesh.cu",
        "functions": [
            {
                "kind": "defnt",
                "lines": 5,
                "name": "Allocate",
                "offset": 1919,
                "qualifiers": [],
                "return_type": "T*",
                "signature": "Allocate(size_t size)",
                "templates": [
                    "template <typename T>"
                ]
            },
            {
                "kind": "defnt",
                "lines": 8,
                "name": "Release",
                "offset": 1925,
                "qualifiers": [],
                "return_type": "void",
                "signature": "Release(T **ptr)",
                "templates": [
                    "template <typename T>"
                ]
            }
        ]
    }
]
'''



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
    {"file": "lulesh.cu", "kernel": "fill_sig", "line": 686, "offset": 685, "lines": 13},
    {"file": "lulesh.cu", "kernel": "integrateStress", "line": 699, "offset": 698, "lines": 74},
    {"file": "lulesh.cu", "kernel": "acc_final_force", "line": 773, "offset": 772, "lines": 31},
    {"file": "lulesh.cu", "kernel": "hgc", "line": 804, "offset": 803, "lines": 87},
    {"file": "lulesh.cu", "kernel": "fb", "line": 891, "offset": 890, "lines": 207},
    {"file": "lulesh.cu", "kernel": "collect_final_force", "line": 1098, "offset": 1097, "lines": 31},
    {"file": "lulesh.cu", "kernel": "accelerationForNode", "line": 1129, "offset": 1128, "lines": 18},
    {"file": "lulesh.cu", "kernel": "applyAccelerationBoundaryConditionsForNodes", "line": 1147, "offset": 1146, "lines": 20},
    {"file": "lulesh.cu", "kernel": "calcVelocityForNodes", "line": 1167, "offset": 1166, "lines": 30},
    {"file": "lulesh.cu", "kernel": "calcPositionForNodes", "line": 1197, "offset": 1196, "lines": 17},
    {"file": "lulesh.cu", "kernel": "calcKinematicsForElems", "line": 1214, "offset": 1213, "lines": 113},
    {"file": "lulesh.cu", "kernel": "calcStrainRates", "line": 1327, "offset": 1326, "lines": 29},
    {"file": "lulesh.cu", "kernel": "calcMonotonicQGradientsForElems", "line": 1356, "offset": 1355, "lines": 158},
    {"file": "lulesh.cu", "kernel": "calcMonotonicQForElems", "line": 1514, "offset": 1513, "lines": 172},
    {"file": "lulesh.cu", "kernel": "applyMaterialPropertiesForElems", "line": 1686, "offset": 1685, "lines": 227}
]
