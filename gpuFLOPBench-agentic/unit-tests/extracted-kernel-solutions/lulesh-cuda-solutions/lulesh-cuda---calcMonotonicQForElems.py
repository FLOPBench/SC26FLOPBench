solution = [
r"""__global__ void calcMonotonicQForElems (
    const Index_t *__restrict__ elemBC,
    const Real_t *__restrict__ elemMass,
    Real_t *__restrict__ ql,
    Real_t *__restrict__ qq,
    const Real_t *__restrict__ vdov,
    const Real_t *__restrict__ volo,
    const Real_t *__restrict__ delv_eta,
    const Real_t *__restrict__ delx_eta,
    const Real_t *__restrict__ delv_zeta,
    const Real_t *__restrict__ delx_zeta,
    const Real_t *__restrict__ delv_xi,
    const Real_t *__restrict__ delx_xi,
    const Index_t *__restrict__ lxim,
    const Index_t *__restrict__ lxip,
    const Index_t *__restrict__ lzetam,
    const Index_t *__restrict__ lzetap,
    const Index_t *__restrict__ letap,
    const Index_t *__restrict__ letam,
    const Real_t *__restrict__ vnew,
    const Real_t monoq_limiter_mult,
    const Real_t monoq_max_slope,
    const Real_t qlc_monoq,
    const Real_t qqc_monoq,
    const Index_t numElem )
{
  Index_t i = blockDim.x*blockIdx.x+threadIdx.x;
  if (i >= numElem) return;

  Real_t qlin, qquad ;
  Real_t phixi, phieta, phizeta ;
  Int_t bcMask = elemBC[i] ;
  Real_t delvm = 0.0, delvp =0.0;

  /*  phixi     */
  Real_t norm = Real_t(1.) / (delv_xi[i]+ PTINY ) ;

  switch (bcMask & XI_M) {
    case XI_M_COMM: /* needs comm data */
    case 0:         delvm = delv_xi[lxim[i]]; break ;
    case XI_M_SYMM: delvm = delv_xi[i] ;       break ;
    case XI_M_FREE: delvm = Real_t(0.0) ;      break ;
    default: //fprintf(stderr, "Error in switch at %s line %d\n", __FILE__, __LINE__);
        delvm = 0; /* ERROR - but quiets the compiler */
        break;
  }
  switch (bcMask & XI_P) {
    case XI_P_COMM: /* needs comm data */
    case 0:         delvp = delv_xi[lxip[i]] ; break ;
    case XI_P_SYMM: delvp = delv_xi[i] ;       break ;
    case XI_P_FREE: delvp = Real_t(0.0) ;      break ;
    default: //fprintf(stderr, "Error in switch at %s line %d\n", __FILE__, __LINE__);
        delvp = 0; /* ERROR - but quiets the compiler */
        break;
  }

  delvm = delvm * norm ;
  delvp = delvp * norm ;

  phixi = Real_t(.5) * ( delvm + delvp ) ;

  delvm *= monoq_limiter_mult ;
  delvp *= monoq_limiter_mult ;

  if ( delvm < phixi ) phixi = delvm ;
  if ( delvp < phixi ) phixi = delvp ;
  if ( phixi < Real_t(0.)) phixi = Real_t(0.) ;
  if ( phixi > monoq_max_slope) phixi = monoq_max_slope;


  /*  phieta     */
  norm = Real_t(1.) / ( delv_eta[i] + PTINY ) ;

  switch (bcMask & ETA_M) {
    case ETA_M_COMM: /* needs comm data */
    case 0:          delvm = delv_eta[letam[i]] ; break ;
    case ETA_M_SYMM: delvm = delv_eta[i] ;        break ;
    case ETA_M_FREE: delvm = Real_t(0.0) ;        break ;
    default: //fprintf(stderr, "Error in switch at %s line %d\n", __FILE__, __LINE__);
         delvm = 0; /* ERROR - but quiets the compiler */
         break;
  }
  switch (bcMask & ETA_P) {
    case ETA_P_COMM: /* needs comm data */
    case 0:          delvp = delv_eta[letap[i]] ; break ;
    case ETA_P_SYMM: delvp = delv_eta[i] ;        break ;
    case ETA_P_FREE: delvp = Real_t(0.0) ;        break ;
    default: 
         delvp = 0; /* ERROR - but quiets the compiler */
         break;
  }

  delvm = delvm * norm ;
  delvp = delvp * norm ;

  phieta = Real_t(.5) * ( delvm + delvp ) ;

  delvm *= monoq_limiter_mult ;
  delvp *= monoq_limiter_mult ;

  if ( delvm  < phieta ) phieta = delvm ;
  if ( delvp  < phieta ) phieta = delvp ;
  if ( phieta < Real_t(0.)) phieta = Real_t(0.) ;
  if ( phieta > monoq_max_slope)  phieta = monoq_max_slope;

  /*  phizeta     */
  norm = Real_t(1.) / ( delv_zeta[i] + PTINY ) ;

  switch (bcMask & ZETA_M) {
    case ZETA_M_COMM: /* needs comm data */
    case 0:           delvm = delv_zeta[lzetam[i]] ; break ;
    case ZETA_M_SYMM: delvm = delv_zeta[i] ;         break ;
    case ZETA_M_FREE: delvm = Real_t(0.0) ;          break ;
    default: 
          delvm = 0; /* ERROR - but quiets the compiler */
          break;
  }
  switch (bcMask & ZETA_P) {
    case ZETA_P_COMM: /* needs comm data */
    case 0:           delvp = delv_zeta[lzetap[i]] ; break ;
    case ZETA_P_SYMM: delvp = delv_zeta[i] ;         break ;
    case ZETA_P_FREE: delvp = Real_t(0.0) ;          break ;
    default:
          delvp = 0; /* ERROR - but quiets the compiler */
          break;
  }

  delvm = delvm * norm ;
  delvp = delvp * norm ;

  phizeta = Real_t(.5) * ( delvm + delvp ) ;

  delvm *= monoq_limiter_mult ;
  delvp *= monoq_limiter_mult ;

  if ( delvm   < phizeta ) phizeta = delvm ;
  if ( delvp   < phizeta ) phizeta = delvp ;
  if ( phizeta < Real_t(0.)) phizeta = Real_t(0.);
  if ( phizeta > monoq_max_slope  ) phizeta = monoq_max_slope;

  /* Remove length scale */

  if ( vdov[i] > Real_t(0.) )  {
    qlin  = Real_t(0.) ;
    qquad = Real_t(0.) ;
  }
  else {
    Real_t delvxxi   = delv_xi[i]   * delx_xi[i]   ;
    Real_t delvxeta  = delv_eta[i]  * delx_eta[i]  ;
    Real_t delvxzeta = delv_zeta[i] * delx_zeta[i] ;

    if ( delvxxi   > Real_t(0.) ) delvxxi   = Real_t(0.) ;
    if ( delvxeta  > Real_t(0.) ) delvxeta  = Real_t(0.) ;
    if ( delvxzeta > Real_t(0.) ) delvxzeta = Real_t(0.) ;

    Real_t rho = elemMass[i] / (volo[i] * vnew[i]) ;

    qlin = -qlc_monoq * rho *
      (  delvxxi   * (Real_t(1.) - phixi) +
         delvxeta  * (Real_t(1.) - phieta) +
         delvxzeta * (Real_t(1.) - phizeta)  ) ;

    qquad = qqc_monoq * rho *
      (  delvxxi*delvxxi     * (Real_t(1.) - phixi*phixi) +
         delvxeta*delvxeta   * (Real_t(1.) - phieta*phieta) +
         delvxzeta*delvxzeta * (Real_t(1.) - phizeta*phizeta)  ) ;
  }

  qq[i] = qquad ;
  ql[i] = qlin  ;
}""".strip()
]
