solution = [
r"""__global__ void applyMaterialPropertiesForElems(
    const Real_t *__restrict__ ql,
    const Real_t *__restrict__ qq,
    const Real_t *__restrict__ delv,
    const Index_t *__restrict__ elemRep,
    const Index_t *__restrict__ elemElem,
    Real_t *__restrict__ q,
    Real_t *__restrict__ p,
    Real_t *__restrict__ e,
    Real_t *__restrict__ ss,
    Real_t *__restrict__ v,
    Real_t *__restrict__ vnewc,
    const Real_t  e_cut,
    const Real_t  p_cut,
    const Real_t  ss4o3,
    const Real_t  q_cut,
    const Real_t  v_cut,

    const Real_t eosvmax,
    const Real_t eosvmin,
    const Real_t pmin,
    const Real_t emin,
    const Real_t rho0,
    const Index_t numElem )
{
  Index_t elem = blockDim.x*blockIdx.x+threadIdx.x;
  if (elem >= numElem) return;
  Index_t rep = elemRep[elem];
  Real_t e_old, delvc, p_old, q_old, qq_old, ql_old;
  Real_t p_new, q_new, e_new;
  Real_t work, compression, compHalfStep, bvc, pbvc, pHalfStep;
  Real_t vchalf ;
  Real_t vhalf ;
  Real_t ssc ;
  Real_t q_tilde ;
  Real_t ssTmp ;

  if (eosvmin != ZERO) {
    if (vnewc[elem] < eosvmin)
      vnewc[elem] = eosvmin ;
  }

  if (eosvmax != ZERO) {
    if (vnewc[elem] > eosvmax)
      vnewc[elem] = eosvmax ;
  }

  // This check may not make perfect sense in LULESH, but
  // it's representative of something in the full code -
  // just leave it in, please
  Real_t vc = v[elem] ;
  if (eosvmin != ZERO) {
    if (vc < eosvmin)
      vc = eosvmin ;
  }
  if (eosvmax != ZERO) {
    if (vc > eosvmax)
      vc = eosvmax ;
  }

  Real_t vnewc_t = vnewc[elem];

  Real_t e_temp    =    e[elem];
  Real_t delv_temp = delv[elem];
  Real_t p_temp    =    p[elem];
  Real_t q_temp    =    q[elem];
  Real_t qq_temp   =   qq[elem];
  Real_t ql_temp   =   ql[elem];
  for(Index_t j = 0; j < rep; j++) {

    e_old  =    e_temp ;
    delvc  = delv_temp ;
    p_old  =    p_temp ;
    q_old  =    q_temp ;
    qq_old =   qq_temp ;
    ql_old =   ql_temp ;

    compression = ONE / vnewc_t - ONE;
    vchalf = vnewc_t - delvc * HALF;
    compHalfStep = ONE / vchalf - ONE;
    if (vnewc_t <= eosvmin) { /* impossible due to calling func? */
      compHalfStep = compression ;
    }
    if (vnewc_t >= eosvmax) { /* impossible due to calling func? */
      p_old        = ZERO ;
      compression  = ZERO ;
      compHalfStep = ZERO ;
    }
    work = ZERO ;

    e_new = e_old - HALF * delvc * (p_old + q_old)
      + HALF * work;

    if (e_new  < emin ) {
      e_new = emin ;
    }

    bvc = C1S * (compHalfStep + ONE);
    pbvc = C1S;

    pHalfStep = bvc * e_new ;

    if    (fabs(pHalfStep) <  p_cut   )
      pHalfStep = ZERO ;

    if    ( vnewc_t >= eosvmax ) /* impossible condition here? */
      pHalfStep = ZERO ;

    if    (pHalfStep      <  pmin)
      pHalfStep   = pmin ;

    vhalf = ONE / (ONE + compHalfStep) ;

    if ( delvc > ZERO ) {
      q_new /* = qq_old[elem] = ql_old[elem] */ = ZERO ;
    } else {
      ssc = ( pbvc * e_new + vhalf * vhalf * bvc * pHalfStep ) / rho0 ;

      if ( ssc <= C1 ) {
        ssc = C2 ;
      } else {
        ssc = sqrt(ssc) ;
      }

      q_new = (ssc*ql_old + qq_old) ;
    }

    e_new = e_new + HALF * delvc
      * (THREE*(p_old     + q_old)
          - FOUR*(pHalfStep + q_new)) ;

    e_new += HALF * work;

    if (fabs(e_new) < e_cut) {
      e_new = ZERO  ;
    }
    if (     e_new  < emin ) {
      e_new = emin ;
    }

    bvc = C1S * (compression + ONE);
    pbvc = C1S;

    p_new = bvc * e_new ;

    if    (fabs(p_new) <  p_cut   )
      p_new = ZERO ;

    if    ( vnewc_t >= eosvmax ) /* impossible condition here? */
      p_new = ZERO ;

    if    (p_new  <  pmin)
      p_new   = pmin ;


    if (delvc > ZERO) {
      q_tilde = ZERO ;
    }
    else {
      Real_t ssc = ( pbvc * e_new + vnewc_t * vnewc_t * bvc * p_new ) / rho0 ;

      if ( ssc <= C1 ) {
        ssc = C2 ;
      } else {
        ssc = sqrt(ssc) ;
      }

      q_tilde = (ssc * ql_old + qq_old) ;
    }

    e_new = e_new - (  SEVEN*(p_old     + q_old)
        - EIGHT*(pHalfStep + q_new)
        + (p_new + q_tilde)) * delvc*SIXTH ;

    if (fabs(e_new) < e_cut) {
      e_new = ZERO  ;
    }
    if (e_new < emin) {
      e_new = emin ;
    }

    bvc = C1S * (compression + ONE);
    pbvc = C1S;

    p_new = bvc * e_new ;

    if ( fabs(p_new) <  p_cut )
      p_new = ZERO ;

    if ( vnewc_t >= eosvmax ) /* impossible condition here? */
      p_new = ZERO ;

    if (p_new < pmin)
      p_new = pmin ;
    if ( delvc <= ZERO ) {
      ssc = ( pbvc * e_new + vnewc_t * vnewc_t * bvc * p_new ) / rho0 ;

      if ( ssc <= C1 ) {
        ssc = C2 ;
      } else {
        ssc = sqrt(ssc) ;
      }

      q_new = (ssc*ql_old + qq_old) ;

      if (fabs(q_new) < q_cut) q_new = ZERO ;
    }
  } //this is the end of the rep loop

  p[elem] = p_new ;
  e[elem] = e_new ;
  q[elem] = q_new ;

  ssTmp = (pbvc * e_new + vnewc_t * vnewc_t * bvc * p_new) / rho0;
  if (ssTmp <= C1) {
    ssTmp = C2;
  } else {
    ssTmp = sqrt(ssTmp);
  }
  ss[elem] = ssTmp ;

  if ( fabs(vnewc_t - ONE) < v_cut )
    vnewc_t = ONE ;

  v[elem] = vnewc_t ;
}""".strip()
]
