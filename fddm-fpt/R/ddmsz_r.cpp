/* .Call glue for core/ddm_sz.h.  Gradients come back as a 5 x n (or 7 x n) matrix, which is
   the layout the core already writes.  No Rcpp. */
#include <R.h>
#include <Rinternals.h>
#include "ddm_sz.h"

extern "C" SEXP R_ddm_sz(SEXP t, SEXP par) {        /* par: nu, eta, a, w, sz, tol */
    int n = LENGTH(t);
    double *p = REAL(par);
    SEXP g = PROTECT(Rf_allocVector(REALSXP, n)), dg = PROTECT(Rf_allocMatrix(REALSXP, 5, n));
    if (ddm_sz_w(REAL(t), n, p[0], p[1], p[2], p[3], p[4], p[5], REAL(g), REAL(dg)))
        Rf_error("need a > 0, eta >= 0, 0 <= sz < 2 min(w, 1 - w), 0 < tol < 1");
    SEXP out = PROTECT(Rf_allocVector(VECSXP, 2));
    SET_VECTOR_ELT(out, 0, g);
    SET_VECTOR_ELT(out, 1, dg);
    UNPROTECT(3);
    return out;
}

extern "C" SEXP R_ddm_sz7(SEXP t, SEXP par) {       /* par: nu, eta, a, w, sz, t0, st0, tol */
    int n = LENGTH(t);
    double *p = REAL(par);
    SEXP f = PROTECT(Rf_allocVector(REALSXP, n)), df = PROTECT(Rf_allocMatrix(REALSXP, 7, n));
    if (ddm_sz7_w(REAL(t), n, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], REAL(f), REAL(df)))
        Rf_error("need a > 0, eta >= 0, 0 <= sz < 2 min(w, 1 - w), st0 >= 0, 0 < tol < 1");
    SEXP out = PROTECT(Rf_allocVector(VECSXP, 2));
    SET_VECTOR_ELT(out, 0, f);
    SET_VECTOR_ELT(out, 1, df);
    UNPROTECT(3);
    return out;
}
