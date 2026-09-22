/* Result 1: lower-barrier first-passage density with drift ~ N(nu, eta^2) and relative
   starting point ~ U(w1, w2), unit diffusion.  Closed form plus its five derivatives;
   see the manuscript, https://doi.org/10.2139/ssrn.7507587.
   Upper barrier: (nu, w1, w2) -> (-nu, 1 - w2, 1 - w1). */
#ifndef DDM_SZ_H
#define DDM_SZ_H

/* g[i] = density at t[i]; dg[5*i + r] = d g / d(nu, eta, a, w1, w2)_r.
   tol sets the series truncation (1e-12 in the paper).  Returns 0, or 1 if the
   parameters are outside a > 0, eta >= 0, 0 < w1 <= w2 < 1, tol in (0, 1). */
int ddm_sz(const double *t, int n, double nu, double eta, double a, double w1, double w2,
           double tol, double *g, double *dg);

/* Seven-parameter density: non-decision time uniform on [t0, t0 + st0], one 48-node
   Gauss-Legendre rule over that window.  f[i] = density at t[i]; df[7*i + r] =
   d f / d(nu, eta, a, w1, w2, t0, st0)_r.  Returns 0, or 1 as above or if st0 < 0. */
int ddm_sz7(const double *t, int n, double nu, double eta, double a, double w1, double w2,
            double t0, double st0, double tol, double *f, double *df);

/* Same, in the (w, sz) parametrization other packages use: start ~ U(w - sz/2, w + sz/2),
   relative like WienR's (w, sw) and HDDM's (z, sz).  Gradient rows replace w1, w2 by w, sz.
   For absolute z = a w and absolute sz, divide both by a first. */
int ddm_sz_w(const double *t, int n, double nu, double eta, double a, double w, double sz,
             double tol, double *g, double *dg);
int ddm_sz7_w(const double *t, int n, double nu, double eta, double a, double w, double sz,
              double t0, double st0, double tol, double *f, double *df);

#endif
