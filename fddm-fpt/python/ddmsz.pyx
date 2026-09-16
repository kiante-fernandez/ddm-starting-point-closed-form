# distutils: language = c++
# distutils: sources = ../core/ddm_sz.cpp ../core/Faddeeva.cc
# distutils: include_dirs = ../core
"""Python wrapper for core/ddm_sz.h.  Build: cythonize -3 -i ddmsz.pyx (from python/)"""
import numpy as np

cdef extern from "ddm_sz.h":
    int ddm_sz_w(const double *t, int n, double nu, double eta, double a, double w, double sz,
                 double tol, double *g, double *dg) nogil
    int ddm_sz7_w(const double *t, int n, double nu, double eta, double a, double w, double sz,
                  double t0, double st0, double tol, double *f, double *df) nogil

BAD = "need a > 0, sv >= 0, 0 <= sz < 2 min(z, 1 - z), st >= 0, 0 < tol < 1"

def density(t, double nu, double eta, double a, double w, double sz=0.0, double tol=1e-12):
    """Density and d/d(nu, eta, a, w, sz); shapes (n,) and (n, 5).  Start ~ U(w - sz/2, w + sz/2)."""
    cdef double[::1] tt = np.ascontiguousarray(t, float).ravel()
    g = np.empty(tt.shape[0]); dg = np.empty((tt.shape[0], 5))
    cdef double[::1] gv = g
    cdef double[:, ::1] dv = dg
    if tt.shape[0] and ddm_sz_w(&tt[0], tt.shape[0], nu, eta, a, w, sz, tol, &gv[0], &dv[0, 0]):
        raise ValueError(BAD)
    return g, dg

def density7(t, double nu, double eta, double a, double w, double sz,
             double t0, double st0, double tol=1e-12):
    """Adds t0 ~ U(t0, t0 + st0).  Density and d/d(nu, eta, a, w, sz, t0, st0); (n,) and (n, 7)."""
    cdef double[::1] tt = np.ascontiguousarray(t, float).ravel()
    f = np.empty(tt.shape[0]); df = np.empty((tt.shape[0], 7))
    cdef double[::1] fv = f
    cdef double[:, ::1] dv = df
    if tt.shape[0] and ddm_sz7_w(&tt[0], tt.shape[0], nu, eta, a, w, sz, t0, st0, tol, &fv[0], &dv[0, 0]):
        raise ValueError(BAD)
    return f, df

PARAMS = ("v", "a", "z", "t", "sz", "sv", "st")

def full_ddm(rt, response, double v, double a, double z, double t,
             double sz=0.0, double sv=0.0, double st=0.0, double tol=1e-12):
    """full_ddm as HSSM parametrizes it: barriers at +-a, z and sz relative, t the centre of the
    non-decision window, response > 0.5 the upper barrier.  Density (n,) and d/dPARAMS (n, 7)."""
    rt = np.ascontiguousarray(rt, float).ravel()
    up = np.ascontiguousarray(response, float).ravel() > 0.5
    f, df = np.empty(rt.size), np.empty((rt.size, 7))
    for hit in (False, True):
        m = up == hit
        if not m.any():
            continue
        s = -1.0 if hit else 1.0
        g, dg = density7(rt[m], s*v, sv, 2*a, (1 - z) if hit else z, sz, t - st/2, st, tol)
        f[m] = g
        df[m] = np.column_stack([s*dg[:, 0], 2*dg[:, 2], s*dg[:, 3], dg[:, 5],
                                 dg[:, 4], dg[:, 1], dg[:, 6] - dg[:, 5]/2])
    return f, df
