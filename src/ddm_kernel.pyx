# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compiled scalar kernel for Result 1 (density only): the same arithmetic as ddm_fast.py,
using scipy's own log_ndtr / ndtr / wofz.  Build:  cythonize -3 -i src/ddm_kernel.pyx"""
from libc.math cimport exp, log, sqrt, ceil, cos, sin, M_PI
from scipy.special.cython_special cimport erfcx, ndtr, wofz
import numpy as np

cdef double SQ2PI = sqrt(2 * M_PI), SQRT1_2 = sqrt(0.5), TOL = 1e-12   # series tolerance, as ddm_fast's default

cdef inline double eK_dPhi(double x1, double x2, double K, double E1, double E2) noexcept nogil:
    """e^K [Phi(x2) - Phi(x1)] without forming e^K: uses e^K phi(x_i) sqrt(2 pi) = E_i and the
    Mills ratio Phi(-x) = phi(x) sqrt(pi/2) erfcx(x/sqrt 2), stable in both tails."""
    if x1 >= 0:
        return 0.5 * (E1 * erfcx(x1 * SQRT1_2) - E2 * erfcx(x2 * SQRT1_2))
    if x2 <= 0:
        return 0.5 * (E2 * erfcx(-x2 * SQRT1_2) - E1 * erfcx(-x1 * SQRT1_2))
    return exp(K) * (ndtr(x2) - ndtr(x1))      # centre inside [w1, w2]: K is the peak log value

cdef double g_small(double t, double nu, double eta, double a, double w1, double w2, double tol) noexcept nogil:
    cdef double S = 1 + eta*eta*t, A = a*a/(t*S), sA = sqrt(A)
    cdef int J = <int>ceil(sqrt(2*t*log(1/tol))/a) + 1
    cdef int j, k
    cdef double sj, p, q, B, C, m, G, E1, E2, tot = 0
    if J < 2: J = 2
    for j in range(J):
        if j % 2 == 0: k = j;     sj = -1; q = a
        else:          k = j + 1; sj = 1;  q = -a
        p = k*a
        B = -nu*a/S + sj*k*a*a/t
        C = -nu*nu*t/(2*S) - (k*a)*(k*a)/(2*t)
        m = B/A
        E1 = exp(C + B*w1 - A*w1*w1/2); E2 = exp(C + B*w2 - A*w2*w2/2)
        G = SQ2PI/sA*eK_dPhi((w1-m)*sA, (w2-m)*sA, C + B*B/(2*A), E1, E2)
        tot += (1 - 2*(j % 2)) * ((p + q*m)*G - (q/A)*(E2 - E1))
    return tot/((w2-w1)*sqrt(2*M_PI*t*t*t*S))

cdef double g_large(double t, double nu, double eta, double a, double w1, double w2, double tol) noexcept nogil:
    cdef double S = 1 + eta*eta*t, kap = eta*eta*a*a/(2*S), lam = -nu*a/S, sk, series = 0
    cdef int K = <int>ceil(a*sqrt(2*log(1/tol)/(M_PI*M_PI*t)))
    cdef int k
    cdef double complex mu, E1, E2, Iw
    for k in range(1, K + 1):
        mu = lam + 1j*M_PI*k
        E1 = exp(kap*w1*w1 + lam*w1) * (cos(M_PI*k*w1) + 1j*sin(M_PI*k*w1))
        E2 = exp(kap*w2*w2 + lam*w2) * (cos(M_PI*k*w2) + 1j*sin(M_PI*k*w2))
        if kap > 0:
            sk = sqrt(kap)
            Iw = -1j*sqrt(M_PI)/(2*sk) * (E2*wofz(sk*w2 + mu/(2*sk)) - E1*wofz(sk*w1 + mu/(2*sk)))
        else:
            Iw = (E2 - E1)/mu
        series += k*exp(-k*k*M_PI*M_PI*t/(2*a*a)) * Iw.imag
    return M_PI/(a*a)/sqrt(S)*exp(-nu*nu*t/(2*S))*series/(w2-w1)

cdef inline double g1(double t, double nu, double eta, double a, double w1, double w2, double tol) noexcept nogil:
    if t <= 0: return 0
    if t/(a*a) <= 1: return g_small(t, nu, eta, a, w1, w2, tol)
    return g_large(t, nu, eta, a, w1, w2, tol)

def g_full_sz(double[::1] t, double nu, double eta, double a, double w1, double w2):
    """Six-parameter density at each t (one parameter set).  Same values as ddm_fast.g_full_sz."""
    out = np.empty(t.shape[0])
    cdef double[::1] o = out
    cdef Py_ssize_t i
    for i in range(t.shape[0]): o[i] = g1(t[i], nu, eta, a, w1, w2, TOL)
    return out

_X, _W = np.polynomial.legendre.leggauss(32)
cdef double[::1] GX = _X, GW = _W

def f7(double[::1] t, double nu, double eta, double a, double w1, double w2, double t0, double st0):
    """Seven-parameter density at each t (one parameter set).  Same values as ddm_fast.f7."""
    out = np.empty(t.shape[0])
    cdef double[::1] o = out
    cdef Py_ssize_t i, n
    cdef double hi, lo, s, acc
    for i in range(t.shape[0]):
        hi = t[i] - t0
        if hi <= 0: o[i] = 0; continue
        if st0 == 0: o[i] = g1(hi, nu, eta, a, w1, w2, TOL); continue
        lo = hi - st0 if hi > st0 else 0
        acc = 0
        for n in range(32):
            s = (GX[n] + 1)/2
            acc += GW[n] * (hi-lo)*3*s*s/2 * g1(lo + (hi-lo)*s*s*s, nu, eta, a, w1, w2, TOL)
        o[i] = acc/st0
    return out
