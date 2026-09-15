"""Vectorized implementation of Result 1: lower-barrier RT density with
normal drift N(nu, eta^2) and uniform relative starting point on [w1, w2].
Unit diffusion coefficient. t may be an array."""
import numpy as np
from scipy.special import log_ndtr

SQ2PI = np.sqrt(2*np.pi)

def _logPhi_diff(x1, x2):
    """log(Phi(x2)-Phi(x1)), x2>x1, elementwise stable."""
    both_pos = x1 >= 0
    both_neg = x2 <= 0
    with np.errstate(divide="ignore", invalid="ignore"):
        up  = log_ndtr(-x1) + np.log1p(-np.exp(log_ndtr(-x2) - log_ndtr(-x1)))
        lo  = log_ndtr(x2)  + np.log1p(-np.exp(log_ndtr(x1)  - log_ndtr(x2)))
        mid = np.log(np.maximum(np.exp(log_ndtr(x2)) - np.exp(log_ndtr(x1)), 1e-300))
    return np.where(both_pos, up, np.where(both_neg, lo, mid))

PARAMS = ("nu", "eta", "a", "w1", "w2")

def _small(t, nu, eta, a, w1, w2, tol=1e-12):
    """Small-time series: returns (g, dg) with g shape (T,), dg shape (5,T) in PARAMS order."""
    t = np.atleast_1d(np.asarray(t, float))[:, None]          # (T,1)
    J = max(2, int(np.ceil(np.sqrt(2 * np.max(t) * np.log(1 / tol)) / a)) + 1)   # first dropped term O(exp(-J^2 a^2/2t)) < tol
    j = np.arange(J)[None, :]                                  # (1,J)
    S = 1 + eta**2 * t
    A = a*a/(t*S)
    even = (j % 2 == 0)
    k = np.where(even, j, j+1)
    sj = np.where(even, -1.0, 1.0)
    p = k*a
    q = np.where(even, a, -a)
    B = -nu*a/S + sj*k*a*a/t
    C = -nu**2*t/(2*S) - (k*a)**2/(2*t)
    m = B/A
    G = SQ2PI/np.sqrt(A)*np.exp(C + B*B/(2*A) + _logPhi_diff((w1-m)*np.sqrt(A), (w2-m)*np.sqrt(A)))
    E1 = np.exp(C + B*w1 - A*w1*w1/2); E2 = np.exp(C + B*w2 - A*w2*w2/2)
    I = (p + q*m)*G - (q/A)*(E2 - E1)
    def dI(dA=0, dB=0, dC=0, dp=0, dq=0, dw1=0, dw2=0):
        """total differential of I under (dA, dB, dC, dp, dq, dw1, dw2)"""
        dm = (dB - m*dA)/A
        dK = dC + m*dB - m*m*dA/2
        dG = G*(dK - dA/(2*A)) + E2*((dw2 - dm) + (w2-m)*dA/(2*A)) - E1*((dw1 - dm) + (w1-m)*dA/(2*A))
        dE1 = E1*(dC + w1*dB - w1*w1*dA/2 + (B - A*w1)*dw1)
        dE2 = E2*(dC + w2*dB - w2*w2*dA/2 + (B - A*w2)*dw2)
        return (dp + dq*m + q*dm)*G + (p + q*m)*dG - (dq/A - q*dA/(A*A))*(E2 - E1) - (q/A)*(dE2 - dE1)
    dI_all = np.stack([
        dI(dB=-a/S, dC=-nu*t/S),                                                       # nu
        dI(dA=-2*a*a*eta/S**2, dB=2*nu*a*eta*t/S**2, dC=nu**2*eta*t*t/S**2),           # eta
        dI(dA=2*a/(t*S), dB=-nu/S + 2*sj*k*a/t, dC=-k*k*a/t, dp=k, dq=np.sign(q)),     # a
        dI(dw1=1.0),                                                                    # w1
        dI(dw2=1.0),                                                                    # w2
    ])                                                                                  # (5,T,J)
    sgn = (-1.0)**j
    P = 1/((w2-w1)*np.sqrt(2*np.pi*t[:, 0]**3*S[:, 0]))
    dlogP = np.array([0.0, 0.0, 0.0, 1/(w2-w1), -1/(w2-w1)])[:, None] + np.stack(
        [0*t[:, 0], -eta*t[:, 0]/S[:, 0], 0*t[:, 0], 0*t[:, 0], 0*t[:, 0]])
    g = P*np.sum(sgn*I, axis=1)
    return g, P*np.sum(sgn*dI_all, axis=2) + dlogP*g


# ---------------------------------------------------------------------------
# Large-time counterpart of Result 1 (for t/a^2 large), via the Faddeeva function
# ---------------------------------------------------------------------------
from scipy.special import wofz
from math import factorial

def _large(t, nu, eta, a, w1, w2, tol=1e-12):
    """Large-time series via Faddeeva w(z): returns (g, dg) as in _small.  Stable for large t/a^2."""
    t = np.atleast_1d(np.asarray(t, float))[:, None]
    K = max(1, int(np.ceil(a * np.sqrt(2 * np.log(1 / tol) / (np.pi**2 * np.min(t))))) + 1)   # first dropped term O(exp(-K^2 pi^2 t/2a^2)) < tol
    k = np.arange(1, K + 1)[None, :]
    S = 1 + eta**2 * t
    kap = eta**2 * a**2 / (2 * S)                 # >= 0, coefficient of w^2 (growing)
    mu = -nu * a / S + 1j * np.pi * k              # complex linear coefficient
    E1 = np.exp(kap * w1 * w1 + mu * w1); E2 = np.exp(kap * w2 * w2 + mu * w2)
    with np.errstate(all='ignore'):
        sk = np.sqrt(kap)
        z = lambda w: sk * w + mu / (2 * sk)       # Im z > 0  ->  wofz bounded
        Iw = (-1j) * np.sqrt(np.pi) / (2 * sk) * (E2 * wofz(z(w2)) - E1 * wofz(z(w1)))
        dmu = ((E2 - E1) - mu * Iw) / (2 * kap)                    # int w e^{...} dw
        dkap = ((w2 * E2 - w1 * E1) - Iw - mu * dmu) / (2 * kap)   # int w^2 e^{...} dw
    e1, e2 = np.exp(mu * w1), np.exp(mu * w2)
    P = [(e2 - e1) / mu]
    for n in range(1, 23):
        P.append(((w2**n * e2 - w1**n * e1) - n * P[-1]) / mu)
    T0, T1, T2 = (sum(kap**m / factorial(m) * P[2 * m + i] for m in range(11)) for i in range(3))
    small = kap < 0.1
    Iw, dmu, dkap = np.where(small, T0, Iw), np.where(small, T1, dmu), np.where(small, T2, dkap)
    damp = k * np.exp(-k**2 * np.pi**2 * t / (2 * a**2))
    P = np.pi / a**2 / np.sqrt(S[:, 0]) * np.exp(-nu**2 * t[:, 0] / (2 * S[:, 0])) / (w2 - w1)
    g = P * np.sum(damp * Iw.imag, axis=1)
    dIw = np.stack([
        dmu * (-a / S),                                            # nu
        dmu * (2 * nu * a * eta * t / S**2) + dkap * (eta * a * a / S**2),   # eta
        dmu * (-nu / S) + dkap * (eta**2 * a / S),                 # a
        -E1,                                                       # w1
        E2,                                                        # w2
    ])
    dlogP = np.stack([-nu * t[:, 0] / S[:, 0],
                      -eta * t[:, 0] / S[:, 0] + nu**2 * eta * t[:, 0]**2 / S[:, 0]**2,
                      -2 / a + 0 * t[:, 0],
                      1 / (w2 - w1) + 0 * t[:, 0],
                      -1 / (w2 - w1) + 0 * t[:, 0]])
    ddamp = np.zeros(dIw.shape); ddamp[2] = damp * k**2 * np.pi**2 * t / a**3
    dg = P * np.sum(damp * dIw.imag + ddamp * Iw.imag, axis=2) + dlogP * g
    return g, dg

def grad_full_sz(t, nu, eta, a, w1, w2, upper=False, tol=1e-12):
    """(g, dg) for the density with drift ~ N(nu, eta^2) and start ~ U(w1, w2).
    dg has shape (5, T), rows = d/d(nu, eta, a, w1, w2).  upper=True gives the upper-barrier
    density via (nu, w1, w2) -> (-nu, 1-w2, 1-w1)."""
    if not (a > 0 and eta >= 0 and 0 < w1 < w2 < 1):
        # ponytail: w1 == w2 (sw = 0) is Blurton et al. 2017's density, not implemented here
        raise ValueError("need a > 0, eta >= 0, 0 < w1 < w2 < 1")
    if upper:
        g, dg = grad_full_sz(t, -nu, eta, a, 1 - w2, 1 - w1, False, tol)
        return g, -dg[[0, 1, 2, 4, 3]] * np.array([1, -1, -1, 1, 1])[:, None]
    t = np.atleast_1d(np.asarray(t, float))
    pos = t > 0                                   # density and gradient are 0 for t <= 0
    small = pos & (t / a**2 <= 1.0)             # small-/large-time switch at t/a^2 = 1
    g, dg = np.zeros_like(t), np.zeros((5, t.size))
    if small.any():
        g[small], dg[:, small] = _small(t[small], nu, eta, a, w1, w2, tol)
    large = pos & ~small
    if large.any():
        g[large], dg[:, large] = _large(t[large], nu, eta, a, w1, w2, tol)
    return g, dg

def g_full_sz(t, nu, eta, a, w1, w2, **kw):
    """Result 1 with automatic small-/large-time switch on t/a^2."""
    return grad_full_sz(t, nu, eta, a, w1, w2, **kw)[0]


# ---------------------------------------------------------------------------
# Seven-parameter density (sv, sw, st0): closed form in v and w, one 1-D
# quadrature over the non-decision-time window with an edge-clustering map.
# WienR convention: t0 ~ U(t0, t0 + st0).
# ---------------------------------------------------------------------------
_X, _W = np.polynomial.legendre.leggauss(32)   # fewer nodes fail near u = 0 (Table 1 of the manuscript)
PARAMS7 = PARAMS + ("t0", "st0")

def grad_f7(t, nu, eta, a, w1, w2, t0, st0, **kw):
    """(f, df) for one RT t; df has shape (7,), rows = d/d(nu, eta, a, w1, w2, t0, st0)."""
    if st0 < 0:
        raise ValueError("need st0 >= 0")
    hi = t - t0
    if hi <= 0:
        return 0.0, np.zeros(7)
    if st0 == 0:
        raise ValueError("st0 = 0: use grad_full_sz(t - t0, ...); d/dt0 = -dg/dt is not implemented")
    lo = max(hi - st0, 0.0)
    s = (_X + 1) / 2
    u = lo + (hi - lo) * s**3
    jac = (hi - lo) * 3 * s**2 / 2
    g, dg = grad_full_sz(np.concatenate([u, [hi, lo]]), nu, eta, a, w1, w2, **kw)
    g_hi, g_lo = g[-2], g[-1]                    # g(0) = 0 handles the clamped case
    f = np.sum(_W * jac * g[:-2]) / st0
    df = np.concatenate([np.sum(_W * jac * dg[:, :-2], axis=1) / st0,
                         [(g_lo - g_hi) / st0, (g_lo - f) / st0]])
    return float(f), df

def f7(t, nu, eta, a, w1, w2, t0, st0, **kw):
    return grad_f7(t, nu, eta, a, w1, w2, t0, st0, **kw)[0]
