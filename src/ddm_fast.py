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

def _coef(t, nu, eta, a, tol):
    """Coefficient table (4) for t of shape (T,1): the j-th term is (p + q w) exp(-A w^2/2 + B w + C).
    dth holds the differentials of (A, B, C, p, q) with respect to nu, eta, a for the chain rule."""
    J = max(2, int(np.ceil(np.sqrt(2 * np.max(t) * np.log(1 / tol)) / a)) + 1)   # first dropped term O(exp(-J^2 a^2/2t)) < tol
    j = np.arange(J)[None, :]                                  # (1,J)
    S = 1 + eta**2 * t
    A = a*a/(t*S)
    even = (j % 2 == 0)
    p = np.where(even, j, j+1)*a
    q = np.where(even, a, -a)
    B = -nu*a/S - p*q/t
    C = -nu**2*t/(2*S) - p*p/(2*t)
    dth = [dict(dB=-a/S, dC=-nu*t/S),                                                       # nu
           dict(dA=-2*a*a*eta/S**2, dB=2*nu*a*eta*t/S**2, dC=nu**2*eta*t*t/S**2),           # eta
           dict(dA=2*a/(t*S), dB=-nu/S - 2*p*q/(a*t), dC=-p*p/(a*t), dp=p/a, dq=q/a)]       # a
    return j, S, A, p, q, B, C, dth

def _small(t, nu, eta, a, w1, w2, tol=1e-12):
    """Small-time series: returns (g, dg) with g shape (T,), dg shape (5,T) in PARAMS order."""
    t = np.atleast_1d(np.asarray(t, float))[:, None]          # (T,1)
    j, S, A, p, q, B, C, dth = _coef(t, nu, eta, a, tol)
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
    dI_all = np.stack([dI(**d) for d in dth] + [dI(dw1=1.0), dI(dw2=1.0)])                 # (5,T,J)
    sgn = (-1.0)**j
    P = 1/((w2-w1)*np.sqrt(2*np.pi*t[:, 0]**3*S[:, 0]))
    dlogP = np.array([0.0, 0.0, 0.0, 1/(w2-w1), -1/(w2-w1)])[:, None] + np.stack(
        [0*t[:, 0], -eta*t[:, 0]/S[:, 0], 0*t[:, 0], 0*t[:, 0], 0*t[:, 0]])
    g = P*np.sum(sgn*I, axis=1)
    return g, P*np.sum(sgn*dI_all, axis=2) + dlogP*g

def _small_pt(t, nu, eta, a, w, tol=1e-12):
    """Fixed-start small-time density (1) at w: (f, df/d(nu,eta,a), df/dw), shapes (T,), (3,T), (T,)."""
    t = np.atleast_1d(np.asarray(t, float))[:, None]
    j, S, A, p, q, B, C, dth = _coef(t, nu, eta, a, tol)
    E = np.exp(C + B*w - A*w*w/2); T = (p + q*w)*E
    dT = lambda dA=0, dB=0, dC=0, dp=0, dq=0: (dp + w*dq)*E + T*(dC + w*dB - w*w*dA/2)
    sgn = (-1.0)**j
    P = 1/np.sqrt(2*np.pi*t[:, 0]**3*S[:, 0])
    f = P*np.sum(sgn*T, axis=1)
    df = P*np.stack([np.sum(sgn*dT(**d), axis=1) for d in dth])
    df[1] -= eta*t[:, 0]/S[:, 0]*f
    return f, df, P*np.sum(sgn*(q + (p + q*w)*(B - A*w))*E, axis=1)


# ---------------------------------------------------------------------------
# Large-time counterpart of Result 1 (for t/a^2 large), via the Faddeeva function
# ---------------------------------------------------------------------------
from scipy.special import wofz
from math import factorial

def _lcoef(t, nu, eta, a, tol):
    """Large-time setup for t of shape (T,1): k-th term is damp_k Im exp(kap w^2 + mu_k w) times P."""
    K = max(1, int(np.ceil(a * np.sqrt(2 * np.log(1 / tol) / (np.pi**2 * np.min(t))))) + 1)   # first dropped term O(exp(-K^2 pi^2 t/2a^2)) < tol
    k = np.arange(1, K + 1)[None, :]
    S = 1 + eta**2 * t
    kap = eta**2 * a**2 / (2 * S)                 # >= 0, coefficient of w^2 (growing)
    mu = -nu * a / S + 1j * np.pi * k              # complex linear coefficient
    damp = k * np.exp(-k**2 * np.pi**2 * t / (2 * a**2))
    P = np.pi / a**2 / np.sqrt(S[:, 0]) * np.exp(-nu**2 * t[:, 0] / (2 * S[:, 0]))
    dlogP = np.stack([-nu * t[:, 0] / S[:, 0],
                      -eta * t[:, 0] / S[:, 0] + nu**2 * eta * t[:, 0]**2 / S[:, 0]**2,
                      -2 / a + 0 * t[:, 0]])
    return k, S, kap, mu, damp, P, dlogP

def _lchain(X1, X2, nu, eta, a, S, t):
    """rows nu, eta, a from the mu- and kappa-derivatives X1, X2 of a term"""
    return [X1 * (-a / S),
            X1 * (2 * nu * a * eta * t / S**2) + X2 * (eta * a * a / S**2),
            X1 * (-nu / S) + X2 * (eta**2 * a / S)]

def _large(t, nu, eta, a, w1, w2, tol=1e-12):
    """Large-time series via Faddeeva w(z): returns (g, dg) as in _small.  Stable for large t/a^2."""
    t = np.atleast_1d(np.asarray(t, float))[:, None]
    k, S, kap, mu, damp, P, dlogP = _lcoef(t, nu, eta, a, tol)
    E1 = np.exp(kap * w1 * w1 + mu * w1); E2 = np.exp(kap * w2 * w2 + mu * w2)
    with np.errstate(all='ignore'):
        sk = np.sqrt(kap)
        z = lambda w: sk * w + mu / (2 * sk)       # Im z > 0  ->  wofz bounded
        Iw = (-1j) * np.sqrt(np.pi) / (2 * sk) * (E2 * wofz(z(w2)) - E1 * wofz(z(w1)))
        dmu = ((E2 - E1) - mu * Iw) / (2 * kap)                    # int w e^{...} dw
        dkap = ((w2 * E2 - w1 * E1) - Iw - mu * dmu) / (2 * kap)   # int w^2 e^{...} dw
    e1, e2 = np.exp(mu * w1), np.exp(mu * w2)
    Pn = [(e2 - e1) / mu]
    for n in range(1, 23):
        Pn.append(((w2**n * e2 - w1**n * e1) - n * Pn[-1]) / mu)
    T0, T1, T2 = (sum(kap**m / factorial(m) * Pn[2 * m + i] for m in range(11)) for i in range(3))
    small = kap < 0.1
    Iw, dmu, dkap = np.where(small, T0, Iw), np.where(small, T1, dmu), np.where(small, T2, dkap)
    P = P / (w2 - w1)
    g = P * np.sum(damp * Iw.imag, axis=1)
    dIw = np.stack(_lchain(dmu, dkap, nu, eta, a, S, t) + [-E1, E2])                       # (5,T,K)
    dlogP = np.concatenate([dlogP, [1 / (w2 - w1) + 0 * t[:, 0], -1 / (w2 - w1) + 0 * t[:, 0]]])
    ddamp = np.zeros(dIw.shape); ddamp[2] = damp * k**2 * np.pi**2 * t / a**3
    dg = P * np.sum(damp * dIw.imag + ddamp * Iw.imag, axis=2) + dlogP * g
    return g, dg

def _large_pt(t, nu, eta, a, w, tol=1e-12):
    """Fixed-start large-time density (8) at w: (f, df/d(nu,eta,a), df/dw), shapes (T,), (3,T), (T,)."""
    t = np.atleast_1d(np.asarray(t, float))[:, None]
    k, S, kap, mu, damp, P, dlogP = _lcoef(t, nu, eta, a, tol)
    e = np.exp(kap * w * w + mu * w)
    f = P * np.sum(damp * e.imag, axis=1)
    df = P * np.stack([np.sum(damp * r.imag, axis=1) for r in _lchain(w * e, w * w * e, nu, eta, a, S, t)]) + dlogP * f
    df[2] += P * np.sum(damp * k**2 * np.pi**2 * t / a**3 * e.imag, axis=1)
    return f, df, P * np.sum(damp * ((2 * kap * w + mu) * e).imag, axis=1)

_X5, _W5 = np.polynomial.legendre.leggauss(5)

def _avg(pt, t, nu, eta, a, w1, w2, tol):
    """w2 - w1 < 1e-2: 5-node Gauss-Legendre average of the fixed-start density `pt` over [w1, w2].
    The closed form's F(w2) - F(w1) loses digits like 1/(w2 - w1); the average is exact to 1e-15
    below the threshold, sums instead of differences, and allows w1 = w2."""
    wb, D = (w1 + w2) / 2, w2 - w1
    g, dg = 0.0, 0.0
    for x, om in zip(_X5, _W5 / 2):
        f, d3, fw = pt(t, nu, eta, a, wb + D * x / 2, tol)
        g = g + om * f
        dg = dg + om * np.vstack([d3, fw * (1 - x) / 2, fw * (1 + x) / 2])   # d/dw1, d/dw2 via (wbar, Delta)
    return g, dg

def grad_full_sz(t, nu, eta, a, w1, w2, tol=1e-12):
    """(g, dg) for the lower-barrier density with drift ~ N(nu, eta^2) and start ~ U(w1, w2).
    dg has shape (5, T), rows = d/d(nu, eta, a, w1, w2).  Upper barrier: (nu, w1, w2) -> (-nu, 1-w2, 1-w1)."""
    if not (a > 0 and eta >= 0 and 0 < w1 <= w2 < 1):
        raise ValueError("need a > 0, eta >= 0, 0 < w1 <= w2 < 1")
    t = np.atleast_1d(np.asarray(t, float))
    pos = t > 0                                   # density and gradient are 0 for t <= 0
    small = pos & (t / a**2 <= 1.0)             # small-/large-time switch at t/a^2 = 1
    mid = w2 - w1 < 1e-2                        # small-Delta branch, see _avg
    g, dg = np.zeros_like(t), np.zeros((5, t.size))
    for mask, F, pt in ((small, _small, _small_pt), (pos & ~small, _large, _large_pt)):
        if mask.any():
            g[mask], dg[:, mask] = _avg(pt, t[mask], nu, eta, a, w1, w2, tol) if mid else F(t[mask], nu, eta, a, w1, w2, tol)
    return g, dg


# ---------------------------------------------------------------------------
# Seven-parameter density (sv, sw, st0): closed form in v and w, one 1-D
# quadrature over the non-decision-time window with an edge-clustering map.
# WienR convention: t0 ~ U(t0, t0 + st0).
# ---------------------------------------------------------------------------
_X, _W = np.polynomial.legendre.leggauss(48)   # 32 nodes lose digits for w1 < 0.2 (Table tab:nodes of the manuscript)
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
