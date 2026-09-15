"""
Scalar reference implementation of Result 1: lower-barrier first-passage density with
normal drift N(nu, eta^2) AND uniform starting point on [w1, w2], in closed form.
Unit diffusion coefficient; barriers at 0 and a; r_j = j a + a w (even j),
r_j = (j+1) a - a w (odd j).  Upper barrier: (v, w) -> (-v, 1-w).

Kept deliberately independent of ddm_fast.py (scalar loops, scipy.stats.norm) so that
agreement between the two is an independent-implementation check.
"""
import numpy as np
from scipy.stats import norm
from scipy import integrate

Phi, logPhi = norm.cdf, norm.logcdf
SQ2PI = np.sqrt(2 * np.pi)

# ----------------------------------------------------------------------------
# reference (constant drift, fixed w)
# ----------------------------------------------------------------------------
def r_j(j, a, w):
    return j * a + a * w if j % 2 == 0 else (j + 1) * a - a * w

def g_eta(t, nu, eta, a, w, J=40):
    """lower-barrier density with normal drift N(nu, eta^2)  (Horrocks & Thompson / Blurton Eq.1)"""
    if t <= 0:
        return 0.0
    S = 1 + eta ** 2 * t
    s = 0.0
    for j in range(J):
        r = r_j(j, a, w)
        s += (-1) ** j * r * np.exp(-r * r / (2 * t))
    pref = np.exp((-nu ** 2 * t - 2 * nu * a * w + eta ** 2 * (a * w) ** 2) / (2 * S))
    return pref * s / np.sqrt(2 * np.pi * t ** 3 * S)

# ----------------------------------------------------------------------------
# generic antiderivatives in w
# ----------------------------------------------------------------------------
def logPhi_diff(x1, x2):
    """log( Phi(x2) - Phi(x1) ), x2 > x1, stable in both tails."""
    if x1 >= 0:   # both positive: use upper tails
        return logPhi(-x1) + np.log1p(-np.exp(logPhi(-x2) - logPhi(-x1)))
    if x2 <= 0:   # both negative: use lower tails
        return logPhi(x2) + np.log1p(-np.exp(logPhi(x1) - logPhi(x2)))
    return np.log(Phi(x2) - Phi(x1))

def def_int_lin_gauss(p, q, A, B, C, w1, w2):
    """int_{w1}^{w2} (p + q w) exp(-(A/2) w^2 + B w + C) dw,  A > 0, stable."""
    m = B / A
    sig = 1 / np.sqrt(A)
    logK = C + B * B / (2 * A)
    t1 = (p + q * m) * SQ2PI * sig * np.exp(logK + logPhi_diff((w1 - m) / sig, (w2 - m) / sig))
    t2 = q * sig ** 2 * (np.exp(C + B * w2 - A * w2 * w2 / 2) - np.exp(C + B * w1 - A * w1 * w1 / 2))
    return t1 - t2

# ----------------------------------------------------------------------------
# Result 1: density with normal drift AND uniform starting point
# ----------------------------------------------------------------------------
def g_eta_sz(t, nu, eta, a, w1, w2, J=40):
    if t <= 0:
        return 0.0
    S = 1 + eta ** 2 * t
    A = a * a / (t * S)                      # always > 0
    tot = 0.0
    for j in range(J):
        if j % 2 == 0:
            B = -nu * a / S - j * a * a / t
            C = -nu ** 2 * t / (2 * S) - (j * a) ** 2 / (2 * t)
            p, q = j * a, a
        else:
            k = j + 1
            B = -nu * a / S + k * a * a / t
            C = -nu ** 2 * t / (2 * S) - (k * a) ** 2 / (2 * t)
            p, q = k * a, -a
        tot += (-1) ** j * def_int_lin_gauss(p, q, A, B, C, w1, w2)
    return tot / ((w2 - w1) * np.sqrt(2 * np.pi * t ** 3 * S))

# ----------------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------------
def check_all():
    """Result 1 vs numerical w-integration of the (known) eta-density, 40 random parameter sets."""
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(40):
        a = rng.uniform(0.6, 2.5)
        w0 = rng.uniform(0.3, 0.7)
        sw = rng.uniform(0.05, 0.4)
        w1, w2 = w0 - sw / 2, w0 + sw / 2
        t = rng.uniform(0.15, 2.5)
        nu, eta = rng.uniform(-2, 2), rng.uniform(0.3, 1.5)
        ref = integrate.quad(lambda w: g_eta(t, nu, eta, a, w), w1, w2, epsabs=1e-13)[0] / (w2 - w1)
        worst = max(worst, abs(g_eta_sz(t, nu, eta, a, w1, w2) - ref))
    return worst

if __name__ == "__main__":
    print("max abs error vs quadrature over 40 random parameter sets:", check_all())
