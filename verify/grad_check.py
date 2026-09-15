"""Gradient of Result 1 in (nu, eta, a, w1, w2): closed form vs central finite differences in each
regime, and vs WienR's d{v,sv,a,w,sw}WienerPDF on the 400-set grid."""
import numpy as np, sys
sys.path.insert(0, "src")
from ddm_fast import grad_full_sz, _small, _large, PARAMS
a0 = 1.2; h = 1e-6; th = dict(nu=0.7, eta=1.1, a=a0, w1=0.35, w2=0.65)
for name, F, t in [("small", _small, np.linspace(0.05, a0*a0, 40)), ("large", _large, np.linspace(a0*a0, 6.0, 40))]:
    f = lambda **kw: F(t, **{**th, **kw})
    dg = f()[1]
    for i, pn in enumerate(PARAMS):
        fd = (f(**{pn: th[pn] + h})[0] - f(**{pn: th[pn] - h})[0]) / (2*h)
        m = np.abs(fd) > 1e-12
        rel = np.abs(dg[i][m] - fd[m]) / np.abs(fd[m])
        print(f"{name:5s} d/d{pn:3s} vs FD  max rel {rel.max():.1e}")
        assert rel.max() < 1e-6
# small eta, large-time branch, across the kappa switch: five gradients vs FD
t0_ = np.linspace(a0*a0, 6.0, 20)
for eta0 in [0.0, 1e-3, 1e-2, 0.1, 0.5, 0.6]:
    th0 = dict(nu=0.7, eta=eta0, a=a0, w1=0.35, w2=0.65)
    dg0 = _large(t0_, **th0)[1]
    for i, pn in enumerate(PARAMS):
        fd = (_large(t0_, **{**th0, pn: th0[pn] + h})[0] - _large(t0_, **{**th0, pn: th0[pn] - h})[0]) / (2*h)
        m = np.abs(fd) > 1e-12
        if m.any(): assert np.max(np.abs(dg0[i][m] - fd[m]) / np.abs(fd[m])) < 1e-6, (eta0, pn)
print("small-eta large-time gradients vs FD ok")
csv = lambda p: np.genfromtxt(p, delimiter=",", names=True).view(np.recarray)
g = csv("data/wienr_grad.csv")
ours = np.array([grad_full_sz(r.t, r.v, r.sv, r.a, r.w-r.sw/2, r.w+r.sw/2)[1][:, 0] for r in g])  # (400,5)
dnu, deta, da, dw1, dw2 = ours.T
# WienR parametrizes by w = (w1+w2)/2 and sw = w2-w1
cmp = {"dv": dnu, "dsv": deta, "da": da, "dw": dw1 + dw2, "dsw": (dw2 - dw1)/2}
for c, o in cmp.items():   # mixed tolerance: WienR reports ~1e-14 absolute error on its derivatives
    big = np.abs(g[c]) > 1e-10 * np.abs(g[c]).max()
    rel = (np.abs(o - g[c]) / np.abs(g[c]))[big]
    print(f"vs WienR {c:3s} ({len(g)} sets): max abs {np.abs(o-g[c]).max():.1e}  max rel {rel.max():.1e}")
    assert np.allclose(o, g[c], rtol=1e-8, atol=1e-12), c
print("GRAD CHECK PASS")

# --- seven-parameter density: closed-form gradient vs finite differences, both with and without
#     the window clamped at t - t0 - st0 < 0
from ddm_fast import grad_f7, f7, PARAMS7
for t, th7 in [(1.1, dict(nu=0.7, eta=1.1, a=1.2, w1=0.35, w2=0.65, t0=0.3, st0=0.2)),
               (0.42, dict(nu=0.7, eta=1.1, a=1.2, w1=0.35, w2=0.65, t0=0.3, st0=0.2))]:   # second: clamped
    f = lambda **kw: grad_f7(t, **{**th7, **kw})
    df = f()[1]
    for i, pn in enumerate(PARAMS7):
        fd = (f(**{pn: th7[pn] + h})[0] - f(**{pn: th7[pn] - h})[0]) / (2*h)
        rel = abs(df[i] - fd) / max(abs(fd), 1e-12)
        assert rel < 1e-6, (t, pn, df[i], fd)
    print(f"f7 t={t} all 7 gradients vs FD ok (max rel {rel:.1e})")

# --- upper barrier = lower barrier under (nu, w1, w2) -> (-nu, 1-w2, 1-w1), gradient included
from ddm_fast import grad_full_sz
tt = np.linspace(0.1, 3, 20)
gu, dgu = grad_full_sz(tt, 0.7, 1.1, 1.2, 0.3, 0.5, upper=True)
gl, dgl = grad_full_sz(tt, -0.7, 1.1, 1.2, 0.5, 0.7)
assert np.allclose(gu, gl) and np.allclose(dgu[[0,1,2]], dgl[[0,1,2]] * np.array([[-1],[1],[1]]))
assert np.allclose(dgu[3], -dgl[4]) and np.allclose(dgu[4], -dgl[3])
print("upper barrier ok")

# --- truncation rule: tol=1e-12 vs the fixed J=K=40 reference is below tol on the WienR grid
loose = np.array([grad_full_sz(r.t, r.v, r.sv, r.a, r.w-r.sw/2, r.w+r.sw/2, tol=1e-12)[0][0] for r in g])
tight = np.array([grad_full_sz(r.t, r.v, r.sv, r.a, r.w-r.sw/2, r.w+r.sw/2, tol=1e-40)[0][0] for r in g])
print(f"truncation tol=1e-12 vs tol=1e-40: max abs {np.abs(loose-tight).max():.1e}; vs WienR {np.abs(loose-g.wienr).max():.1e}")
assert np.abs(loose - tight).max() < 1e-12
print("ALL CHECKS PASS")

# --- compiled kernel (optional: cythonize -3 -i src/ddm_kernel.pyx): same values as numpy
try:
    import ddm_kernel as K
except ImportError:
    print("ddm_kernel not built; skipping kernel check")
else:
    kg = np.array([K.g_full_sz(np.array([r.t]), r.v, r.sv, r.a, r.w-r.sw/2, r.w+r.sw/2)[0] for r in g])
    print(f"compiled kernel vs numpy, 400 sets: max abs {np.abs(kg - loose).max():.1e}")
    assert np.abs(kg - loose).max() < 1e-13
    f = csv("data/wienr_full.csv")
    k7 = np.array([K.f7(np.array([r.t]), r.v, r.sv, r.a, r.w-r.sw/2, r.w+r.sw/2, 0.3, r.st0)[0] for r in f])
    n7 = np.array([f7(r.t, r.v, r.sv, r.a, r.w-r.sw/2, r.w+r.sw/2, 0.3, r.st0) for r in f])
    print(f"compiled kernel f7 vs numpy, 200 sets: max abs {np.abs(k7 - n7).max():.1e};  vs WienR {np.abs(k7 - f.wienr).max():.1e}")
    assert np.abs(k7 - n7).max() < 1e-13
    w9 = csv("data/wienr_full_p9.csv").w9; big7 = k7 > 1e-3
    print(f"WienR at tolerance 1e-9 vs closed form, 200 sets: max rel {np.max(np.abs(w9 - k7)[big7] / k7[big7]):.1e}   (Table 2 row)")
    print("KERNEL CHECK PASS")

# --- seven-parameter quadrature: 32-node rule with the cubic edge map vs adaptive quadrature of
#     g_full_sz over the window, on the 200 seven-parameter sets; and why 32 mapped nodes
from scipy.integrate import quad
def rule(n, p, r):
    x, wq = np.polynomial.legendre.leggauss(n)
    hi = r.t - 0.3; lo = max(hi - r.st0, 0.0); s = (x + 1) / 2
    u = lo + (hi - lo) * s**p; jac = (hi - lo) * p * s**(p - 1) / 2
    return np.sum(wq * jac * grad_full_sz(u, r.v, r.sv, r.a, r.w-r.sw/2, r.w+r.sw/2)[0]) / r.st0
ref = np.array([quad(lambda u: grad_full_sz(u, r.v, r.sv, r.a, r.w-r.sw/2, r.w+r.sw/2)[0][0],
                     max(r.t-0.3-r.st0, 0.0), r.t-0.3, epsabs=0, epsrel=1e-13, limit=200)[0] / r.st0
                for r in f])
big = ref > 1e-3
print("seven-parameter quadrature, max rel error vs adaptive reference (densities > 1e-3):")
for n in (8, 16, 24, 32):
    e_plain = max(abs(rule(n, 1, r) - ref[i]) / ref[i] for i, r in enumerate(f) if big[i])
    e_map = max(abs(rule(n, 3, r) - ref[i]) / ref[i] for i, r in enumerate(f) if big[i])
    print(f"  {n:2d} nodes: plain Gauss-Legendre {e_plain:.1e}   with cubic edge map {e_map:.1e}")
e32 = max(abs(rule(32, 3, r) - ref[i]) / ref[i] for i, r in enumerate(f) if big[i])
assert e32 < 1e-8
print("QUADRATURE CHECK PASS")
