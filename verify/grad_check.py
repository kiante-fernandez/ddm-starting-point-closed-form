"""Gradient of Result 1 in (nu, eta, a, w1, w2): closed form vs central finite differences in each
regime, and vs WienR's d{v,sv,a,w,sw}WienerPDF on the 400-set grid."""
import numpy as np, sys
sys.path.insert(0, "src")
from ddm_fast import grad_full_sz, _small, _large, PARAMS
a0 = 1.2; h = 1e-6; th = dict(nu=0.7, eta=1.1, a=a0, w1=0.35, w2=0.65)
fd = lambda f, th, pn: (f(**{pn: th[pn] + h})[0] - f(**{pn: th[pn] - h})[0]) / (2*h)   # central difference in pn
for name, F, t in [("small", _small, np.linspace(0.05, a0*a0, 40)), ("large", _large, np.linspace(a0*a0, 6.0, 40))]:
    f = lambda **kw: F(t, **{**th, **kw})
    dg = f()[1]
    for i, pn in enumerate(PARAMS):
        d = fd(f, th, pn); m = np.abs(d) > 1e-12
        rel = np.abs(dg[i][m] - d[m]) / np.abs(d[m])
        print(f"{name:5s} d/d{pn:3s} vs FD  max rel {rel.max():.1e}")
        assert rel.max() < 1e-6
# small eta, large-time branch, across the kappa switch: five gradients vs FD
t0_ = np.linspace(a0*a0, 6.0, 20)
for eta0 in [0.0, 1e-3, 1e-2, 0.1, 0.5, 0.6]:
    th0 = dict(nu=0.7, eta=eta0, a=a0, w1=0.35, w2=0.65)
    f0 = lambda **kw: _large(t0_, **{**th0, **kw}); dg0 = f0()[1]
    for i, pn in enumerate(PARAMS):
        d = fd(f0, th0, pn); m = np.abs(d) > 1e-12
        if m.any(): assert np.max(np.abs(dg0[i][m] - d[m]) / np.abs(d[m])) < 1e-6, (eta0, pn)
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
# --- independent scalar implementation (src/ddm_closed.py): its own quadrature self-check
import ddm_closed
w_scalar = ddm_closed.check_all()
print(f"scalar reference vs quadrature, 40 sets: max abs {w_scalar:.1e}")
assert w_scalar < 1e-13

print("GRAD CHECK PASS")

# --- literature grid (Tran et al. 2021 priors, verify/tran_grid.R): density and five gradients vs WienR
tg = csv("data/tran_grid.csv")
res = [grad_full_sz(r.t, r.v, r.sv, r.a, r.w-r.sw/2, r.w+r.sw/2) for r in tg]
tgg = np.array([x[0][0] for x in res]); tgd = np.array([x[1][:, 0] for x in res]).T
big = tg.wienr > 1e-10                                    # WienR's own relative accuracy degrades below this
rel = np.abs(tgg/tg.wienr - 1)
print(f"Tran grid ({len(tg)} sets, {big.sum()} with g > 1e-10): density max rel {rel[big].max():.1e}; "
      f"below 1e-10: max rel {rel[~big & (tg.wienr > 1e-300)].max():.1e} (WienR's error, see highprec2)")
assert rel[big].max() < 1e-10
for lo, hi in ((-3, 1), (-6, -3), (-10, -6), (-20, -10), (-300, -20)):
    mb = (tg.wienr >= 10.0**lo) & (tg.wienr < 10.0**hi)
    print(f"  g in [1e{lo}, 1e{hi}): n = {mb.sum():4d}, max rel {rel[mb].max():.1e}")
assert (tgg >= 0).all()          # no negative density from the truncated alternating series
for c, o in {"dv": tgd[0], "dsv": tgd[1], "da": tgd[2], "dw": tgd[3] + tgd[4], "dsw": (tgd[4] - tgd[3])/2}.items():
    e = (np.abs(o - tg[c]) / np.maximum(np.abs(tg[c]), tg.wienr))[big]
    print(f"  Tran grid {c:3s}: max |diff|/max(|ref|, g) {e.max():.1e}")
    assert e.max() < 5e-9, c
print("TRAN GRID CHECK PASS")

# --- small-Delta branch (w2 - w1 < 1e-2): continuity across the threshold, gradient vs FD inside the
#     branch, and w1 = w2.  Accuracy vs a 40-digit reference is in highprec2.py.
for t_ in (np.linspace(0.05, a0*a0, 20), np.linspace(a0*a0, 6.0, 20)):
    ga, dga = grad_full_sz(t_, 0.7, 1.1, a0, 0.5 - 0.005, 0.5 + 0.005)            # closed form, Delta = 1e-2
    gb, dgb = grad_full_sz(t_, 0.7, 1.1, a0, 0.5 - 0.005 + 1e-9, 0.5 + 0.005)     # average, Delta = 1e-2 - 1e-9
    assert np.allclose(ga, gb, rtol=1e-9) and np.allclose(dga, dgb, rtol=1e-6, atol=1e-9*np.abs(dga).max())
    for D in (1e-3, 0.0):
        thD = dict(nu=0.7, eta=1.1, a=a0, w1=0.5 - D/2, w2=0.5 + D/2)
        f = lambda **kw: grad_full_sz(t_, **{**thD, **kw})
        dg = f()[1]
        for i, pn in enumerate(PARAMS[:3]):
            d = fd(f, thD, pn); m = np.abs(d) > 1e-12
            assert np.max(np.abs(dg[i][m] - d[m]) / np.abs(d[m])) < 1e-6, (D, pn)
        if D == 0:
            fdw = (f(w1=0.5 + h, w2=0.5 + h)[0] - f(w1=0.5 - h, w2=0.5 - h)[0]) / (2*h)   # d/dwbar; even in Delta so dw1 = dw2
            assert np.allclose(dg[3], fdw/2, rtol=1e-6) and np.allclose(dg[4], fdw/2, rtol=1e-6)
        else:
            for i, pn in ((3, "w1"), (4, "w2")):
                d = fd(f, thD, pn)
                assert np.allclose(dg[i], d, rtol=1e-6, atol=1e-8*np.abs(d).max()), (D, pn)
print("small-Delta branch: threshold continuity and gradients vs FD ok (Delta = 1e-3 and 0)")

# --- seven-parameter density: closed-form gradient vs finite differences, both with and without
#     the window clamped at t - t0 - st0 < 0
from ddm_fast import grad_f7, PARAMS7
for t, th7 in [(1.1, dict(nu=0.7, eta=1.1, a=1.2, w1=0.35, w2=0.65, t0=0.3, st0=0.2)),
               (0.42, dict(nu=0.7, eta=1.1, a=1.2, w1=0.35, w2=0.65, t0=0.3, st0=0.2))]:   # second: clamped
    f = lambda **kw: grad_f7(t, **{**th7, **kw})
    df = f()[1]
    for i, pn in enumerate(PARAMS7):
        d = fd(f, th7, pn)
        rel = abs(df[i] - d) / max(abs(d), 1e-12)
        assert rel < 1e-6, (t, pn, df[i], d)
    print(f"f7 t={t} all 7 gradients vs FD ok (max rel {rel:.1e})")

# --- small-/large-time representations across the switch at t/a^2 = 1, and positivity of both
for x in (0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0):
    gs = _small(np.array([x]), 1.0, 1.0, 1.0, 0.4, 0.6)[0][0]; gl = _large(np.array([x]), 1.0, 1.0, 1.0, 0.4, 0.6)[0][0]
    print(f"  t/a^2 = {x:3.1f}: |small/large - 1| = {abs(gs/gl - 1):.1e}")
    if x <= 1.0: assert abs(gs/gl - 1) < 1e-12
for r in g:
    assert (grad_full_sz(np.linspace(0.01, 6, 600), r.v, r.sv, r.a, r.w-r.sw/2, r.w+r.sw/2)[0] >= 0).all()
print("switch agreement and positivity on 400 x 600 t ok")

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
    for D in (9e-3, 1e-3, 0.0):     # small-Delta branch of the kernel
        tt_ = np.linspace(0.05, 4, 50)
        assert np.abs(K.g_full_sz(tt_, 0.7, 1.1, a0, 0.5 - D/2, 0.5 + D/2) - grad_full_sz(tt_, 0.7, 1.1, a0, 0.5 - D/2, 0.5 + D/2)[0]).max() < 1e-13
    print("compiled kernel small-Delta branch vs numpy ok")
    f = csv("data/wienr_full.csv")
    k7 = np.array([K.f7(np.array([r.t]), r.v, r.sv, r.a, r.w-r.sw/2, r.w+r.sw/2, 0.3, r.st0)[0] for r in f])
    n7 = np.array([grad_f7(r.t, r.v, r.sv, r.a, r.w-r.sw/2, r.w+r.sw/2, 0.3, r.st0)[0] for r in f])
    print(f"compiled kernel f7 vs numpy, 400 sets: max abs {np.abs(k7 - n7).max():.1e};  vs WienR {np.abs(k7 - f.wienr).max():.1e}")
    assert np.abs(k7 - n7).max() < 1e-13
    print("KERNEL CHECK PASS")

# --- seven-parameter quadrature: 48-node rule with the cubic edge map vs adaptive quadrature of
#     grad_full_sz over the window, on the 400 seven-parameter sets; and why 48 mapped nodes
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
for n in (16, 24, 32, 48):
    e_plain = max(abs(rule(n, 1, r) - ref[i]) / ref[i] for i, r in enumerate(f) if big[i])
    e_map = max(abs(rule(n, 3, r) - ref[i]) / ref[i] for i, r in enumerate(f) if big[i])
    print(f"  {n:2d} nodes: plain Gauss-Legendre {e_plain:.1e}   with cubic edge map {e_map:.1e}")
print("48 mapped nodes, rel error by w1 (rows) and (t - t0)/st0 (cols 0.04, 0.2, 0.8), a = 1.2, nu = 1, eta = 1, w2 = w1 + 0.2, st0 = 0.25:")
for w1_ in (0.2, 0.1, 0.05, 0.02):
    row = []
    for frac in (0.04, 0.2, 0.8):
        r7 = np.rec.fromrecords([(0.3 + frac*0.25, 1.2, 1.0, w1_ + 0.1, 1.0, 0.2, 0.25)], names="t,a,v,w,sv,sw,st0")[0]
        R = quad(lambda u: grad_full_sz(u, 1.0, 1.0, 1.2, w1_, w1_ + 0.2)[0][0], max(r7.t-0.3-0.25, 0.0), r7.t-0.3, epsabs=0, epsrel=1e-13, limit=500)[0] / 0.25
        row.append(abs(rule(48, 3, r7) - R) / R)
    print(f"  w1 = {w1_:4.2f}: " + "  ".join(f"{e:.0e}" for e in row))
e48 = max(abs(rule(48, 3, r) - ref[i]) / ref[i] for i, r in enumerate(f) if big[i])
assert e48 < 1e-10
print("QUADRATURE CHECK PASS")
