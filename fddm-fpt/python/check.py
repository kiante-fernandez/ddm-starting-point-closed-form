"""ddmsz (Cython wrapper) vs ../../src/ddm_fast.py and ../R/fddm_fixed.csv.
  cythonize -3 -i ddmsz.pyx && python check.py"""
import sys
import numpy as np
import ddmsz
sys.path.insert(0, "../../src")
from ddm_fast import grad_full_sz

def core(t, nu, eta, a, w1, w2, tol=1e-12):      # rows as ddm_fast: d/dw1, d/dw2
    g, dg = ddmsz.density(np.atleast_1d(t), nu, eta, a, (w1 + w2)/2, w2 - w1, tol)
    return g, to_edges(dg.T)

def to_edges(d):
    return np.vstack([d[:3], d[3]/2 - d[4], d[3]/2 + d[4], d[5:]])

t = np.concatenate([np.linspace(0.02, 1.4, 60), np.linspace(1.5, 8.0, 40)])   # both sides of t/a^2 = 1
worst = 0.0
for nu, eta, a, w1, w2 in [(0.7, 1.1, 1.2, 0.35, 0.65), (-2.0, 0.0, 0.8, 0.2, 0.9),
                           (3.0, 2.5, 2.0, 0.45, 0.55), (0.0, 0.05, 1.0, 0.499, 0.501),
                           (1.5, 1.0, 0.6, 0.5, 0.5), (-1.0, 3.0, 1.5, 0.1, 0.3)]:
    c, dc = core(t, nu, eta, a, w1, w2)
    p, dp = grad_full_sz(t, nu, eta, a, w1, w2)
    scale = np.maximum(np.abs(p), np.abs(p).max()*1e-12)
    e = max(np.max(np.abs(c - p)/scale), np.max(np.abs(dc - dp)/np.maximum(np.abs(dp), scale)))
    print(f"nu={nu:5} eta={eta:5} a={a:4} w=({w1}, {w2}): max rel {e:.1e}")
    worst = max(worst, e)
assert worst < 1e-8, worst   # d/deta at eta >= 2.5: ddm_fast is the loose one (4e-9 vs 3e-11 here)
print(f"PORT CHECK PASS (worst {worst:.1e})")

# seven parameters vs ddm_fast.grad_f7, including a window clamped at u = 0
from ddm_fast import grad_f7

def core7(t, nu, eta, a, w1, w2, t0, st0, tol=1e-12):
    f, df = ddmsz.density7(np.atleast_1d(t), nu, eta, a, (w1 + w2)/2, w2 - w1, t0, st0, tol)
    return f, to_edges(df.T)

worst7 = 0.0
for p in [(0.7, 1.1, 1.2, 0.35, 0.65, 0.2, 0.1), (-1.5, 0.8, 0.9, 0.45, 0.55, 0.15, 0.3),
          (2.0, 2.0, 1.5, 0.2, 0.202, 0.25, 0.05)]:
    for tt in (p[5] + 0.01, p[5] + 0.06, p[5] + 0.4, p[5] + 2.0):     # first two clamp the window
        f, df = core7(tt, *p)
        fp, dfp = grad_f7(tt, *p)
        e = max(abs(f[0]/fp - 1), np.max(np.abs(df[:, 0] - dfp)/np.maximum(np.abs(dfp), abs(fp))))
        worst7 = max(worst7, e)
assert worst7 < 1e-9, worst7

t0, st0 = 0.2, 0.0                                                   # st0 = 0 limit
f, df = core7(0.7, 0.7, 1.1, 1.2, 0.35, 0.65, t0, st0)
g, dg = grad_full_sz(np.array([0.7 - t0]), 0.7, 1.1, 1.2, 0.35, 0.65)
assert abs(f[0]/g[0] - 1) < 1e-14 and np.allclose(df[:5, 0], dg[:, 0], rtol=1e-12)
print(f"F7 CHECK PASS (worst {worst7:.1e}; st0 = 0 returns the six-parameter density)")

# fixed start (w1 = w2) vs fddm (fddm_ref.R)
r = np.genfromtxt("../R/fddm_fixed.csv", delimiter=",", names=True)
big = r["f"] > 1e-10                    # below this fddm's err_tol dominates
res = [core(np.array([x["t"]]), x["v"], x["sv"], x["a"], x["w"], x["w"]) for x in r]
dg = np.array([q[1][:, 0] for q in res])
ours = {"f": np.array([q[0][0] for q in res]), "dv": dg[:, 0], "dsv": dg[:, 1],
        "da": dg[:, 2], "dw": dg[:, 3] + dg[:, 4]}
worstf = {k: np.max(np.abs(v - r[k])[big] / np.maximum(np.abs(r[k]), r["f"])[big])
          for k, v in ours.items()}
print(f"vs fddm at w1 = w2, {big.sum()} sets: "
      + "  ".join(f"{k} {v:.1e}" for k, v in worstf.items()))
# loose on derivatives: fddm's dw is off 2e-4 where a 40-digit reference matches this core to 12 digits
assert worstf.pop("f") < 1e-7 and max(worstf.values()) < 1e-1, worstf
print("FDDM CHECK PASS (density strict, derivatives to fddm's own accuracy)")

# HSSM/HDDM parametrization vs hddm-wfpt's full_pdf, and its gradient vs finite differences
from hddm_wfpt import wfpt

P = dict(v=1.2, a=0.75, z=0.45, t=0.3, sz=0.1, sv=0.8, st=0.12)
x = np.array([-0.55, -0.9, 0.42, 0.8, 1.6, -2.2])
f, df = ddmsz.full_ddm(np.abs(x), x > 0, **P)
ref = np.array([wfpt.full_pdf(xi, P["v"], P["sv"], 2*P["a"], P["z"], P["sz"], P["t"], P["st"],
                              1e-10, 10, 10, 1, 1e-10) for xi in x])
e = np.max(np.abs(f/ref - 1))
print(f"full_ddm vs hddm-wfpt, {len(x)} trials: max rel {e:.1e}")
assert e < 1e-6, e

h = 1e-6
fd = np.column_stack([(ddmsz.full_ddm(np.abs(x), x > 0, **{**P, k: P[k] + h})[0]
                       - ddmsz.full_ddm(np.abs(x), x > 0, **{**P, k: P[k] - h})[0])/(2*h)
                      for k in ddmsz.PARAMS])
e = np.max(np.abs(df - fd)/np.maximum(np.abs(fd), f[:, None]))
print(f"full_ddm gradient vs finite differences: max rel {e:.1e}")
assert e < 1e-6, e
print("FULL_DDM CHECK PASS")
