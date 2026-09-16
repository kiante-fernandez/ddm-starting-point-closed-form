"""full_ddm cost and accuracy vs hddm-wfpt (HSSM's blackbox likelihood) over parameter sets
drawn from the priors of Tran et al. (2021); see ../R/bench_sets.R.
  OMP_NUM_THREADS=1 python bench.py  ->  bench_py.csv"""
import os
import time
import numpy as np
import ddmsz
from hddm_wfpt import wfpt

NRT = int(os.environ.get("NRT", 1000))
rng = np.random.default_rng(1)

g = np.genfromtxt("../R/bench_sets.csv", delimiter=",", names=True)   # Tran et al. priors
NSET = len(g)
sets = [dict(v=r["v"], a=r["a"]/2, z=r["w"], sz=r["sw"], sv=r["sv"],  # HSSM: a is half, t centred
             t=r["t0"] + r["st0"]/2, st=r["st0"]) for r in g]

def draw(n, P):
    """signed rts from the model itself, by rejection against the density's peak."""
    lo, hi = P["t"] - P["st"]/2 + 1e-4, P["t"] + P["st"]/2 + 4.0
    grid = np.linspace(lo, hi, 400)
    top = 1.05 * max(ddmsz.full_ddm(grid, np.full(400, h), **P)[0].max() for h in (0, 1))
    x = np.empty(0)
    while x.size < n:
        c, hit = rng.uniform(lo, hi, 8*n), rng.random(8*n) < 0.5
        keep = rng.uniform(0, top, c.size) < ddmsz.full_ddm(c, hit, **P)[0]
        x = np.concatenate([x, np.where(hit[keep], c[keep], -c[keep])])
    return x[:n]

def ours(P, tol):
    return lambda x: ddmsz.full_ddm(np.abs(x), x > 0, tol=tol, **P)[0]

def hddm(P, err):
    h = dict(v=P["v"], sv=P["sv"], a=2*P["a"], z=P["z"], sz=P["sz"], t=P["t"], st=P["st"])
    return lambda x: np.exp(wfpt.wiener_logp_array(
        x=x, err=err, **{k: np.full(x.size, float(v)) for k, v in h.items()}))

def timed(f, x, reps=10):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f(x)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) / x.size * 1e6

METHODS = [("ours 1e-12 (density + 7 grads)", ours, 1e-12),
           ("ours 1e-8", ours, 1e-8),
           ("hddm-wfpt err=1e-8", hddm, 1e-8),
           ("hddm-wfpt err=1e-4", hddm, 1e-4)]

rows = []
for i, P in enumerate(sets, 1):
    x = draw(NRT, P)
    ref = ours(P, 1e-12)(x)
    big = ref > 1e-10
    for name, mk, prec in METHODS:
        f = mk(P, prec)
        us = timed(f, x)
        acc = 0.0 if name.endswith("grads)") else float(np.max(np.abs(f(x)[big]/ref[big] - 1)))
        rows.append((i, name, us, acc))
        print(f"set {i:2d}  {name:30s} {us:8.2f} us/trial  rel {acc:.1e}", flush=True)

with open("bench_py.csv", "w") as fh:
    fh.write("set,method,us_per_trial,rel_diff\n")
    for r in rows:
        fh.write("%d,%s,%.4f,%.3e\n" % r)

print(f"\n== median over {NSET} sets (IQR), accuracy vs ours at 1e-12 ==")
for name, _, _ in METHODS:
    u = np.array([r[2] for r in rows if r[1] == name])
    a = np.array([r[3] for r in rows if r[1] == name])
    q = np.percentile(u, [25, 75])
    print(f"{name:30s} {np.median(u):8.2f} us  [{q[0]:7.2f}, {q[1]:8.2f}]   "
          f"rel: median {np.median(a):.1e}  max {a.max():.1e}")

# anchor: the reference above is ours at 1e-12, so check it against mpmath at 30 digits
import mpmath as mp
mp.mp.dps = 30

def hp(rt, P):                       # f = (1/st) int_u (1/sz) int_w g_eta(u, w) dw du
    v, a, z, t, sz, sv, st = (mp.mpf(P[k]) for k in ddmsz.PARAMS)
    a2 = 2*a
    def g(u, w):
        S = 1 + sv**2*u
        s = sum((-1)**j * (j*a2 + a2*w if j % 2 == 0 else (j+1)*a2 - a2*w)
                * mp.e**(-(j*a2 + a2*w if j % 2 == 0 else (j+1)*a2 - a2*w)**2/(2*u)) for j in range(60))
        return mp.e**((-v**2*u - 2*v*a2*w + sv**2*(a2*w)**2)/(2*S)) * s/mp.sqrt(2*mp.pi*u**3*S)
    gb = lambda u: mp.quad(lambda w: g(u, w), [z - sz/2, z + sz/2])/sz if sz > 0 else g(u, z)
    hi = mp.mpf(rt) - (t - st/2)
    lo = max(hi - st, mp.mpf(0))
    return mp.quad(gb, [lo, hi])/st if st > 0 else gb(hi)

print("\nmpmath anchor (30 digits), 3 rts on each of 3 sets:")
worst = 0.0
for P in sets[:3]:
    x = np.abs(draw(3, P))
    f = ddmsz.full_ddm(x, np.zeros(3), tol=1e-12, **P)[0]
    for xi, fi in zip(x, f):
        e = abs(fi/float(hp(xi, P)) - 1)
        worst = max(worst, e)
print(f"  ours at 1e-12 vs 30-digit reference: max rel {worst:.1e}")
assert worst < 1e-9, worst
