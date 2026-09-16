"""Wall time per likelihood evaluation vs trials per evaluation, Python side.
Median over 5 published parameter sets, response times simulated from each.
  OMP_NUM_THREADS=1 python curve.py  ->  curve_py.csv"""
import csv
import time
import numpy as np
import ddmsz
from hddm_wfpt import wfpt

N = [30, 100, 300, 1000, 3000, 10_000, 30_000, 100_000]
BUDGET, NSET = 10.0, 5
rng = np.random.default_rng(1)

g = np.genfromtxt("../R/bench_sets.csv", delimiter=",", names=True)[:NSET]
sets = [dict(v=r["v"], a=r["a"]/2, z=r["w"], sz=r["sw"], sv=r["sv"],
             t=r["t0"] + r["st0"]/2, st=r["st0"]) for r in g]

def draw(n, P):
    lo, hi = P["t"] - P["st"]/2 + 1e-4, P["t"] + P["st"]/2 + 4.0
    grid = np.linspace(lo, hi, 400)
    top = 1.05 * max(ddmsz.full_ddm(grid, np.full(400, h), **P)[0].max() for h in (0, 1))
    x = np.empty(0)
    while x.size < n:
        c, hit = rng.uniform(lo, hi, 8*n), rng.random(8*n) < 0.5
        keep = rng.uniform(0, top, c.size) < ddmsz.full_ddm(c, hit, **P)[0]
        x = np.concatenate([x, np.where(hit[keep], c[keep], -c[keep])])
    return x[:n]

def once(f, x):
    t0 = time.perf_counter()
    f(x)
    return time.perf_counter() - t0

def hddm(P, err):
    h = dict(v=P["v"], sv=P["sv"], a=2*P["a"], z=P["z"], sz=P["sz"], t=P["t"], st=P["st"])
    return lambda x: wfpt.wiener_logp_array(
        x=x, err=err, **{k: np.full(x.size, float(v)) for k, v in h.items()})

METHODS = [("ours, density + 7 gradients", lambda P: (lambda x: ddmsz.full_ddm(np.abs(x), x > 0, **P))),
           ("hddm-wfpt (HSSM blackbox)", lambda P: hddm(P, 1e-8))]

rows = []
for name, mk in METHODS:
    for n in N:
        ms = []
        for P in sets:
            x, f = draw(n, P), mk(P)
            f(x[:5])                                    # warm-up
            ms.append(min(once(f, x) for _ in range(3)) * 1e3)
        med = float(np.median(ms))
        rows.append((name, n, med))
        print(f"{name:30s} n={n:7d}  {med:10.3f} ms", flush=True)
        if med / 1e3 > BUDGET:
            break

with open("curve_py.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["method", "n", "ms"])
    w.writerows(rows)
print("wrote curve_py.csv")
