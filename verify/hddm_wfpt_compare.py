"""hddm-wfpt (HDDM / HSSM likelihood, Navarro-Fuss series + adaptive Simpson over sz and st)
vs the closed form: hddm-wfpt's error relative to the closed form on the 400 + 200 parameter
sets of data/, and single-thread cost.  (The closed form itself matches WienR to 1e-14 there.)
pip install hddm-wfpt.  Run from the repo root with OMP_NUM_THREADS=1.
hddm-wfpt conventions: signed RT (negative = lower barrier), relative z and sz, window t +- st/2."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, time
from hddm_wfpt import wfpt
import ddm_kernel as K
g = pd.read_csv("data/wienr_grid.csv"); f = pd.read_csv("data/wienr_full.csv")
# reference = the closed form
k6 = np.array([K.g_full_sz(np.array([r.t]), r.v, r.sv, r.a, r.w-r.sw/2, r.w+r.sw/2)[0] for r in g.itertuples()])
k7 = np.array([K.f7(np.array([r.t]), r.v, r.sv, r.a, r.w-r.sw/2, r.w+r.sw/2, 0.3, r.st0)[0] for r in f.itertuples()])
b6 = k6 > 1e-3; b7 = k7 > 1e-3
def acc(**kw):
    h6 = np.array([wfpt.full_pdf(-r.t, r.v, r.sv, r.a, r.w, r.sw, 0.0, 0.0, 1e-4, **kw) for r in g.itertuples()])
    h7 = np.array([wfpt.full_pdf(-r.t, r.v, r.sv, r.a, r.w, r.sw, 0.3 + r.st0/2, r.st0, 1e-4, **kw) for r in f.itertuples()])
    return (np.abs(h6-k6).max(), np.max(np.abs(h6-k6)[b6]/k6[b6]),
            np.abs(h7-k7).max(), np.max(np.abs(h7-k7)[b7]/k7[b7]))
tt = np.random.default_rng(1).uniform(0.1, 3, 2000)
def tm(fn):
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return 1e6*best/len(tt)
print("error of hddm-wfpt relative to the closed form; cost per evaluation, single thread")
print(f"{'setting':40s} {'6p abs':>8s} {'6p rel':>8s} {'7p abs':>8s} {'7p rel':>8s} {'6p us':>8s} {'7p us':>8s}")
# HDDM's fitting likelihood (wiener_like) settings: err=1e-4 (HDDM default), depth 10, simps_err=1e-8.
kw = dict(n_st=10, n_sz=10, simps_err=1e-8)
a6, r6, a7, r7 = acc(**kw)
t6 = tm(lambda: wfpt.pdf_array(-tt, 1.0, 1.0, 1.2, 0.5, 0.2, 0.0, 0.0, 1e-4, **kw))
t7 = tm(lambda: wfpt.pdf_array(-tt, 1.0, 1.0, 1.2, 0.5, 0.2, 0.1, 0.2, 1e-4, **kw))
print(f"{'hddm-wfpt (wiener_like settings)':40s} {a6:8.1e} {r6:8.1e} {a7:8.1e} {r7:8.1e} {t6:8.2f} {t7:8.2f}")
print(f"{'closed form, compiled (ddm_kernel)':40s} {'0':>8s} {'0':>8s} {'0':>8s} {'0':>8s} "
      f"{tm(lambda: K.g_full_sz(tt, 1.0, 1.0, 1.2, 0.4, 0.6)):8.2f} {tm(lambda: K.f7(tt, 1.0, 1.0, 1.2, 0.4, 0.6, 0.0, 0.2)):8.2f}")
print("(closed form's own error: 6p exact to ~1e-14; 7p quadrature 1.5e-9 relative, see STATUS.md)")
