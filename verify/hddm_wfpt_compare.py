"""Difference columns of Table 2: WienR, rtdists and hddm-wfpt (HDDM / HSSM likelihood,
Navarro-Fuss series + adaptive Simpson over sz and st) relative to the closed form on the 400
parameter sets of data/ (six parameters, and seven with t0 = 0.3), densities above 1e-3; and
hddm-wfpt's single-thread cost.  Run wienr_grid.R, wienr_full.R, rtdists_prec.R, rtdists_full.R first.
pip install hddm-wfpt.  Run from the repo root with OMP_NUM_THREADS=1.
hddm-wfpt conventions: signed RT (negative = lower barrier), relative z and sz, window t +- st/2."""
import sys; sys.path.insert(0, "src")
import numpy as np, timeit
from hddm_wfpt import wfpt
import ddm_kernel as K
csv = lambda p: np.genfromtxt(p, delimiter=",", names=True).view(np.recarray)
g = csv("data/wienr_grad.csv"); f = csv("data/wienr_full.csv")
# reference = the closed form
k6 = np.array([K.g_full_sz(np.array([r.t]), r.v, r.sv, r.a, r.w-r.sw/2, r.w+r.sw/2)[0] for r in g])
k7 = np.array([K.f7(np.array([r.t]), r.v, r.sv, r.a, r.w-r.sw/2, r.w+r.sw/2, 0.3, r.st0)[0] for r in f])
b6 = k6 > 1e-3; b7 = k7 > 1e-3
rel = lambda x, k, b: np.max(np.abs(x - k)[b] / k[b])
p6 = csv("data/rtdists_prec.csv"); p7 = csv("data/rtdists_full.csv")
print(f"WienR   6p {rel(g.wienr, k6, b6):.1e}  7p {rel(f.wienr, k7, b7):.1e}")
print(f"rtdists 6p {rel(p6.p3, k6, b6):.1e}  7p {rel(p7.rtd3, k7, b7):.1e}   (precision 3)")
def acc(**kw):
    h6 = np.array([wfpt.full_pdf(-r.t, r.v, r.sv, r.a, r.w, r.sw, 0.0, 0.0, 1e-4, **kw) for r in g])
    h7 = np.array([wfpt.full_pdf(-r.t, r.v, r.sv, r.a, r.w, r.sw, 0.3 + r.st0/2, r.st0, 1e-4, **kw) for r in f])
    return (np.abs(h6-k6).max(), np.max(np.abs(h6-k6)[b6]/k6[b6]),
            np.abs(h7-k7).max(), np.max(np.abs(h7-k7)[b7]/k7[b7]))
tt = np.random.default_rng(1).uniform(0.1, 3, 2000)
tm = lambda fn: 1e6 * min(timeit.repeat(fn, number=1, repeat=3)) / len(tt)
print("error of hddm-wfpt relative to the closed form; cost per evaluation, single thread")
print(f"{'setting':40s} {'6p abs':>8s} {'6p rel':>8s} {'7p abs':>8s} {'7p rel':>8s} {'6p us':>8s} {'7p us':>8s}")
# HDDM's fitting likelihood (wiener_like) settings: err=1e-4 (HDDM default), depth 10, simps_err=1e-8.
kw = dict(n_st=10, n_sz=10, simps_err=1e-8)
a6, r6, a7, r7 = acc(**kw)
t6 = tm(lambda: wfpt.pdf_array(-tt, 1.0, 1.0, 1.2, 0.5, 0.2, 0.0, 0.0, 1e-4, **kw))
t7 = tm(lambda: wfpt.pdf_array(-tt, 1.0, 1.0, 1.2, 0.5, 0.2, 0.1, 0.2, 1e-4, **kw))
print(f"{'hddm-wfpt (wiener_like settings)':40s} {a6:8.1e} {r6:8.1e} {a7:8.1e} {r7:8.1e} {t6:8.2f} {t7:8.2f}")
print("(closed form's own error: 6p exact to ~1e-14; 7p 48-node quadrature 5e-11 relative)")
