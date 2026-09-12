import numpy as np
from ddm_fast import g_full_sz
from scipy import integrate
rng = np.random.default_rng(1)
a, nu, eta, w0, sw = 1.2, 1.0, 1.2, 0.5, 0.5
w1, w2 = w0 - sw/2, w0 + sw/2
N, dt = 400_000, 5e-5
shift = 0.5826*np.sqrt(dt)          # Broadie-Glasserman-Kou continuity correction
lo_b, up_b = shift, a - shift
v = rng.normal(nu, eta, N); x = a*rng.uniform(w1, w2, N)
t = np.zeros(N); alive = np.ones(N, bool); hit_low = np.zeros(N, bool); sq = np.sqrt(dt)
for step in range(int(6/dt)):
    idx = np.flatnonzero(alive)
    if idx.size == 0: break
    x[idx] += v[idx]*dt + sq*rng.standard_normal(idx.size); t[idx] += dt
    low = x[idx] <= lo_b; up = x[idx] >= up_b
    hit_low[idx[low]] = True; alive[idx[low | up]] = False
edges = np.linspace(0, 3.0, 31)
p_cf = np.array([integrate.quad(lambda u: g_full_sz(u, nu, eta, a, w1, w2)[0], lo, hi)[0] for lo, hi in zip(edges[:-1], edges[1:])])
p_sim = np.histogram(t[hit_low], bins=edges)[0]/N
z = (p_sim - p_cf)/np.sqrt(p_cf*(1-p_cf)/N)
print(f"P(lower): sim {hit_low.mean():.4f}  closed-form {integrate.quad(lambda u: g_full_sz(u,nu,eta,a,w1,w2)[0],0,20)[0]:.4f}")
print("z-scores per 100ms bin:", np.round(z, 2))
chi2 = np.sum(z**2); print(f"chi2 = {chi2:.1f} on 30 bins")
