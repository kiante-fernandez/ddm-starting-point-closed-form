"""ddm_fast and the compiled core against a 50-digit mpmath quadrature of Blurton Eq. 1 over w
(never the closed form) on N adversarial sets; gradients by central differences on every fifth set.
    python verify/highprec_grid.py [N]      # default 1000, ~10 min; writes data/highprec_grid.csv
"""
import sys
sys.path.insert(0, 'src')
import numpy as np
import mpmath as mp
from ddm_fast import grad_full_sz, PARAMS

mp.mp.dps = 50
H = mp.mpf('1e-12')                                  # finite-difference step, 50 digits

def g_eta(t, nu, eta, a, w, J=40):
    S = 1 + eta**2*t
    r = lambda j: j*a + a*w if j % 2 == 0 else (j + 1)*a - a*w
    s = sum((-1)**j*r(j)*mp.exp(-r(j)**2/(2*t)) for j in range(J))
    return mp.exp((-nu**2*t - 2*nu*a*w + eta**2*(a*w)**2)/(2*S))*s/mp.sqrt(2*mp.pi*t**3*S)

def ref(t, nu, eta, a, w1, w2, check=False):
    f = lambda w: g_eta(t, nu, eta, a, w)        # scaled to O(1): mp.quad's stopping rule is absolute
    s = max(abs(f(x)) for x in (w1, (w1 + w2)/2, w2))
    pts = [w1 + (w2 - w1)*mp.mpf(x) for x in ([0, 0.003, 0.03, 0.3, 0.7, 0.97, 0.997, 1] if check else [0, 1])]
    return s*mp.quad(lambda w: f(w)/s, pts, maxdegree=9 if check else 6)/(w2 - w1)

def ref_grad(t, nu, eta, a, w1, w2):
    th = [mp.mpf(x) for x in (nu, eta, a, w1, w2)]
    out = []
    for i in range(5):
        up, dn = th[:], th[:]
        up[i] += H; dn[i] -= H
        out.append((ref(t, *up) - ref(t, *dn))/(2*H))
    return out

def rel(x, r, scale):
    return float(abs(mp.mpf(x) - r)/max(abs(r), scale)) if r != 0 or scale != 0 else 0.0

try:
    sys.path.insert(0, 'fddm-fpt/python'); import ddmsz
except ImportError:
    ddmsz = None

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
rng = np.random.default_rng(20260916)
lu = lambda lo, hi: np.exp(rng.uniform(np.log(lo), np.log(hi), N))
a = lu(0.2, 6); t = a**2*lu(0.01, 5); nu = rng.uniform(-10, 10, N)
eta = np.where(rng.random(N) < 0.1, 0.0, lu(1e-4, 3))
w1 = np.where(rng.random(N) < 0.5, lu(1e-6, 0.5), rng.uniform(0.02, 0.5, N))
w2 = np.where(rng.random(N) < 0.5, w1 + lu(1e-4, 1)*(1 - 1e-6 - w1), 1 - lu(1e-6, 0.5)*(1 - w1))

rows = []
for i in range(N):
    p = (t[i], nu[i], eta[i], a[i], w1[i], w2[i]); R = ref(*[mp.mpf(x) for x in p])
    g, dg = grad_full_sz(*p); g = g[0]; dg = dg[:, 0]
    row = dict(t=t[i], nu=nu[i], eta=eta[i], a=a[i], w1=w1[i], w2=w2[i], ref=float(R),
               err=rel(g, R, 0), err_core=np.nan, err_grad=np.nan,
               self_check=rel(ref(*[mp.mpf(x) for x in p], check=True), R, 0) if i % 10 == 0 else np.nan)
    if ddmsz is not None:
        row['err_core'] = rel(ddmsz.density(np.array([t[i]]), nu[i], eta[i], a[i], (w1[i] + w2[i])/2, w2[i] - w1[i])[0][0], R, 0)
    if i % 5 == 0:
        row['err_grad'] = max(rel(x, r, abs(R)) for x, r in zip(dg, ref_grad(*[mp.mpf(x) for x in p])))
    rows.append(row)
    if i % 100 == 99: print(f"{i + 1}/{N} sets", flush=True)

cols = list(rows[0])
np.savetxt('data/highprec_grid.csv', np.array([[r[c] for c in cols] for r in rows]), delimiter=',',
           header=','.join(cols), comments='', fmt='%.17g')
D = {c: np.array([r[c] for r in rows]) for c in cols}
print(f"\nreference self-check (two subdivisions, degrees 6 and 9): max rel diff {np.nanmax(D['self_check']):.1e} "
      f"on {np.sum(~np.isnan(D['self_check']))} sets")
under = D['ref'] < 1e-300
print(f"{under.sum()} sets below the double range, code returns 0 on {(under & (D['err'] == 1)).sum()} of them; the rest:")
print(f"{N} sets, 50-digit reference; max relative error, density (ddm_fast | compiled core) and gradients:")
for name, m in [('all', ~under), ('w1 <= 1e-3', D['w1'] <= 1e-3), ('w2 >= 1 - 1e-3', D['w2'] >= 1 - 1e-3),
                ('w2 - w1 < 1e-2 (average branch)', D['w2'] - D['w1'] < 1e-2), ('eta <= 1e-2', D['eta'] <= 1e-2),
                ('t/a^2 >= 1 (large-time form)', D['t']/D['a']**2 >= 1), ('density < 1e-10', D['ref'] < 1e-10),
                ('density < 1e-100', D['ref'] < 1e-100)]:
    m = m & ~under; gm = m & ~np.isnan(D['err_grad'])
    mx = lambda x: np.nanmax(x) if x.size and not np.all(np.isnan(x)) else np.nan
    print(f"  {name:34s} {m.sum():5d} sets   {mx(D['err'][m]):.1e} | {mx(D['err_core'][m]):.1e}"
          f"   grad {mx(D['err_grad'][gm]):.1e} ({gm.sum()} sets)")

print("\ngradients vs eta across the kappa < 0.1 switch, large-time form, max relative error over the five:")
for (a0, t0) in [(1.2, 2.5), (0.8, 3.0)]:
    line = []
    for e in [0.0, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
        p = (t0, 0.7, e, a0, 0.35, 0.65); R = ref(*[mp.mpf(x) for x in p])
        dg = grad_full_sz(*p)[1][:, 0]
        line.append(max(rel(x, r, abs(R)) for x, r in zip(dg, ref_grad(*[mp.mpf(x) for x in p]))))
    print(f"  a = {a0}, t/a^2 = {t0/a0**2:.2f}: " + "  ".join(f"{v:.0e}" for v in line))
print("  (eta = 0, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5, 1, 2)")
