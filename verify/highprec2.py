"""High-precision check (40 dps; second block re-runs the large-time form at 60 dps of both representations over their regimes.
Reference: numerical w-quadrature of Blurton Eq.1, which symbolic_proof.py L6
verified exactly equals the drift integral of the constant-drift density."""
import mpmath as mp
mp.mp.dps = 40
def rj(j,a,w): return j*a+a*w if j%2==0 else (j+1)*a-a*w
def g_eta(t,nu,eta,a,w,J=40):
    S=1+eta**2*t
    s=sum((-1)**j*rj(j,a,w)*mp.e**(-rj(j,a,w)**2/(2*t)) for j in range(J))
    return mp.e**((-nu**2*t-2*nu*a*w+eta**2*(a*w)**2)/(2*S))*s/mp.sqrt(2*mp.pi*t**3*S)
def ref(t,nu,eta,a,w1,w2,J=40): return mp.quad(lambda w: g_eta(t,nu,eta,a,w,J),[w1,w2])/(w2-w1)
def closed_small(t,nu,eta,a,w1,w2,J=40):
    S=1+eta**2*t; A=a**2/(t*S); tot=mp.mpf(0)
    for j in range(J):
        e=(j%2==0); k=j if e else j+1; p=k*a; q=a if e else -a
        B=-nu*a/S+(-1 if e else 1)*k*a**2/t; C=-nu**2*t/(2*S)-(k*a)**2/(2*t)
        m=B/A; sig=1/mp.sqrt(A); K=C+B**2/(2*A)
        tot+=(-1)**j*((p+q*m)*mp.sqrt(2*mp.pi)*sig*mp.e**K*(mp.ncdf((w2-m)/sig)-mp.ncdf((w1-m)/sig))
                      -(q/A)*(mp.e**(C+B*w2-A*w2**2/2)-mp.e**(C+B*w1-A*w1**2/2)))
    return tot/((w2-w1)*mp.sqrt(2*mp.pi*t**3*S))
def closed_large(t,nu,eta,a,w1,w2,K=None,tol=mp.mpf(10)**-45):
    """large-time form via erfi; K terms if given (shows the fixed-K blow-up), else stop when the
    k-th damping factor is below tol.  Returns (value, terms used)."""
    S=1+eta**2*t; kap=eta**2*a**2/(2*S); lam=-nu*a/S; sk=mp.sqrt(kap); tot=mp.mpf(0); k=1
    while (k<=K) if K else True:
        damp=mp.e**(-k**2*mp.pi**2*t/(2*a**2))
        if K is None and damp<tol and k>3: break
        mu=lam+1j*mp.pi*k
        pref=mp.sqrt(mp.pi)/(2*sk)*mp.e**(-mu**2/(4*kap))
        tot+=k*damp*mp.im(pref*(mp.erfi(sk*w2+mu/(2*sk))-mp.erfi(sk*w1+mu/(2*sk)))); k+=1
    return mp.pi/a**2/mp.sqrt(S)*mp.e**(-nu**2*t/(2*S))*tot/(w2-w1), k-1
cases=[(0.30,1.0,1.2,1.2,0.25,0.75),(0.90,-0.5,0.8,1.0,0.40,0.60),
       (0.15,3.0,2.0,2.5,0.30,0.70),(2.00,0.0,1.5,1.5,0.45,0.55),
       (0.60,2.0,0.3,0.8,0.20,0.80),(3.00,-1.0,1.0,0.8,0.35,0.65)]
print(f"{'t':>5}{'t/a^2':>7} {'closed (small-time)':>26} {'|small-ref|':>11} {'|large-ref|':>11}")
for c in cases:
    t,nu,eta,a,w1,w2=map(mp.mpf,c); R=ref(t,nu,eta,a,w1,w2)
    cs=closed_small(t,nu,eta,a,w1,w2); cl,_=closed_large(t,nu,eta,a,w1,w2,K=120)
    print(f"{float(t):5.2f}{float(t/a**2):7.2f} {mp.nstr(cs,22):>26} {mp.nstr(abs(cs-R),3):>11} {mp.nstr(abs(cl-R),3):>11}")

# --- large-time form at 60 dps, truncated adaptively where the k-th damping factor < 1e-45
mp.mp.dps = 60
print()
for c in [(2.00,0.0,1.5,1.5,0.45,0.55),(3.00,-1.0,1.0,0.8,0.35,0.65),
          (0.90,-0.5,0.8,1.0,0.40,0.60),(1.50,2.0,1.0,1.0,0.30,0.70)]:
    t,nu,eta,a,w1,w2=map(mp.mpf,c)
    R=ref(t,nu,eta,a,w1,w2,J=60); cl,K=closed_large(t,nu,eta,a,w1,w2)
    print(f"t={float(t):4.2f} t/a^2={float(t/a**2):5.2f}  K={K:3d}  large-time={mp.nstr(cl,20):>24}  |large-ref|={mp.nstr(abs(cl-R),3)}")

# --- small-Delta branch (w2 - w1 < 1e-2, ddm_fast._avg): density and all five gradients vs a 40-dps
#     reference (quadrature of g_eta over [w1, w2]; gradients by central differences with h = 1e-12)
import sys, numpy as np
sys.path.insert(0, "src")
from ddm_fast import grad_full_sz
mp.mp.dps = 40
def gbar(t, nu, eta, a, wb, D):
    return g_eta(t, nu, eta, a, wb, J=60) if D == 0 else mp.quad(lambda w: g_eta(t, nu, eta, a, w, J=60), [wb - D/2, wb + D/2])/D
def ref6(t, nu, eta, a, wb, D):
    h = mp.mpf("1e-12"); P = dict(t=t, nu=nu, eta=eta, a=a, wb=wb, D=D)
    d = lambda k: (gbar(**{**P, k: P[k] + h}) - gbar(**{**P, k: P[k] - h}))/(2*h)
    dwb, dD = d("wb"), (0 if D == 0 else d("D"))
    return [float(x) for x in (gbar(**P), d("nu"), d("eta"), d("a"), dwb/2 - dD, dwb/2 + dD)]
print("\nsmall-Delta branch vs 40-dps reference: worst rel err over density and five gradients")
worst = 0
for (t, nu, eta, a, wb) in [(0.5, 1, 1, 1, 0.5), (2.5, 1, 1, 1, 0.5), (0.05, -3, 2, 2.5, 0.3), (3, 4, 2, 0.6, 0.7)]:
    row = []
    for D in (2e-2, 1e-2, 1e-3, 1e-6, 0.0):
        R = np.array(ref6(*map(mp.mpf, (t, nu, eta, a, wb, D))))
        g, dg = grad_full_sz(np.array([t]), nu, eta, a, wb - D/2, wb + D/2)
        e = np.max(np.abs(np.concatenate([g, dg[:, 0]]) - R)/np.maximum(np.abs(R), abs(R[0])))
        worst = max(worst, e); row.append(f"Delta={D:.0e}: {e:.0e}")
    print(f"  t={t} nu={nu} eta={eta} a={a}:  " + "  ".join(row))
print(f"worst {worst:.1e}"); assert worst < 1e-11
print("SMALL-DELTA CHECK PASS")
