"""
Symbolic verification of Result 1 (density with normal drift and uniform starting point).  Every check is an exact CAS identity
(simplify(lhs - rhs) == 0), not a numerical comparison.

Strategy: an antiderivative claim  int f = F  is verified by checking dF/dx - f = 0
symbolically.  Combined with the boundary/limit arguments stated in the note, this
establishes the definite-integral formulas.
"""
import sympy as sp

w = sp.Symbol('w', real=True)
a, nu, eta = sp.symbols('a nu eta', real=True)
A, B, C, p, q = sp.symbols('A B C p q', real=True)

Phi = lambda z: (1 + sp.erf(z/sp.sqrt(2)))/2          # standard normal CDF
phi = lambda z: sp.exp(-z**2/2)/sp.sqrt(2*sp.pi)
SQ2PI = sp.sqrt(2*sp.pi)

results = {}

# ---------------------------------------------------------------------------
# L1.  Core lemma for Result 1:
#      int (p + q w) exp(-A w^2/2 + B w + C) dw   [A > 0]
# ---------------------------------------------------------------------------
Apos = sp.Symbol('A', positive=True)
m = B/Apos
sig = 1/sp.sqrt(Apos)
K = C + B**2/(2*Apos)
F1 = ((p + q*m)*SQ2PI*sig*sp.exp(K)*Phi((w - m)/sig)
      - (q/Apos)*sp.exp(C + B*w - Apos*w**2/2))
integrand1 = (p + q*w)*sp.exp(-Apos*w**2/2 + B*w + C)
results['L1  linear x Gaussian antiderivative'] = sp.simplify(sp.diff(F1, w) - integrand1)

tp = sp.Symbol('t', positive=True)

# ---------------------------------------------------------------------------
# L4.  The structural claim that makes Result 1 work:
#      the w^2 coefficient of (drift-integrated density term) is NEGATIVE,
#      equal to -a^2/(2 t S) with S = 1 + eta^2 t.
# ---------------------------------------------------------------------------
S = 1 + eta**2*tp
results['L4  w^2 coefficient a^2/(2t) - eta^2 a^2/(2S) = a^2/(2 t S)'] = sp.simplify(
    a**2/(2*tp) - eta**2*a**2/(2*S) - a**2/(2*tp*S))

# ---------------------------------------------------------------------------
# L5.  Coefficient table.  Check that the j-th term of the drift-integrated
#      density, as a function of w, equals  (p_j + q_j w) exp(-A w^2/2 + B_j w + C_j)
#      divided by sqrt(2 pi t^3 S), with the coefficients claimed in the note.
#      r_j = j a + a w (even j);  r_j = (j+1) a - a w (odd j).
# ---------------------------------------------------------------------------
def eta_density_term(rj):
    """r_j * exp(-r_j^2/(2t)) * exp[(-nu^2 t - 2 nu a w + eta^2 (a w)^2)/(2 S)]"""
    return rj*sp.exp(-rj**2/(2*tp))*sp.exp((-nu**2*tp - 2*nu*a*w + eta**2*(a*w)**2)/(2*S))

Acoef = a**2/(tp*S)
checks_L5 = {}
# even j
jj = sp.Symbol('j', integer=True, nonnegative=True)
rj_even = jj*a + a*w
p_e, q_e = jj*a, a
B_e = -nu*a/S - jj*a**2/tp
C_e = -nu**2*tp/(2*S) - (jj*a)**2/(2*tp)
claim_e = (p_e + q_e*w)*sp.exp(-Acoef*w**2/2 + B_e*w + C_e)
checks_L5['L5e even-j coefficients'] = sp.simplify(
    sp.expand(sp.log(sp.simplify(eta_density_term(rj_even)/claim_e))))
# odd j  (k = j+1)
kk = sp.Symbol('k', integer=True, positive=True)
rj_odd = kk*a - a*w
p_o, q_o = kk*a, -a
B_o = -nu*a/S + kk*a**2/tp
C_o = -nu**2*tp/(2*S) - (kk*a)**2/(2*tp)
claim_o = (p_o + q_o*w)*sp.exp(-Acoef*w**2/2 + B_o*w + C_o)
checks_L5['L5o odd-j coefficients'] = sp.simplify(
    sp.expand(sp.log(sp.simplify(eta_density_term(rj_odd)/claim_o))))
results.update(checks_L5)

# ---------------------------------------------------------------------------
# L6.  Blurton Eq.1 (the starting point) is itself the drift-integral of the
#      constant-drift density: verify for a single term by direct integration.
# ---------------------------------------------------------------------------
xs = sp.Symbol('x', real=True)
etap = sp.Symbol('eta', positive=True)
awq = sp.Symbol('aw', real=True)
rq = sp.Symbol('r', positive=True)
const_term = sp.exp(-xs*awq - xs**2*tp/2)*rq*sp.exp(-rq**2/(2*tp))/sp.sqrt(2*sp.pi*tp**3)
normal = sp.exp(-(xs - nu)**2/(2*etap**2))/(etap*sp.sqrt(2*sp.pi))
lhs = sp.integrate(sp.expand(const_term*normal), (xs, -sp.oo, sp.oo))
Sq = 1 + etap**2*tp
rhs = (rq*sp.exp(-rq**2/(2*tp))/sp.sqrt(2*sp.pi*tp**3*Sq)
       * sp.exp((-nu**2*tp - 2*nu*awq + etap**2*awq**2)/(2*Sq)))
results['L6  drift integral reproduces Blurton Eq.1'] = sp.simplify(lhs - rhs)


# ---------------------------------------------------------------------------
# L7.  Gradient of Result 1, small-time term.  ddm_fast._small computes the total
#      differential dI of  I = (p+qm) G - (q/A)(E2-E1)  by hand; check it equals
#      the symbolic derivative of I with respect to each of (A, B, C, p, q, w1, w2).
# ---------------------------------------------------------------------------
w1s, w2s = sp.symbols('w1 w2', real=True)
mA = B/Apos
KA = C + B**2/(2*Apos)
G = SQ2PI/sp.sqrt(Apos)*sp.exp(KA)*(Phi((w2s - mA)*sp.sqrt(Apos)) - Phi((w1s - mA)*sp.sqrt(Apos)))
E1 = sp.exp(C + B*w1s - Apos*w1s**2/2)
E2 = sp.exp(C + B*w2s - Apos*w2s**2/2)
I_closed = (p + q*mA)*G - (q/Apos)*(E2 - E1)
def dI(dA=0, dB=0, dC=0, dp=0, dq=0, dw1=0, dw2=0):          # verbatim from ddm_fast._small
    A_, m = Apos, mA
    dm = (dB - m*dA)/A_
    dK = dC + m*dB - m*m*dA/2
    dG = G*(dK - dA/(2*A_)) + E2*((dw2 - dm) + (w2s-m)*dA/(2*A_)) - E1*((dw1 - dm) + (w1s-m)*dA/(2*A_))
    dE1 = E1*(dC + w1s*dB - w1s*w1s*dA/2 + (B - A_*w1s)*dw1)
    dE2 = E2*(dC + w2s*dB - w2s*w2s*dA/2 + (B - A_*w2s)*dw2)
    return (dp + dq*m + q*dm)*G + (p + q*m)*dG - (dq/A_ - q*dA/(A_*A_))*(E2 - E1) - (q/A_)*(dE2 - dE1)
for name, var, kw in [('A', Apos, dict(dA=1)), ('B', B, dict(dB=1)), ('C', C, dict(dC=1)), ('p', p, dict(dp=1)),
                      ('q', q, dict(dq=1)), ('w1', w1s, dict(dw1=1)), ('w2', w2s, dict(dw2=1))]:
    results[f'L7  small-time gradient: dI/d{name}'] = sp.simplify(sp.diff(I_closed, var) - dI(**kw))

# ---------------------------------------------------------------------------
# L8.  Gradient of Result 1, large-time term.  With e(w) = exp(kappa w^2 + mu w) and
#      F' = e, ddm_fast._large uses
#        int w e dw   = ( e      - mu F ) / (2 kappa)                      (d/dmu)
#        int w^2 e dw = ( w e - F - mu * int w e dw ) / (2 kappa)          (d/dkappa)
#      Checked in differentiated form, as antiderivative identities.
# ---------------------------------------------------------------------------
kap, mu_ = sp.symbols('kappa mu', positive=True)
Fw = sp.Function('F')(w)
e = sp.exp(kap*w**2 + mu_*w)
M1 = (e - mu_*Fw)/(2*kap)
M2 = (w*e - Fw - mu_*M1)/(2*kap)
results['L8a large-time gradient: int w e^{kw^2+mu w}'] = sp.simplify(sp.diff(M1, w).subs(sp.Derivative(Fw, w), e) - w*e)
results['L8b large-time gradient: int w^2 e^{kw^2+mu w}'] = sp.simplify(sp.diff(M2, w).subs(sp.Derivative(Fw, w), e) - w**2*e)

# ---------------------------------------------------------------------------
# L9.  Chain-rule coefficients used in ddm_fast (both regimes) and the log-prefactor
#      derivatives, against symbolic differentiation of the definitions.
# ---------------------------------------------------------------------------
ap, ep, kp = sp.symbols('a eta k', positive=True)
sjs = sp.Symbol('s_j')
Sd = 1 + ep**2*tp
A_d = ap**2/(tp*Sd); B_d = -nu*ap/Sd + sjs*kp*ap**2/tp; C_d = -nu**2*tp/(2*Sd) - (kp*ap)**2/(2*tp)
P_small = 1/((w2s - w1s)*sp.sqrt(2*sp.pi*tp**3*Sd))
kap_d = ep**2*ap**2/(2*Sd); lam_d = -nu*ap/Sd
P_large = sp.pi/ap**2/sp.sqrt(Sd)*sp.exp(-nu**2*tp/(2*Sd))/(w2s - w1s)
coded = {  # (expression, variable, coefficient as written in ddm_fast)
 'dB/dnu':     (B_d, nu, -ap/Sd),            'dC/dnu':  (C_d, nu, -nu*tp/Sd),
 'dA/deta':    (A_d, ep, -2*ap*ap*ep/Sd**2),  'dB/deta': (B_d, ep, 2*nu*ap*ep*tp/Sd**2),
 'dC/deta':    (C_d, ep, nu**2*ep*tp*tp/Sd**2),
 'dA/da':      (A_d, ap, 2*ap/(tp*Sd)),       'dB/da':   (B_d, ap, -nu/Sd + 2*sjs*kp*ap/tp),
 'dC/da':      (C_d, ap, -kp*kp*ap/tp),
 'dlogP/deta (small)': (sp.log(P_small), ep, -ep*tp/Sd),
 'dlogP/dw1 (small)':  (sp.log(P_small), w1s, 1/(w2s-w1s)),
 'dkappa/deta': (kap_d, ep, ep*ap*ap/Sd**2),  'dkappa/da': (kap_d, ap, ep**2*ap/Sd),
 'dlam/dnu':   (lam_d, nu, -ap/Sd),           'dlam/deta': (lam_d, ep, 2*nu*ap*ep*tp/Sd**2),
 'dlam/da':    (lam_d, ap, -nu/Sd),
 'dlogP/dnu (large)':  (sp.log(P_large), nu, -nu*tp/Sd),
 'dlogP/deta (large)': (sp.log(P_large), ep, -ep*tp/Sd + nu**2*ep*tp**2/Sd**2),
 'dlogP/da (large)':   (sp.log(P_large), ap, -2/ap),
}
for name, (expr, var, coef) in coded.items():
    results[f'L9  chain rule {name}'] = sp.simplify(sp.diff(expr, var) - coef)

# ---------------------------------------------------------------------------
if __name__ == '__main__':
    ok = True
    for name, res in results.items():
        z = sp.simplify(res)
        good = (z == 0)
        ok &= bool(good)
        print(f"{'PASS' if good else 'FAIL'}  {name}" + ("" if good else f"   residual: {z}"))
    print("\nALL SYMBOLIC CHECKS PASS" if ok else "\nSOME CHECKS FAILED")
