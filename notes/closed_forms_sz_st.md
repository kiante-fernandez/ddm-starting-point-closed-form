# A closed form for the Ratcliff diffusion density with drift and starting-point variability

Working note, September 2026. Every formula below is verified symbolically
(`verify/symbolic_proof.py`), to 30–48 digits (`verify/highprec2.py`), against WienR and
rtdists (`data/`), against process simulation (`verify/validate_sim2.py`), and is
machine-checked in Lean 4 (`RequestProject/`, see the README).

## Setup

Wiener process, unit diffusion coefficient, absorbing barriers at 0 and a, start a·w, drift v.
Lower-barrier quantities throughout; the upper barrier follows from (v, w) → (−v, 1−w). Define

    r_j = j a + a w          (j even)
    r_j = (j+1) a − a w      (j odd)

Constant-drift density (small-time series):

    g(t | v, a, w) = (2π t³)^{−1/2} e^{−vaw − v²t/2} Σ_j (−1)^j r_j e^{−r_j²/(2t)}                          (g)

Normal drift N(ν, η²), S := 1 + η² t (Horrocks & Thompson 2004; Blurton et al. 2017 Eq. 1):

    g(t | ν, η, a, w) = (2π t³ S)^{−1/2} exp[(−ν²t − 2νaw + η²a²w²)/(2S)] Σ_j (−1)^j r_j e^{−r_j²/(2t)}     (gη)

The j-th term is O(exp(−j²a²/(2t))), so summation and integration may be exchanged
(dominated convergence; proved in `Exchange.lean`).

---

## Result 1 — Density with normal drift AND uniform starting point (closed form)

Starting point uniform on [w₁, w₂] (Ratcliff's s_z = a(w₂ − w₁)). Then

    g(t | ν, η, a, [w₁,w₂]) = 1 / ((w₂ − w₁) √(2π t³ S)) · Σ_j (−1)^j I_j

where I_j = ∫_{w₁}^{w₂} (p_j + q_j w) exp(−A w²/2 + B_j w + C_j) dw with

    A = a² / (t S)                                   (same for all j, always > 0)

    even j:  p_j = j a,      q_j =  a,   B_j = −νa/S − j a²/t,        C_j = −ν²t/(2S) − j²a²/(2t)
    odd  j:  p_j = (j+1) a,  q_j = −a,   B_j = −νa/S + (j+1) a²/t,    C_j = −ν²t/(2S) − (j+1)²a²/(2t)

and, with m = B/A,

    I = e^{C + B²/(2A)} (p + q m) √(2π/A) [ Φ(√A (w₂ − m)) − Φ(√A (w₁ − m)) ]
        − (q/A) [ e^{C + B w₂ − A w₂²/2} − e^{C + B w₁ − A w₁²/2} ]

**Why it works.** In (gη) the drift has already been integrated out, and the only w-dependence
of the j-th term is the prefactor exp[(−2νaw + η²a²w²)/(2S)] times r_j φ(r_j/√t), where r_j is
linear in w. The η² term in the prefactor is a *growing* Gaussian in w (coefficient
+η²a²/(2S)), but the φ(r_j/√t) factor contributes −a²/(2t), and

    a²/(2t) − η²a²/(2S) = a² (S − η²t) / (2tS) = a² / (2tS) > 0,

so the combined exponent is a proper (decaying) Gaussian for every j and every t. Linear ×
Gaussian integrates to Φ and exp terms.

**Why it was missed.** Tuerlinckx (2004, *BRMIC* 36, 702–716) worked in the large-time
(Ratcliff 1978) series, where w enters through sin(kπw) and there is no φ(r_j/√t) factor. There
the drift integral leaves a growing Gaussian in w and the w-integral lands on erfi; he
concluded in print that the starting-point integral "must be done numerically because there
are no closed-form solutions available." That is correct for his representation and wrong for
the small-time one. HDDM (Wiecki et al. 2013), WienR (Hartmann & Klauer 2021), the Stan
implementation (Henrich et al. 2024), and fast-dm / rtdists (`src/Density.h`:
`integrate_v_over_zr`, adaptive Simpson) all integrate s_z numerically.

**Numerical note.** Evaluate e^{C+B²/(2A)}·[Φ − Φ] in log space with tail-aware differences
(`_logPhi_diff` in `ddm_fast.py`); for large j the Gaussian centre m sits far outside
[w₁, w₂] and naive evaluation overflows or cancels.

### Large-time representation

The small-time series loses precision to cancellation for t/a² ≳ 2. With κ = η²a²/(2S),
λ = −νa/S, μ_k = λ + ikπ, z_k(w) = √κ w + μ_k/(2√κ), and w(z) the Faddeeva function,

    g(t) = π/(a² √S (w₂−w₁)) · e^{−ν²t/(2S)} Σ_k k e^{−k²π²t/(2a²)}
           · Im{ (−i)(√π/(2√κ)) [ e^{κw₂²+μ_k w₂} w(z_k(w₂)) − e^{κw₁²+μ_k w₁} w(z_k(w₁)) ] }

Because Im z_k > 0, w(z) is bounded and no overflow occurs even though the w-integral is
formally an erfi of complex argument; this is what Tuerlinckx's route was missing. For η = 0
the bracket reduces to (e^{μw₂} − e^{μw₁})/μ. Written with `erfi` instead of `w(z)` the form
loses roughly π²k²/(4κ) nats to cancellation and returns garbage at fixed large K
(`highprec2.py`, first block). Switch between the two representations at t/a² = 1
(`g_full_sz`); they agree to 1.5e-11 relative across the switch.

### Gradient

Every term is exp, Φ, and (large-time) w(z), so ∂g/∂θ for θ ∈ {ν, η, a, w₁, w₂} is closed
form. Implementation: `grad_full_sz` in `ddm_fast.py`. Small-time: one total differential of
I_j in its seven coefficients (A, B, C, p, q, w₁, w₂), then a chain-rule line per parameter,
using e^{K} φ(√A(w_i − m)) √(2π) = e^{C + B w_i − A w_i²/2} to keep everything in the stable
pieces. Large-time: ∂/∂μ ∫ e^{κw²+μw} dw = [(E₂ − E₁) − μ I]/(2κ) and
∂/∂κ ∫ e^{κw²+μw} dw = [(w₂E₂ − w₁E₁) − I − μ ∂I/∂μ]/(2κ), with E_i = e^{κw_i²+μw_i}.
∂/∂w₂ and ∂/∂w₁ are the integrand at the endpoints. Checked (`verify/grad_check.py`) against
central finite differences in each regime (≤ 3e-7 small-time, ≤ 2e-9 large-time) and against
WienR's `d{v,sv,a,w,sw}WienerPDF` on 400 parameter sets (≤ 2.9e-13 absolute on all five).

### Seven-parameter likelihood

With uniform T_er on [t₀, t₀ + s_t] (WienR convention),

    f_RT(t) = (1/s_t) ∫_{t − t₀ − s_t}^{t − t₀} g(u | ν, η, a, [w₁,w₂]) du,

a single 1-D integral of a closed-form smooth function over a short interval, versus nested
numerical integrals over s_z and s_t in current software. This integral is not removable:
s_t shifts the time argument inside √t and 1/t in every term. Gauss–Legendre after the
map u = lo + (hi − lo)s³ (clusters nodes at the leading edge); max relative error against
adaptive quadrature over the 200 seven-parameter sets (`grad_check.py`):

| nodes | plain Gauss–Legendre | with cubic edge map |
|---|---|---|
| 8 | 4.1e-3 | 4.3e-3 |
| 16 | 7.3e-5 | 3.5e-6 |
| 24 | 4.1e-6 | 2.7e-8 |
| 32 | 1.1e-8 | 1.5e-9 |

The leading edge (density rising from zero at u = 0) is what defeats the small rules; 32 mapped
nodes is the fixed rule used (`f7` in `ddm_fast.py`). Its
gradient in all seven parameters is the same rule applied to the gradient rows, plus
∂f/∂t₀ = [g(lo) − g(hi)]/s_t and ∂f/∂s_t = [g(lo) − f]/s_t from the window endpoints
(`grad_f7`; checked against finite differences to 3e-11, including the case where the window
is clamped at u = 0).

### Truncation

The j-th small-time term is bounded by (j+1)·a·exp(−j²a²/(2t)) (proved in `Exchange.lean`),
the k-th large-time term by k·exp(−k²π²t/(2a²)). For a tolerance τ the series are cut at

    J = ⌈ √(2 t log(1/τ)) / a ⌉ + 1,      K = ⌈ a √(2 log(1/τ) / (π² t)) ⌉ + 1,

evaluated at the largest (J) or smallest (K) t in the batch. With τ = 1e-12 and t/a² ≤ 1 this
gives J ≤ 9; with t/a² ≥ 1 it gives K ≤ 4. Against τ = 1e-40 on the 400-set grid the difference
is 2.2e-16.

### Verification and timing

All checks, reference comparisons, and timings, with the numbers and the scripts that
regenerate them, are in the manuscript, `paper/main.tex`, Section 6.

---

## Future work

The corresponding **cumulative** distribution with η and s_z does not follow by the same
route: after the drift integral, half of the series terms carry a w²-coefficient of exactly
+2a²η² (a growing Gaussian times Φ) and reduce to an Owen-T-type integral with no elementary
form; the other half close via ∫ e^{cw} Φ(αw + β) dw. For maximum-likelihood and Bayesian
fitting the density suffices, so this is left open.
