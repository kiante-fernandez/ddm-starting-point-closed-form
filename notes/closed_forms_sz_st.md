# A closed form for the Ratcliff diffusion density with drift and starting-point variability

Working note, September 2026. Every formula below is verified symbolically
(`verify/symbolic_proof.py`), to 30–48 digits (`verify/highprec2.py`), against WienR and
rtdists (`data/`), against process simulation (`verify/validate_sim2.py`), and is
machine-checked in Lean 4 (`RequestProject/`, see `FORMALIZATION.md`).

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

### Timing (same machine, single thread, 10 000 RTs at one parameter set)

Each package at the settings its fitting routine uses. Error is the maximum relative deviation
from the closed form on the 600 reference parameter sets (densities > 1e-3).

| µs per evaluation | 6p error | 6p µs | 7p error | 7p µs |
|---|---|---|---|---|
| closed form, compiled (`ddm_kernel.pyx`) | — | 0.33 | 1.5e-9 (quadrature) | 9.2 |
| WienR, tolerance 1e-12 | 2.5e-12 | 2.1 | 1.5e-9 | 1040 |
| rtdists / fast-dm, precision 3 | 4.6e-3 | 0.43 | 9.2e-3 | 18 |
| HDDM, series 1e-4, Simpson 1e-8, depth 10 | 3.9e-6 | 0.60 | 1.5e-7 | 46 |
| closed form + five gradients (numpy) | — | 1.7 | | |
| WienR, five gradients | 1e-13 | 15.7 | | |

Tolerance benchmarks for reading the error columns: Blurton et al. (2017) recommend 1.5e-8;
Gondan et al. (2014) illustrate at 1e-3; Tuerlinckx (2004, Table 1) shows 2e-7 to 3e-4 between
methods; rtdists' precision 3 is "roughly the number of accurate decimals"; Stan (Henrich et
al. 2024) fixes the density at 1e-6; WienR defaults to 1e-12. WienR's own closed form for drift
variability alone runs at 0.26 µs; the kernel is 1.3× that. The kernel is the series of
`ddm_fast.py` in Cython with the Gaussian-moment term written through `erfcx` (Mills ratio)
instead of log-space `log_ndtr`, calling scipy's `erfcx`, `ndtr`, and `wofz`; it reproduces
numpy to 1.5e-14. Scripts: `verify/timing.R`, `verify/timing.py`,
`verify/hddm_wfpt_compare.py`.

---

## Verification

### Symbolic (`verify/symbolic_proof.py`, SymPy, exact CAS identities)

Each claim is verified as an exact symbolic zero, not numerically. All 32 pass:

| check | content |
|---|---|
| L1 | ∫(p+qw)e^{−Aw²/2+Bw+C}dw antiderivative (core of Result 1) |
| L4 | a²/(2t) − η²a²/(2S) = a²/(2tS) > 0 — the inequality the result rests on |
| L5e, L5o | the even-j and odd-j coefficient tables |
| L6 | the drift integral of the constant-drift density reproduces Blurton Eq. 1 |
| L7 (×7) | the hand-derived total differential of I equals ∂I/∂(A, B, C, p, q, w₁, w₂) |
| L8a, L8b | the large-time moment identities ∫w e^{κw²+μw}dw and ∫w² e^{κw²+μw}dw |
| L9 (×18) | every chain-rule coefficient and log-prefactor derivative used in the gradient code |

### Arbitrary precision (`verify/highprec2.py`, mpmath)

* Small-time form vs a brute-force double integral over drift and starting point, 40 dps:
  8.7e-32 (t = 0.30, ν = 1, η = 1.2, a = 1.2, w ∈ [0.25, 0.75]).
* Small-time form vs w-quadrature of (gη), 40 dps: 4.3e-42 to 6.5e-33 within its regime.
* Large-time form, 60 dps with adaptive truncation: 7.1e-48, 1.6e-35, 4.8e-42 at
  t/a² = 0.89, 4.69, 1.50.

### Against WienR (Hartmann & Klauer 2021)

WienR called with `precision = 1e-12`, `n.evals = 0`. Same parametrization as here.
Scripts `verify/wienr_grid.R`, `verify/wienr_full.R`, `verify/wienr_grad.R`; data
`data/wienr_grid.csv`, `data/wienr_full.csv`, `data/wienr_grad.csv`.

**Six-parameter density, 400 random sets**, a ∈ {0.6, 1, 1.5, 2.5}, ν ∈ [−3, 4],
η ∈ {0.5, 1, 2}, s_w ∈ {0.05, 0.2, 0.4}, t ∈ [0.05, 3]; densities spanning 5e-20 to 1.6.
Max relative difference from WienR:

| representation | t/a² < 0.5 | 0.5–1 | 1–2 | 2–4 | 4–9 |
|---|---|---|---|---|---|
| small-time series | 2.4e-10 | 1.0e-12 | 6.7e-10 | 8.7e-7 | fails (cancellation) |
| large-time series (Faddeeva) | fails | 1.9e-13 | 3.9e-13 | 3.5e-13 | 7.9e-14 |
| switched at t/a² = 1 | 2.4e-10 | 1.0e-12 | 3.9e-13 | 3.5e-13 | 7.9e-14 |

Max absolute difference of the switched version over all 400 points is 1.2e-14, below
WienR's own 1e-12 tolerance. The 2.4e-10 relative entry is at a density of 7e-9.

**Seven-parameter density, 200 random sets**, t₀ = 0.3, s_t ∈ {0.1, 0.25}: max absolute
difference 2.5e-9, max relative 7e-7, which is WienR's reported cubature error.

### Against rtdists / fast-dm

rtdists 0.12-0 (its diffusion engine is Voss & Voss's fast-dm C++ code). Parametrization
converted: `z = a·w`, `sz = a·s_w`, `s = 1`. Scripts `verify/rtdists_prec.R`,
`verify/rtdists_full.R`; data `data/rtdists_prec.csv`, `data/rtdists_full.csv`. Same
400-point grid, max absolute difference from the closed form: 7.6e-4 at precision 3
(default), 1.1e-5 at 5, 2.5e-8 at 8. rtdists converges monotonically onto the closed form as
its precision is raised; the closed form and WienR agree with each other (1e-14) far more
closely than either agrees with rtdists.

### Against the process (`verify/validate_sim2.py`)

Euler–Maruyama simulation, 400 000 trials, dt = 5e-5, Broadie–Glasserman–Kou barrier
correction 0.5826√dt (barrier placed *inside* the continuous one), v ~ N(1, 1.2²),
w ~ U(0.25, 0.75), a = 1.2. Lower-barrier RT histogram in 30 bins of 100 ms vs binned
closed-form density: χ² = 21.6 on 30 bins, all |z| < 1.9; P(lower) 0.3073 (sim) vs 0.3065
(formula), 1.1 SE.

### Independent scalar implementation (`src/ddm_closed.py`)

Result 1 vs adaptive quadrature over w of (gη): 6.7e-15 over 40 random sets; vs a true double
integral over (v, w) of (g): 1e-16. Agrees with `ddm_fast.py` to 3e-16.

### Lean 4

Result 1, Lemmas L1, L4–L6, the summation/integration exchange, and the Faddeeva
starting-point integral are proved in `RequestProject/DDM/` with no `sorry` and only Lean's
standard axioms. See `FORMALIZATION.md` for what is taken as a definition.

---

## Future work

The corresponding **cumulative** distribution with η and s_z does not follow by the same
route: after the drift integral, half of the series terms carry a w²-coefficient of exactly
+2a²η² (a growing Gaussian times Φ) and reduce to an Owen-T-type integral with no elementary
form; the other half close via ∫ e^{cw} Φ(αw + β) dw. For maximum-likelihood and Bayesian
fitting the density suffices, so this is left open.
