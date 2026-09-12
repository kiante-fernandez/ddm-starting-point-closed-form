# Status

Every number below was recomputed from the code and data in this repo (Sept 2026). Re-run:
`python verify/symbolic_proof.py`, `python verify/grad_check.py`, `Rscript verify/timing.R`,
`python verify/timing.py`, `python verify/hddm_wfpt_compare.py`, `lake build`. Build the compiled kernel first with `cythonize -3 -i src/ddm_kernel.pyx`.

## Done

**Result 1 — density with normal drift `eta` AND uniform starting point `s_z`, closed form.**
`src/ddm_fast.py`: `g_eta_sz` (small-time series), `g_eta_sz_large` (large-time series via
Faddeeva `w(z)`), `g_full_sz` (switch at `t/a² = 1`; the two forms agree to 1.5e-11 relative
across the switch), `grad_full_sz` (density plus its closed-form gradient in
`nu, eta, a, w1, w2`, both regimes, `upper=True` for the upper barrier). Seven-parameter
density `f7` / `grad_f7` = Result 1 + one 32-node Gauss–Legendre rule over the `s_t` window with
the cubic edge map; `grad_f7` adds `∂/∂t0`, `∂/∂st0` from the window endpoints. Series lengths
`J`, `K` are chosen from a tolerance (`tol=1e-12` default) via the term bounds
`exp(−J²a²/2t)` and `exp(−K²π²t/2a²)`.

| check | result |
|---|---|
| SymPy exact identities L1, L4–L9 (`verify/symbolic_proof.py`): closed form, coefficient tables, drift integral, and every gradient formula (small-time total differential, large-time moment identities, all chain-rule coefficients) | 32/32 pass |
| mpmath vs brute-force `(v, w)` double integral, 40–60 dps (`verify/highprec2.py`) | small-time 8.7e-32; large-time 7e-48, 2e-35, 5e-42 |
| WienR `precision=1e-12`, 400 sets (`data/wienr_grid.csv`) | max abs 1.2e-14, max rel 2.4e-10 |
| WienR seven-parameter, 200 sets (`data/wienr_full.csv`) | max abs 2.5e-9, max rel 7e-7 (WienR's own cubature error) |
| rtdists / fast-dm precision 3 → 5 → 8 (`data/rtdists_prec.csv`) | 7.6e-4 → 1.1e-5 → 2.5e-8: converges onto our value |
| hddm-wfpt (HDDM / HSSM) at its fitting settings (`verify/hddm_wfpt_compare.py`) | 3.9e-6 rel (6p), 1.5e-7 rel (7p) |
| Euler–Maruyama 400k trials (`verify/validate_sim2.py`) | χ² = 21.6 on 30 bins |
| Lean 4 / Mathlib (`RequestProject/DDM/Result1.lean`, `theorem result1`) | proved, no `sorry`, standard axioms only |
| compiled kernel vs numpy, 400 + 200 sets (`grad_check.py`) | max abs 1.5e-14 / 1.8e-15 |
| `grad_full_sz` vs WienR `d{v,sv,a,w,sw}WienerPDF`, 400 sets (`verify/grad_check.py`) | max abs ≤ 2.9e-13 on all five; max rel ≤ 5e-10 except `dsw` 1.1e-8 (WienR's own 1e-14 error floor); vs finite differences ≤ 3e-7 small-time, ≤ 2e-9 large-time |
| `grad_f7`, all 7 gradients vs finite differences, clamped and unclamped window | ≤ 3e-11 rel |
| truncation rule `tol=1e-12` vs `tol=1e-40`, 400 sets | max abs 2.2e-16 |
| seven-parameter quadrature vs adaptive reference, 200 sets (`grad_check.py`): 32 nodes + cubic map / 32 plain / 16 plain / 8 plain | 1.5e-9 / 1.1e-8 / 7.3e-5 / 4.1e-3 rel |

**Timing, same machine (Apple laptop, single thread), 10 000 RTs at one parameter set
(`a=1.2, v=1, w=0.5, sv=1, sw=0.2, st0=0.2`), µs per evaluation. Each package at the settings it
fits with. Error = max relative deviation from the closed form on the 400 + 200 sets (densities
> 1e-3). Regenerate with `Rscript verify/timing.R`, `OMP_NUM_THREADS=1 python verify/timing.py`,
`python verify/hddm_wfpt_compare.py`:**

| | 6p error | 6p µs | 7p error | 7p µs |
|---|---|---|---|---|
| closed form, compiled (`src/ddm_kernel.pyx`) | — | 0.33 | 1.5e-9 (quadrature) | 9.2 |
| WienR (tol 1e-12) | 2.5e-12 | 2.1 | 1.5e-9 | 1040 |
| rtdists / fast-dm (precision 3) | 4.6e-3 | 0.43 | 9.2e-3 | 18 |
| HDDM `wiener_like` (series 1e-4, Simpson 1e-8, depth 10) | 3.9e-6 | 0.60 | 1.5e-7 | 46 |
| closed form + 5 gradients (numpy, uncompiled) | — | 1.7 | | |
| WienR, 5 gradients (5 cubatures) | 1e-13 | 15.7 | | |

WienR's own closed form (`sv` only, Blurton et al. 2017) runs at 0.26 µs; the kernel is 1.3× that.
The kernel writes the Gaussian-moment term with `erfcx` (Mills ratio) rather than log-space
`log_ndtr` (2 erfcx ≈ 46 ns vs 2 log_ndtr + log1p ≈ 70 ns per term) and drops the `+1` margin
on `K`; it reproduces numpy to 1.5e-14 and a `tol=1e-40` run to 4e-17.

**Tolerance benchmarks in the literature and packages** (for reading the error columns):
Blurton et al. 2017 recommend 1.5e-8 (√ single-precision ε); Gondan et al. 2014 illustrate at
1e-3; Tuerlinckx 2004 Table 1 shows 2e-7 to 3e-4 between methods; rtdists/fast-dm default
precision 3 ≈ 3 decimals; HDDM series 1e-4 + Simpson 1e-8/depth 10; Stan (Henrich et al. 2024) density fixed at 1e-6, derivatives default 1e-4;
WienR default 1e-12.

**Lean coverage** (`FORMALIZATION.md` has the file map): Result 1, Lemmas L1, L4–L6, the
summation/integration exchange, and the Faddeeva starting-point integral. Not formalized: the
literature starting representations (taken as definitions), the large-time series itself,
boundedness of `w(z)`, the gradient formulas.

## Not done (in priority order)

1. The manuscript. The note is the skeleton.
2. `git init` and an archived snapshot (Zenodo/OSF) for the supplementary material.
3. Gradients in the compiled kernel (currently numpy only; already 10× WienR).

## Future work (not for this paper)

The CDF with `eta` and `s_z`. Half its series closes by the elementary lemma
`∫ e^{cw} Φ(αw+β) dw`; the other half has `w²`-coefficient exactly `+2a²eta²` after the drift
integral and reduces to an Owen-T-type integral with no elementary form. One paragraph in the
note; nothing else needed, since the density covers likelihood-based fitting.

## Ruled out — do not retry

- Swapping the `s_t` distribution for analytic convenience. `T_er` enters as a shift inside
  `√t` and `1/t` in every term; only exponential `T_er` closes, and that is the shape Ratcliff
  (2013, *Psych Review* 120, 281–292) showed breaks recovery.
- Large-time form written with `erfi` instead of `w(z)`: loses `~π²k²/(4κ)` nats to
  cancellation (`highprec2.py` first block shows it blowing up at fixed `K`).
- Plain 8–20 node Gauss–Legendre over the `s_t` window: the leading edge breaks it.

## Ground rules

1. Verify before claiming: symbolic identity, arbitrary precision, independent implementation.
   Several plausible intermediate results failed one of these.
2. State evidence and gaps separately.
3. Numerical agreement is not proof; `simplify(dF/dx − f) == 0` is.
4. Watch cancellation. Two results were wrong before log-space / Faddeeva evaluation and looked
   fine on a coarse grid.
