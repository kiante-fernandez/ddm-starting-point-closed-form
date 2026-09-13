# Closed forms for across-trial variability in the Ratcliff diffusion model

**Result.** A closed-form first-passage density for the Ratcliff diffusion model with normally
distributed drift AND uniformly distributed starting point (Result 1), with closed-form
gradients in all five parameters. Blurton, Kesselmeier & Gondan (2017) removed the drift
integral; this removes the starting-point integral. The seven-parameter likelihood then needs
one fixed 1-D quadrature over the non-decision-time window instead of 2-D cubature. Verified
symbolically, to 48 digits, against WienR and rtdists, against simulation, and machine-checked
in Lean 4.

## Read these in order

1. `paper/main.tex` — the manuscript.
2. `notes/closed_forms_sz_st.md` — the working note the Lean files cite.
3. The Lean formalization section below.

## Layout

```
src/       ddm_fast.py     vectorized Result 1, gradient, seven-parameter density f7 (use this one)
           ddm_kernel.pyx  compiled density kernel (same arithmetic; build with cythonize)
           ddm_closed.py   scalar reference implementation, kept independent on purpose
verify/    symbolic_proof.py    SymPy: exact verification of every algebraic step
           highprec2.py         mpmath: 40–60 digit verification
           validate_sim2.py     Euler–Maruyama process simulation check
           grad_check.py        gradient in (nu, eta, a, w1, w2) vs finite differences and WienR
           hddm_wfpt_compare.py accuracy/cost vs hddm-wfpt (pip install hddm-wfpt; optional)
           timing.R, timing.py  the timing table of the manuscript (WienR / closed form)
           *.R                  WienR and rtdists reference-value generation
data/      *.csv           per-row comparison data (regenerate with the .R scripts)
notes/     the mathematical note
RequestProject/DDM/*.lean   Lean formalization (see below)
```

## Quick start

```bash
pip install -r requirements.txt
python verify/symbolic_proof.py     # should print ALL SYMBOLIC CHECKS PASS
python verify/highprec2.py          # slow (minutes); background it
PYTHONPATH=src python verify/validate_sim2.py   # slow
python verify/grad_check.py         # GRAD CHECK PASS ... ALL CHECKS PASS
cythonize -3 -i src/ddm_kernel.pyx    # optional compiled kernel; needs Cython + a C compiler
lake exe cache get && lake build    # Lean; needs elan/lake, Mathlib v4.28.0
```

## Conventions used throughout

Unit diffusion coefficient (s = 1). Barriers at 0 and `a`. **Relative** starting point
`w in (0,1)`, absolute starting point `a*w`. Lower barrier throughout; the upper barrier is
`(v, w) -> (-v, 1-w)`. Drift `v ~ N(nu, eta^2)` (`eta` = `sv` = `η`). Starting point uniform on
`[w1, w2]`, so Ratcliff's `s_z = a*(w2-w1)`. `S := 1 + eta^2 * t`.

Converting to other packages: Ratcliff's papers use s = 0.1, so divide `a`, `v`, `eta` by 10.
WienR uses this same relative-`w` convention. rtdists uses **absolute** `z = a*w` and
`sz = a*(w2-w1)`.

## R reference implementations (only for regenerating `data/`)

```r
install.packages(c("WienR", "rtdists"))   # WienR: Hartmann & Klauer; rtdists' diffusion engine is fast-dm's C++
```

Scripts use relative CSV paths, so run from `data/`:
`cd data && Rscript ../verify/wienr_grid.R && Rscript ../verify/rtdists_prec.R` (and the
`*_full.R` pair for the seven-parameter set).

Gotchas that cost time:
- `WienerPDF()` returns the density in `$value`, **not** `$pdf`.
- rtdists' default `precision = 3` gives only ~4 correct digits. Raise it for comparisons.
  `precision = 6` on the seven-parameter model costs ~16 ms per evaluation.
- rtdists takes **absolute** `z` and `sz`; WienR and this code take **relative** `w` and `sw`.
- Simulation check: the discrete barrier must sit *inside* the continuous one
  (`shift = +0.5826*sqrt(dt)`). The opposite sign looks plausible and is wrong.

## Lean formalization

The Lean 4 / Mathlib development in `RequestProject/` is a machine-checked formalization of
Result 1 of the note. Every theorem compiles with no `sorry` and no axioms beyond Lean's
standard three (`propext`, `Classical.choice`, `Quot.sound`).

Build with `lake exe cache get && lake build` (Mathlib v4.28.0).

### Files

| file | content |
|---|---|
| `RequestProject/DDM/Normal.lean` | the standard normal density `φ` and CDF `Φ` (absent from Mathlib), integrability of `φ`, and `Φ' = φ` |
| `RequestProject/DDM/Setup.lean` | the Setup section: `r_j`, `S`, the series `(g)`, `(gη)`, the drift density, and the bounds `j a ≤ r_j ≤ (j+1) a` |
| `RequestProject/DDM/Algebra.lean` | Lemmas L4 (the inequality Result 1 rests on), L5e/L5o (the coefficient tables), L6 (the drift integral reproduces `(gη)`) |
| `RequestProject/DDM/Antiderivatives.lean` | Lemma L1 |
| `RequestProject/DDM/Exchange.lean` | the dominated-convergence tool used to exchange summation and integration, and the summability of the `O(exp(−j²a²/2t))` bounds |
| `RequestProject/DDM/Result1.lean` | `theorem result1` (density with normal drift and uniform starting point) and `density_nondecision_window` (its consequence for the seven-parameter likelihood) |
| `RequestProject/DDM/Faddeeva.lean` | the complex error function and the Faddeeva function `w(z)` (both absent from Mathlib), with their derivatives |
| `RequestProject/DDM/LargeTime.lean` | the starting-point integral of the large-time representation in Faddeeva form, and its `η = 0` degeneration |

### What is proved, and what is assumed

The starting representations `(g)` and `(gη)` of the Setup section are *definitions* here,
exactly as the note takes them from the literature (Hall 1997 / Gondan et al. 2014; Horrocks &
Thompson 2004; Blurton et al. 2017). Everything the note derives from them is proved:

* the antiderivative claim L1 and the algebraic identities L4, L5, L6;
* Result 1 in full, including the exchange of summation with integration, proved from the
  note's own Gaussian bounds;
* the starting-point integral underlying the large-time (Faddeeva) representation.

Not formalized: the large-time series representation itself (quoted from the literature by
the note), the boundedness of `w(z)` in the upper half plane behind the note's numerical
remark, the gradient formulas, and the numerical and empirical comparisons.

