# Closed forms for across-trial variability in the Ratcliff diffusion model

**Result.** A closed-form first-passage density for the Ratcliff diffusion model with normally
distributed drift AND uniformly distributed starting point (Result 1), with closed-form
gradients in all five parameters. Blurton, Kesselmeier & Gondan (2017) removed the drift
integral; this removes the starting-point integral. The seven-parameter likelihood then needs
one fixed 1-D quadrature over the non-decision-time window instead of 2-D cubature. Verified
symbolically, to 48 digits, against WienR and rtdists, against simulation, and machine-checked
in Lean 4.

## Read these in order

1. `STATUS.md` — what is done (with recomputed numbers), what to do next, what not to retry.
2. `notes/closed_forms_sz_st.md` — the mathematical note: every formula and derivation.
3. `FORMALIZATION.md` — the Lean 4 / Mathlib formalization in `RequestProject/`.

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
           timing.R, timing.py  the STATUS.md timing table (WienR / closed form)
           *.R                  WienR and rtdists reference-value generation
data/      *.csv           per-row comparison data (regenerate with the .R scripts)
notes/     the mathematical note
docs/      source papers: Blurton et al. 2017, Gondan et al. 2014, Hartmann & Klauer 2021,
           Tuerlinckx 2004, Henrich et al. 2024 (Stan), WienR manual, Cambridge chapter
RequestProject/DDM/*.lean   Lean formalization (see FORMALIZATION.md)
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
