# The first-passage time density for the diffusion model with variable drift and variable starting point

Code, reference data, and Lean 4 proof for the manuscript in `paper/`. The result is the
first-passage time density of the Ratcliff diffusion model with normally distributed drift and
uniformly distributed starting point in closed form, its gradient, and the seven-parameter
likelihood with a single one-dimensional quadrature.

## Layout

```
src/ddm_fast.py          density, gradient, seven-parameter likelihood (numpy)
src/ddm_kernel.pyx       compiled density and seven-parameter likelihood (Cython)
src/ddm_closed.py        independent scalar reference implementation
verify/                  every check and every number in the manuscript (see below)
data/                    WienR and rtdists reference values used by verify/
notes/                   working note the Lean files cite
RequestProject/DDM/      Lean 4 formalization
paper/                   manuscript source
```

## Reproduce

```bash
pip install -r requirements.txt
cythonize -3 -i src/ddm_kernel.pyx        # compiled implementation
python verify/symbolic_proof.py           # 32 exact identities
python verify/grad_check.py               # gradients, quadrature, kernel, WienR comparisons
python verify/timing.py                   # Table 2, closed-form rows (single thread)
Rscript verify/timing.R                   # Table 2, WienR and rtdists rows
python verify/hddm_wfpt_compare.py        # Table 2, HDDM row (pip install hddm-wfpt)
python verify/highprec2.py                # 40–60 digit checks (minutes)
PYTHONPATH=src python verify/validate_sim2.py   # process simulation (minutes)
lake exe cache get && lake build          # Lean proof (Mathlib v4.28.0)
```

The `.R` scripts in `verify/` regenerate `data/`; run them from `data/` with WienR and rtdists
installed. rtdists takes absolute `z = a*w` and `sz = a*(w2-w1)`; WienR and this code take
relative `w` and `w2-w1`.

## Conventions

Unit diffusion coefficient, barriers at 0 and `a`, relative starting point `w`, lower barrier;
upper barrier by `(v, w) -> (-v, 1-w)`. Drift `v ~ N(nu, eta^2)`, starting point uniform on
`[w1, w2]`, `S = 1 + eta^2 t`.

## Lean formalization

`lake build` checks every theorem with no `sorry` and only Lean's standard axioms.

| file | content |
|---|---|
| `Normal.lean` | standard normal density and distribution function, `Φ' = φ` |
| `Setup.lean` | `r_j`, `S`, the small-time series, the drift density |
| `Algebra.lean` | the sign inequality, the coefficient table, the drift integral |
| `Antiderivatives.lean` | the Gaussian-moment antiderivative (manuscript Eq. 6) |
| `Exchange.lean` | exchange of summation and integration |
| `Result1.lean` | the closed-form density and the non-decision window |
| `Faddeeva.lean`, `LargeTime.lean` | the Faddeeva function and the large-time starting-point integral |

The drift-integrated small-time density is a definition, as it is taken from the literature in
the manuscript. Not formalized: the large-time series itself, the bound on the Faddeeva
function, and the gradient formulas.
