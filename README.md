> **The first-passage time density for the diffusion model with variable drift and variable starting point**
> Kianté Fernandez<sup>1</sup>
> <sup>1</sup>Department of Psychology, University of California, Los Angeles, CA, USA

## Abstract
The Ratcliff diffusion model is fitted with across-trial variability in drift rate, starting point, and non-decision time. The drift integral of the first-passage time density has been evaluated in closed form at fixed starting point, and the starting-point and non-decision-time integrals at fixed drift, but the drift and starting-point integrals have not been evaluated together; in particular, every implementation we are aware of that starts from the drift-integrated density integrates the starting point numerically. We show that in the small-time series representation the starting-point integral of the drift-integrated density is a Gaussian moment, and we give the density with normally distributed drift and uniformly distributed starting point in closed form in exponentials and the normal distribution function, together with a large-time counterpart in terms of the Faddeeva function. Where these implementations required two numerical integrations, over the starting point and the non-decision time, the seven-parameter likelihood now requires a single one-dimensional quadrature; the derivatives of the decision density with respect to all of its parameters follow in closed form, and the likelihood gradient reuses that one quadrature. The result is verified symbolically, against three reference implementations and process simulation, and is machine-checked in Lean.

This repository holds the code, reference data, manuscript source, and Lean proof.

## Layout

```
src/ddm_fast.py          density, gradient, seven-parameter likelihood (numpy)
src/ddm_kernel.pyx       compiled density and seven-parameter likelihood (Cython)
src/ddm_closed.py        independent scalar reference implementation
verify/                  every check and every number in the manuscript (see below)
data/                    WienR and rtdists reference values used by verify/
notes/                   working note the Lean files cite
RequestProject/DDM/      Lean formalization
paper/                   manuscript source
```

## Reproduce

```bash
pip install -r requirements.txt
cythonize -3 -i src/ddm_kernel.pyx        # compiled implementation
python verify/symbolic_proof.py           # 38 exact identities
python verify/grad_check.py               # gradients, small-Delta branch, quadrature, kernel, scalar reference, WienR comparisons
python verify/timing.py                   # Table 2, closed-form rows (single thread)
Rscript verify/timing.R                   # Table 2, WienR and rtdists rows
python verify/hddm_wfpt_compare.py        # Table 2, HDDM row (pip install hddm-wfpt)
python verify/highprec2.py                # 40–60 digit checks (minutes)
PYTHONPATH=src python verify/validate_sim2.py   # process simulation (minutes)
lake exe cache get && lake build          # Lean proof (Mathlib v4.28.0)
```

The `.R` scripts in `verify/` regenerate `data/`; run them from `data/` with WienR and rtdists
installed. `tran_grid.R` draws the literature grid: 2000 parameter sets from the informative
priors of Tran et al. (2021, Front. Psychol.), truncated at that review's empirical bounds,
plus fixed points at the bounds. rtdists takes absolute `z = a*w` and `sz = a*(w2-w1)`; WienR and this code take
relative `w` and `w2-w1`.

## Lean

`lake build` checks every theorem with no `sorry`. `#print axioms DDM.result1` reports
`propext, Classical.choice, Quot.sound` and nothing else.

| file | content |
|---|---|
| `Normal.lean` | standard normal density and distribution function, `Φ' = φ` |
| `Setup.lean` | `r_j`, `S`, the small-time series, the drift density |
| `Algebra.lean` | the sign inequality, the coefficient table, the drift integral |
| `Antiderivatives.lean` | the Gaussian-moment antiderivative (manuscript Eq. 6) |
| `Exchange.lean` | exchange of summation and integration |
| `Result1.lean` | the closed-form density and the non-decision window |
| `Faddeeva.lean`, `LargeTime.lean` | the Faddeeva function and the large-time starting-point integral |
