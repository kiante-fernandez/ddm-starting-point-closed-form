> **The first-passage time density for the diffusion model with variable drift and variable starting point**
> 
> Kianté Fernandez<sup>1</sup>
>
> <sup>1</sup>Department of Psychology, University of California, Los Angeles, CA, USA

## Abstract
The Ratcliff diffusion model is fitted with across-trial variability in drift rate, starting point, and non-decision time. The drift integral of the first-passage time density has been evaluated in closed form at fixed starting point, and the starting-point and non-decision-time integrals at fixed drift, but the drift and starting-point integrals have not been evaluated together; in particular, existing software implementations integrate at least one of these two numerically. We show that in the small-time series representation the starting-point integral of the drift-integrated density is a Gaussian moment, and we give the density with normally distributed drift and uniformly distributed starting point in closed form in exponentials and the normal distribution function, together with a large-time counterpart in terms of the Faddeeva function. Where these implementations required two numerical integrations, over the starting point and the non-decision time, the seven-parameter likelihood now requires a single one-dimensional quadrature; the derivatives of the decision density with respect to all of its parameters follow in closed form, and the likelihood gradient reuses that one quadrature. The result is verified symbolically and against three reference implementations, and is machine-checked in Lean.

This repository holds the code, reference data, and Lean proof behind the manuscript, which is available as a preprint at https://doi.org/10.2139/ssrn.7507587.

## Layout

```
fddm-fpt/core/           C++ density and gradient, six and seven parameters (MIT Faddeeva)
fddm-fpt/python/         Cython wrapper, HSSM-compatible full_ddm, benchmarks
fddm-fpt/R/              .Call wrapper (no Rcpp), WienR- and rtdists-compatible calls
src/ddm_fast.py          density, gradient, seven-parameter likelihood (numpy)
src/ddm_closed.py        independent scalar reference implementation
verify/                  every check and every number in the manuscript (see below)
data/                    WienR and rtdists reference values used by verify/
RequestProject/DDM/      Lean formalization
```

## Use

`fddm-fpt/` is the implementation to call from other software: one C++ core, thin R and Python
wrappers, drop-in parametrizations for HSSM, rtdists and WienR, density and gradient in the
same call.  See [fddm-fpt/README.md](fddm-fpt/README.md).

```python
import ddmsz                                   # after: make -C fddm-fpt
f, df = ddmsz.full_ddm(rt, response, v=1.2, a=0.75, z=0.45, t=0.3, sz=0.1, sv=0.8, st=0.12)
```

## Computational Reproducibility

```bash
pip install -r requirements.txt
make -C fddm-fpt                          # C++ core and both wrappers, each self-checked
python verify/symbolic_proof.py           # 52 exact identities
python verify/grad_check.py               # gradients, small-Delta branch, quadrature, compiled core, scalar reference, WienR comparisons
OMP_NUM_THREADS=1 python fddm-fpt/python/bench.py   # cost and accuracy vs hddm-wfpt, 100 published parameter sets
OMP_NUM_THREADS=1 Rscript fddm-fpt/R/bench.R        # same vs rtdists and WienR (slow: rtdists at precision 8)
cd fddm-fpt/python && SWEEP=1 python bench.py && cd ../R && SWEEP=1 Rscript bench.R   # cost vs trials per evaluation
cd fddm-fpt/python && python plot_fig.py            # fig_speed_accuracy.png
python verify/highprec2.py                # 40–60 digit checks (minutes)
python verify/highprec_grid.py            # both implementations vs a 50-digit reference on 1000 adversarial sets (~10 min)
lake exe cache get && lake build          # Lean proof (Mathlib v4.28.0)
```

The `.R` scripts in `verify/` regenerate `data/`; run them from `data/` with WienR and rtdists
installed. `tran_grid.R` draws the literature grid: 2000 parameter sets from the informative
priors of Tran et al. (2021, Front. Psychol.), truncated at that review's empirical bounds,
plus fixed points at the bounds. rtdists takes absolute `z = a*w` and `sz = a*(w2-w1)`; WienR and this code take
relative `w` and `w2-w1`.

## Lean

`lake build` checks every theorem with no `sorry`. `#print axioms DDM.result1` reports
`propext, Classical.choice, Quot.sound` and nothing else. The docstrings cite a working note
(`notes/closed_forms_sz_st.md`) that is not part of this repository; the manuscript contains the
same material, with the note's Result 1 as its Eq. 7 and Lemma L1 as its Eq. 6.

| file | content |
|---|---|
| `Normal.lean` | standard normal density and distribution function, `Φ' = φ` |
| `Setup.lean` | `r_j`, `S`, the small-time series, the drift density |
| `Algebra.lean` | the sign inequality, the coefficient table, the drift integral |
| `Antiderivatives.lean` | the Gaussian-moment antiderivative (manuscript Eq. 6) |
| `Exchange.lean` | exchange of summation and integration |
| `Result1.lean` | the closed-form density and the non-decision window |
| `Faddeeva.lean`, `LargeTime.lean` | the Faddeeva function and the large-time starting-point integral |

## Citation

If you use this software or the closed-form density in your work, please cite:
```bibtex
@article{fernandez_firstpassage,
  author  = {Fernandez, Kiant{\'e}},
  title   = {The first-passage time density for the diffusion model with variable drift and variable starting point},
  journal = {Manuscript in review},
  year    = {2026},
  doi     = {10.2139/ssrn.7507587},
  url     = {https://ssrn.com/abstract=7507587}
}
```
