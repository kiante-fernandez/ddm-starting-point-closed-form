# Lean formalization of `notes/closed_forms_sz_st.md`

The Lean 4 / Mathlib development in `RequestProject/` is a machine-checked formalization of
Result 1 of the note. Every theorem compiles with no `sorry` and no axioms beyond Lean's
standard three (`propext`, `Classical.choice`, `Quot.sound`).

Build with `lake exe cache get && lake build` (Mathlib v4.28.0).

## Files

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

## What is proved, and what is assumed

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
