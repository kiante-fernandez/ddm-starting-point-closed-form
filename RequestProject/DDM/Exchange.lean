import RequestProject.DDM.Setup

/-!
# Exchanging summation and integration

The note `notes/closed_forms_sz_st.md` states, in its Setup section, that

> In every series the `j`-th term is `O(exp(−j²a²/(2t)))`, so summation and integration may be
> exchanged freely (dominated convergence).

This file contains the general dominated-convergence tool used to carry out those exchanges
(a specialization of Mathlib's `MeasureTheory.integral_tsum_of_summable_integral_norm` to
interval integrals), together with the elementary summability facts used to check its
hypotheses for the Gaussian-type bounds `exp(−j²a²/(2t))` appearing in the note.
-/

noncomputable section

open Real MeasureTheory Set Filter

namespace DDM

/-- Termwise integration of a series over a finite interval, under the hypothesis that the
integrals of the norms are summable. -/
theorem intervalIntegral_tsum_of_summable {f : ℕ → ℝ → ℝ} {w₁ w₂ : ℝ} (hle : w₁ ≤ w₂)
    (hint : ∀ i, IntegrableOn (f i) (Ioc w₁ w₂))
    (hsum : Summable fun i => ∫ w in Ioc w₁ w₂, ‖f i w‖) :
    (∫ w in w₁..w₂, ∑' i, f i w) = ∑' i, ∫ w in w₁..w₂, f i w := by
  rw [intervalIntegral.integral_of_le hle]
  have : (∑' i, ∫ w in w₁..w₂, f i w) = ∑' i, ∫ w in Ioc w₁ w₂, f i w := by
    congr 1 with i
    rw [intervalIntegral.integral_of_le hle]
  rw [this]
  exact (integral_tsum_of_summable_integral_norm hint hsum).symm

/-- The Gaussian-type bounds `(j+1) e^{−j²c}` of the note `notes/closed_forms_sz_st.md`
are summable for every `c > 0`. -/
theorem summable_poly_gaussian (c : ℝ) (hc : 0 < c) :
    Summable fun j : ℕ => ((j : ℝ) + 1) * exp (-(j : ℝ) ^ 2 * c) := by
  have hq : ‖Real.exp (-c)‖ < 1 := by
    rw [Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
    exact Real.exp_lt_one_iff.mpr (by linarith)
  have hgeo : Summable fun j : ℕ => ((j : ℝ) + 1) * Real.exp (-c) ^ j := by
    have h1 : Summable fun j : ℕ => (j : ℝ) * Real.exp (-c) ^ j :=
      summable_pow_mul_geometric_of_norm_lt_one 1 hq |>.congr (fun n => by ring)
    have h2 : Summable fun j : ℕ => Real.exp (-c) ^ j := summable_geometric_of_norm_lt_one hq
    exact (h1.add h2).congr (fun n => by ring)
  apply Summable.of_nonneg_of_le (fun j => by positivity) _ hgeo
  intro j
  have hle : Real.exp (-(j : ℝ) ^ 2 * c) ≤ Real.exp (-c) ^ j := by
    rw [← Real.exp_nat_mul]
    apply Real.exp_le_exp.mpr
    have hj : (j : ℝ) ≤ (j : ℝ) ^ 2 := by
      rcases Nat.eq_zero_or_pos j with rfl | hpos
      · simp
      · have h1 : (1:ℝ) ≤ (j:ℝ) := by exact_mod_cast hpos
        nlinarith
    nlinarith
  have hnn : (0:ℝ) ≤ (j : ℝ) + 1 := by positivity
  nlinarith [Real.exp_pos (-(j:ℝ)^2 * c)]

end DDM
