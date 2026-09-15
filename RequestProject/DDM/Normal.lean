import Mathlib

/-!
# The standard normal density and CDF

This file provides the two special functions `φ` and `Φ` that appear throughout the note
`notes/closed_forms_sz_st.md` ("Closed forms for starting-point and non-decision-time
variability in the Ratcliff diffusion model").

The note uses, without definition, the standard normal density `φ` and the standard normal
distribution function `Φ`.  Mathlib has neither `erf` nor a Gaussian CDF, so both are
introduced here together with the analytic facts the note's derivations rely on:

* `DDM.phi` — the standard normal density `φ(x) = e^{-x²/2}/√(2π)`;
* `DDM.Phi` — the standard normal CDF `Φ(x) = ∫_{-∞}^x φ`;
* `DDM.hasDerivAt_Phi` — `Φ' = φ`, the fact used every time the note verifies an
  antiderivative claim as `dF/dx − f = 0` (checks L1, L2, L3 of the note);
* `DDM.hasDerivAt_phi` — `φ'(x) = −x φ(x)`, used in the `c = 0` branch of L2.
-/

noncomputable section

open Real MeasureTheory Set Filter

namespace DDM

/-- The standard normal density `φ(x) = e^{-x²/2}/√(2π)`.

This is the function written `φ` in the note `notes/closed_forms_sz_st.md` (it appears e.g. in
the small-time density series `(g)` through `r_j φ(r_j/√t)`, and in the antiderivatives of
Results 2 and 3). -/
def phi (x : ℝ) : ℝ := Real.exp (-x ^ 2 / 2) / Real.sqrt (2 * π)

/-- The standard normal distribution function `Φ(x) = ∫_{-∞}^x φ(u) du`.

This is the function written `Φ` throughout the note `notes/closed_forms_sz_st.md`. -/
def Phi (x : ℝ) : ℝ := ∫ u in Iic x, phi u

@[simp] lemma phi_neg (x : ℝ) : phi (-x) = phi x := by simp [phi]


/-- `φ` is continuous. -/
@[continuity, fun_prop]
lemma continuous_phi : Continuous phi := by
  unfold phi; fun_prop

/-- The derivative of the standard normal density: `φ'(x) = −x φ(x)`.

This is the elementary fact behind the `c = 0` branch of the note's Lemma L2. -/
lemma hasDerivAt_phi (x : ℝ) : HasDerivAt phi (-x * phi x) x := by
  have h : HasDerivAt (fun y : ℝ => Real.exp (-y ^ 2 / 2)) (Real.exp (-x ^ 2 / 2) * (-x)) x := by
    have h1 : HasDerivAt (fun y : ℝ => -y ^ 2 / 2) (-x) x := by
      have := ((hasDerivAt_pow 2 x).neg).div_const 2
      convert this using 1
      ring
    simpa using (Real.hasDerivAt_exp (-x ^ 2 / 2)).comp x h1
  have h2 := h.div_const (Real.sqrt (2 * π))
  convert h2 using 1
  simp only [phi]
  ring

/-- `φ` is integrable on the whole line. -/
lemma integrable_phi : Integrable phi := by
  have h0 : Integrable (fun x : ℝ => Real.exp (-(1/2 : ℝ) * x ^ 2)) :=
    integrable_exp_neg_mul_sq (by norm_num)
  refine (h0.div_const (Real.sqrt (2 * π))).congr ?_
  filter_upwards with x
  simp only [phi]
  ring_nf

lemma integrableOn_phi_Iic (x : ℝ) : IntegrableOn phi (Iic x) :=
  integrable_phi.integrableOn

/-- `Φ(b) − Φ(a) = ∫_a^b φ`. -/
lemma Phi_sub_Phi (a b : ℝ) : Phi b - Phi a = ∫ u in a..b, phi u :=
  intervalIntegral.integral_Iic_sub_Iic (integrableOn_phi_Iic a) (integrableOn_phi_Iic b)

/-- **`Φ' = φ`.**  The standard normal CDF differentiates to the standard normal density.

Every antiderivative claim of the note `notes/closed_forms_sz_st.md` (Lemmas L1, L2, L3, each
verified there by SymPy as `simplify(dF/dx − f) == 0`) is proved here by differentiating the
claimed antiderivative, and this lemma is what makes those differentiations possible. -/
lemma hasDerivAt_Phi (x : ℝ) : HasDerivAt Phi (phi x) x := by
  have key : ∀ y : ℝ, Phi y = Phi 0 + ∫ u in (0:ℝ)..y, phi u := by
    intro y
    have := Phi_sub_Phi 0 y
    linarith
  have h : HasDerivAt (fun y : ℝ => Phi 0 + ∫ u in (0:ℝ)..y, phi u) (phi x) x := by
    have hint : IntervalIntegrable phi volume 0 x := integrable_phi.intervalIntegrable
    have := intervalIntegral.integral_hasDerivAt_right hint
      (continuous_phi.stronglyMeasurableAtFilter _ _) continuous_phi.continuousAt
    simpa using this.const_add (Phi 0)
  exact h.congr_of_eventuallyEq (Filter.Eventually.of_forall fun y => key y)

end DDM
