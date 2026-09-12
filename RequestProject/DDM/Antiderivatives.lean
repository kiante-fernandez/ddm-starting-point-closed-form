import RequestProject.DDM.Setup

/-!
# The antiderivative lemma L1 of the note

This file formalizes the antiderivative claim of the note `notes/closed_forms_sz_st.md`,
listed there in the table of symbolic checks (`verify/symbolic_proof.py`) as

| check | content |
|---|---|
| L1 | `∫ (p+qw) e^{−Aw²/2+Bw+C} dw` antiderivative (core of Result 1) |

It is proved as an honest theorem about the interval integral, by exhibiting the
antiderivative and differentiating it (which is exactly what the note's symbolic check
`simplify(dF/dx − f) == 0` verifies), and then applying the fundamental theorem of calculus.
-/

noncomputable section

open Real MeasureTheory Set Filter intervalIntegral

namespace DDM

/-! ## Lemma L1 -/

/-- The closed form of `I = ∫_{w₁}^{w₂} (p + q w) exp(−A w²/2 + B w + C) dw` claimed in
Result 1 of the note `notes/closed_forms_sz_st.md`:

    I = e^{C + B²/(2A)} (p + q m) √(2π/A) [ Φ(√A (w₂ − m)) − Φ(√A (w₁ − m)) ]
        − (q/A) [ e^{C + B w₂ − A w₂²/2} − e^{C + B w₁ − A w₁²/2} ],     m = B/A. -/
def L1closed (p q A B C w₁ w₂ : ℝ) : ℝ :=
  exp (C + B ^ 2 / (2 * A)) * (p + q * (B / A)) * sqrt (2 * π / A)
      * (Phi (sqrt A * (w₂ - B / A)) - Phi (sqrt A * (w₁ - B / A)))
    - (q / A) * (exp (C + B * w₂ - A * w₂ ^ 2 / 2) - exp (C + B * w₁ - A * w₁ ^ 2 / 2))

/-- The antiderivative used in the proof of `L1`: the indefinite version of `L1closed`. -/
def L1anti (p q A B C : ℝ) (w : ℝ) : ℝ :=
  exp (C + B ^ 2 / (2 * A)) * (p + q * (B / A)) * sqrt (2 * π / A) * Phi (sqrt A * (w - B / A))
    - (q / A) * exp (C + B * w - A * w ^ 2 / 2)

/-- **Lemma L1 of the note `notes/closed_forms_sz_st.md`, in differentiated form.**
The stated antiderivative of `(p + q w) exp(−A w²/2 + B w + C)` is correct. -/
theorem hasDerivAt_L1anti (p q A B C : ℝ) (hA : 0 < A) (w : ℝ) :
    HasDerivAt (L1anti p q A B C)
      ((p + q * w) * exp (-A * w ^ 2 / 2 + B * w + C)) w := by
  have hA' : A ≠ 0 := ne_of_gt hA
  have h1 : HasDerivAt (fun x : ℝ => Phi (sqrt A * (x - B / A)))
      (phi (sqrt A * (w - B / A)) * sqrt A) w := by
    have hlin : HasDerivAt (fun x : ℝ => sqrt A * (x - B / A)) (sqrt A) w := by
      simpa using ((hasDerivAt_id w).sub_const (B / A)).const_mul (sqrt A)
    simpa using (hasDerivAt_Phi (sqrt A * (w - B / A))).comp w hlin
  have h2 : HasDerivAt (fun x : ℝ => exp (C + B * x - A * x ^ 2 / 2))
      (exp (C + B * w - A * w ^ 2 / 2) * (B - A * w)) w := by
    have hinner : HasDerivAt (fun x : ℝ => C + B * x - A * x ^ 2 / 2) (B - A * w) w := by
      have := (((hasDerivAt_id w).const_mul B).const_add C).sub
        (((hasDerivAt_pow 2 w).const_mul A).div_const 2)
      convert this using 1
      simp
      ring
    exact (Real.hasDerivAt_exp _).comp w hinner
  have hcomb := (h1.const_mul (exp (C + B ^ 2 / (2 * A)) * (p + q * (B / A))
      * sqrt (2 * π / A))).sub (h2.const_mul (q / A))
  convert hcomb using 1
  have hsqrt : sqrt (2 * π / A) * sqrt A = sqrt (2 * π) := by
    rw [← Real.sqrt_mul (by positivity)]
    congr 1
    field_simp
  have hphi : phi (sqrt A * (w - B / A)) = exp (-(A * (w - B / A) ^ 2) / 2) / sqrt (2 * π) := by
    simp only [phi]
    congr 2
    rw [mul_pow, Real.sq_sqrt hA.le]
  have hexp : exp (C + B ^ 2 / (2 * A)) * exp (-(A * (w - B / A) ^ 2) / 2)
      = exp (C + B * w - A * w ^ 2 / 2) := by
    rw [← Real.exp_add]
    congr 1
    field_simp
    ring
  have h2pi : sqrt (2 * π) ≠ 0 := by positivity
  have e1 : exp (C + B ^ 2 / (2 * A)) * (p + q * (B / A)) * sqrt (2 * π / A)
        * (phi (sqrt A * (w - B / A)) * sqrt A)
      = (p + q * (B / A)) * exp (C + B * w - A * w ^ 2 / 2) := by
    rw [hphi]
    rw [show exp (C + B ^ 2 / (2 * A)) * (p + q * (B / A)) * sqrt (2 * π / A)
          * (exp (-(A * (w - B / A) ^ 2) / 2) / sqrt (2 * π) * sqrt A)
        = (p + q * (B / A)) * (exp (C + B ^ 2 / (2 * A)) * exp (-(A * (w - B / A) ^ 2) / 2))
          * (sqrt (2 * π / A) * sqrt A / sqrt (2 * π)) from by ring]
    rw [hexp, hsqrt, div_self h2pi, mul_one]
  rw [e1, show (-A * w ^ 2 / 2 + B * w + C) = (C + B * w - A * w ^ 2 / 2) from by ring]
  field_simp
  ring

/-- **Lemma L1 of the note `notes/closed_forms_sz_st.md`** (check `L1` of
`verify/symbolic_proof.py`, the core of Result 1):

    ∫_{w₁}^{w₂} (p + q w) exp(−A w²/2 + B w + C) dw
      = e^{C+B²/(2A)} (p + q m) √(2π/A) [Φ(√A(w₂−m)) − Φ(√A(w₁−m))]
        − (q/A) [e^{C+Bw₂−Aw₂²/2} − e^{C+Bw₁−Aw₁²/2}],   m = B/A,

valid for every `A > 0`. -/
theorem lemma_L1 (p q A B C w₁ w₂ : ℝ) (hA : 0 < A) :
    (∫ w in w₁..w₂, (p + q * w) * exp (-A * w ^ 2 / 2 + B * w + C))
      = L1closed p q A B C w₁ w₂ := by
  have hcont : Continuous fun w : ℝ => (p + q * w) * exp (-A * w ^ 2 / 2 + B * w + C) := by
    fun_prop
  have hFTC := intervalIntegral.integral_eq_sub_of_hasDerivAt
    (f := L1anti p q A B C) (fun x _ => hasDerivAt_L1anti p q A B C hA x)
    (hcont.intervalIntegrable w₁ w₂)
  rw [hFTC]
  simp only [L1anti, L1closed]
  ring

end DDM
