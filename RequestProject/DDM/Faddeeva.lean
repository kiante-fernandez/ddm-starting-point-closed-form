import Mathlib

/-!
# The complex error function and the Faddeeva function

The section *"Large-time representation of Result 1"* of the note `notes/closed_forms_sz_st.md`
writes the `w`-integral of the large-time series in terms of the **Faddeeva function** `w(z)`
(the scaled complex complementary error function).  Mathlib contains neither `erf` nor `w(z)`,
so both are introduced here:

* `DDM.cerf` — the complex error function `erf(z) = (2/√π) ∫₀^z e^{−s²} ds`, written as an
  integral over the straight segment from `0` to `z`;
* `DDM.hasDerivAt_cerf` — `erf'(z) = (2/√π) e^{−z²}`, the only property of `erf` the note's
  derivation uses;
* `DDM.faddeeva` — the Faddeeva function `w(z) = e^{−z²} erfc(−i z) = e^{−z²} (1 − erf(−i z))`
  of the note;
* `DDM.hasDerivAt_faddeeva` — its derivative.
-/

noncomputable section

open Real MeasureTheory Set Filter intervalIntegral

namespace DDM

/-- The complex error function `erf(z) = (2/√π) ∫₀^z e^{−s²} ds`, the `erf` implicit in the
Faddeeva function `w(z)` of the note `notes/closed_forms_sz_st.md`.  The contour integral from
`0` to `z` is written as an integral over the straight segment, i.e. over the real parameter
`s ∈ [0,1]` of `z e^{−(z s)²}`. -/
def cerf (z : ℂ) : ℂ :=
  (2 / (Real.sqrt π : ℂ)) * ∫ s in (0:ℝ)..1, z * Complex.exp (-(z * (s : ℂ)) ^ 2)

/-- The `s`-antiderivative used to evaluate the derivative of `DDM.cerf`:
`d/ds [ s e^{−(z s)²} ] = e^{−(z s)²} (1 − 2 z² s²)`. -/
lemma hasDerivAt_seg (z : ℂ) (s : ℝ) :
    HasDerivAt (fun s : ℝ => (s : ℂ) * Complex.exp (-(z * (s : ℂ)) ^ 2))
      (Complex.exp (-(z * (s : ℂ)) ^ 2) * (1 - 2 * z ^ 2 * (s : ℂ) ^ 2)) s := by
  have hs : HasDerivAt (fun s : ℝ => (s : ℂ)) 1 s := Complex.ofRealCLM.hasDerivAt
  have h1 : HasDerivAt (fun s : ℝ => z * (s : ℂ)) z s := by simpa using hs.const_mul z
  have h2 : HasDerivAt (fun s : ℝ => (z * (s : ℂ)) ^ 2) (2 * z ^ 2 * (s : ℂ)) s := by
    have h := h1.pow 2
    convert h using 1
    push_cast
    ring
  have hexp : HasDerivAt (fun s : ℝ => Complex.exp (-(z * (s : ℂ)) ^ 2))
      (Complex.exp (-(z * (s : ℂ)) ^ 2) * -(2 * z ^ 2 * (s : ℂ))) s := by
    simpa using h2.neg.cexp
  have hprod : HasDerivAt (fun s : ℝ => (s : ℂ) * Complex.exp (-(z * (s : ℂ)) ^ 2))
      (1 * Complex.exp (-(z * (s : ℂ)) ^ 2)
        + (s : ℂ) * (Complex.exp (-(z * (s : ℂ)) ^ 2) * -(2 * z ^ 2 * (s : ℂ)))) s :=
    hs.mul hexp
  convert hprod using 1
  ring

/-- The `z`-derivative of the integrand of `DDM.cerf` at a fixed point `s` of the segment. -/
lemma hasDerivAt_cerfIntegrand (s : ℝ) (z : ℂ) :
    HasDerivAt (fun z : ℂ => z * Complex.exp (-(z * (s : ℂ)) ^ 2))
      (Complex.exp (-(z * (s : ℂ)) ^ 2) * (1 - 2 * z ^ 2 * (s : ℂ) ^ 2)) z := by
  have hz : HasDerivAt (fun z : ℂ => z) 1 z := hasDerivAt_id z
  have h1 : HasDerivAt (fun z : ℂ => z * (s : ℂ)) (s : ℂ) z := by
    simpa using hz.mul_const (s : ℂ)
  have h2 : HasDerivAt (fun z : ℂ => (z * (s : ℂ)) ^ 2) (2 * z * (s : ℂ) ^ 2) z := by
    have h := h1.pow 2
    convert h using 1
    push_cast
    ring
  have hexp : HasDerivAt (fun z : ℂ => Complex.exp (-(z * (s : ℂ)) ^ 2))
      (Complex.exp (-(z * (s : ℂ)) ^ 2) * -(2 * z * (s : ℂ) ^ 2)) z := by
    simpa using h2.neg.cexp
  have hprod : HasDerivAt (fun z : ℂ => z * Complex.exp (-(z * (s : ℂ)) ^ 2))
      (1 * Complex.exp (-(z * (s : ℂ)) ^ 2)
        + z * (Complex.exp (-(z * (s : ℂ)) ^ 2) * -(2 * z * (s : ℂ) ^ 2))) z :=
    hz.mul hexp
  convert hprod using 1
  ring

/-- **The derivative of the complex error function**, `erf'(z) = (2/√π) e^{−z²}`.

This is the only property of `erf` used in the large-time representation of Result 1 in the
note `notes/closed_forms_sz_st.md`. -/
theorem hasDerivAt_cerf (z₀ : ℂ) :
    HasDerivAt cerf ((2 / (Real.sqrt π : ℂ)) * Complex.exp (-z₀ ^ 2)) z₀ := by
  have hkey : HasDerivAt (fun z : ℂ => ∫ s in (0:ℝ)..1, z * Complex.exp (-(z * (s : ℂ)) ^ 2))
      (Complex.exp (-z₀ ^ 2)) z₀ := by
    have hcontF : ∀ z : ℂ, Continuous (fun s : ℝ => z * Complex.exp (-(z * (s : ℂ)) ^ 2)) := by
      intro z; fun_prop
    have hcontF' : ∀ z : ℂ, Continuous (fun s : ℝ =>
        Complex.exp (-(z * (s : ℂ)) ^ 2) * (1 - 2 * z ^ 2 * (s : ℂ) ^ 2)) := by
      intro z; fun_prop
    have hmain := hasDerivAt_integral_of_dominated_loc_of_deriv_le
      (μ := volume.restrict (Ioc (0:ℝ) 1))
      (F := fun (z : ℂ) (s : ℝ) => z * Complex.exp (-(z * (s : ℂ)) ^ 2))
      (F' := fun (z : ℂ) (s : ℝ) =>
        Complex.exp (-(z * (s : ℂ)) ^ 2) * (1 - 2 * z ^ 2 * (s : ℂ) ^ 2))
      (x₀ := z₀) (s := Metric.ball z₀ 1)
      (bound := fun _ : ℝ => Real.exp ((‖z₀‖ + 1) ^ 2) * (1 + 2 * (‖z₀‖ + 1) ^ 2))
      (Metric.ball_mem_nhds z₀ one_pos)
      (Filter.Eventually.of_forall fun z => (hcontF z).aestronglyMeasurable)
      ((hcontF z₀).integrableOn_Ioc)
      ((hcontF' z₀).aestronglyMeasurable)
      ?_ ?_ ?_
    · have hset : (∫ s in (0:ℝ)..1, Complex.exp (-(z₀ * (s : ℂ)) ^ 2)
            * (1 - 2 * z₀ ^ 2 * (s : ℂ) ^ 2))
          = Complex.exp (-z₀ ^ 2) := by
        have hFTC := intervalIntegral.integral_eq_sub_of_hasDerivAt
          (f := fun s : ℝ => (s : ℂ) * Complex.exp (-(z₀ * (s : ℂ)) ^ 2))
          (f' := fun s : ℝ => Complex.exp (-(z₀ * (s : ℂ)) ^ 2)
            * (1 - 2 * z₀ ^ 2 * (s : ℂ) ^ 2))
          (a := 0) (b := 1) (fun s _ => hasDerivAt_seg z₀ s)
          ((hcontF' z₀).intervalIntegrable 0 1)
        simpa using hFTC
      rw [intervalIntegral.integral_of_le zero_le_one] at hset
      simp only [intervalIntegral.integral_of_le zero_le_one]
      rw [← hset]
      exact hmain.2
    · -- the uniform bound on the derivative over the ball of radius one
      refine (ae_restrict_iff' measurableSet_Ioc).2 (Filter.Eventually.of_forall ?_)
      intro s hs z hz
      have hs0 : (0:ℝ) < s := hs.1
      have hs1 : s ≤ 1 := hs.2
      have hzR : ‖z‖ ≤ ‖z₀‖ + 1 := by
        have := Metric.mem_ball.1 hz
        have h1 : ‖z - z₀‖ < 1 := by simpa [dist_eq_norm] using this
        have h2 : ‖z‖ ≤ ‖z - z₀‖ + ‖z₀‖ := by simpa using norm_add_le (z - z₀) z₀
        linarith
      have hsabs : ‖(s : ℂ)‖ ≤ 1 := by
        rw [Complex.norm_real, Real.norm_eq_abs, abs_of_pos hs0]
        exact hs1
      have hR0 : (0:ℝ) ≤ ‖z₀‖ + 1 := by positivity
      have hexp : ‖Complex.exp (-(z * (s : ℂ)) ^ 2)‖ ≤ Real.exp ((‖z₀‖ + 1) ^ 2) := by
        rw [Complex.norm_exp]
        apply Real.exp_le_exp.mpr
        have h1 : (-(z * (s : ℂ)) ^ 2).re ≤ ‖(-(z * (s : ℂ)) ^ 2)‖ := Complex.re_le_norm _
        have h2 : ‖(-(z * (s : ℂ)) ^ 2)‖ = ‖z‖ ^ 2 * ‖(s : ℂ)‖ ^ 2 := by
          simp [norm_pow, mul_pow]
        have h3 : ‖z‖ ^ 2 * ‖(s : ℂ)‖ ^ 2 ≤ (‖z₀‖ + 1) ^ 2 := by
          have hz0 : (0:ℝ) ≤ ‖z‖ := norm_nonneg z
          have hs0' : (0:ℝ) ≤ ‖(s : ℂ)‖ := norm_nonneg _
          have hz2 : ‖z‖ ^ 2 ≤ (‖z₀‖ + 1) ^ 2 := by nlinarith
          have hs2 : ‖(s : ℂ)‖ ^ 2 ≤ 1 := by nlinarith
          nlinarith [sq_nonneg ‖z‖, sq_nonneg ‖(s : ℂ)‖]
        linarith
      have hlin : ‖1 - 2 * z ^ 2 * (s : ℂ) ^ 2‖ ≤ 1 + 2 * (‖z₀‖ + 1) ^ 2 := by
        have h1 : ‖1 - 2 * z ^ 2 * (s : ℂ) ^ 2‖ ≤ ‖(1 : ℂ)‖ + ‖2 * z ^ 2 * (s : ℂ) ^ 2‖ :=
          norm_sub_le _ _
        have h2 : ‖2 * z ^ 2 * (s : ℂ) ^ 2‖ = 2 * ‖z‖ ^ 2 * ‖(s : ℂ)‖ ^ 2 := by
          simp [norm_pow]
        have h3 : 2 * ‖z‖ ^ 2 * ‖(s : ℂ)‖ ^ 2 ≤ 2 * (‖z₀‖ + 1) ^ 2 := by
          have hz0 : (0:ℝ) ≤ ‖z‖ := norm_nonneg z
          have hs0' : (0:ℝ) ≤ ‖(s : ℂ)‖ := norm_nonneg _
          have hz2 : ‖z‖ ^ 2 ≤ (‖z₀‖ + 1) ^ 2 := by nlinarith
          have hs2 : ‖(s : ℂ)‖ ^ 2 ≤ 1 := by nlinarith
          nlinarith [sq_nonneg ‖z‖, sq_nonneg ‖(s : ℂ)‖]
        simp only [norm_one] at h1
        linarith
      calc ‖Complex.exp (-(z * (s : ℂ)) ^ 2) * (1 - 2 * z ^ 2 * (s : ℂ) ^ 2)‖
          = ‖Complex.exp (-(z * (s : ℂ)) ^ 2)‖ * ‖1 - 2 * z ^ 2 * (s : ℂ) ^ 2‖ := norm_mul _ _
        _ ≤ Real.exp ((‖z₀‖ + 1) ^ 2) * (1 + 2 * (‖z₀‖ + 1) ^ 2) := by
            apply mul_le_mul hexp hlin (norm_nonneg _) (Real.exp_pos _).le
    · apply integrableOn_const
      · rw [Real.volume_Ioc]; exact ENNReal.ofReal_ne_top
      · exact enorm_ne_top
    · refine (ae_restrict_iff' measurableSet_Ioc).2 (Filter.Eventually.of_forall ?_)
      intro s _ z _
      exact hasDerivAt_cerfIntegrand s z
  have hfinal := hkey.const_mul (2 / (Real.sqrt π : ℂ))
  exact hfinal.congr_of_eventuallyEq (Filter.Eventually.of_forall fun z => rfl)

/-- The **Faddeeva function** `w(z) = e^{−z²} erfc(−i z) = e^{−z²} (1 − erf(−i z))` of the note
`notes/closed_forms_sz_st.md`. -/
def faddeeva (z : ℂ) : ℂ := Complex.exp (-z ^ 2) * (1 - cerf (-Complex.I * z))

/-- The derivative of the Faddeeva function:
`w'(z) = −2 z w(z) + 2i/√π`. -/
theorem hasDerivAt_faddeeva (z : ℂ) :
    HasDerivAt faddeeva (-2 * z * faddeeva z + 2 * Complex.I / (Real.sqrt π : ℂ)) z := by
  have hsq : -(-Complex.I * z) ^ 2 = z ^ 2 := by
    have hI : Complex.I ^ 2 = -1 := Complex.I_sq
    have : (-Complex.I * z) ^ 2 = Complex.I ^ 2 * z ^ 2 := by ring
    rw [this, hI]
    ring
  have hgauss : HasDerivAt (fun z : ℂ => Complex.exp (-z ^ 2))
      (Complex.exp (-z ^ 2) * -(2 * z)) z := by
    have h : HasDerivAt (fun z : ℂ => z ^ 2) (2 * z) z := by
      simpa using (hasDerivAt_id z).pow 2
    simpa using h.neg.cexp
  have hlin : HasDerivAt (fun z : ℂ => -Complex.I * z) (-Complex.I) z := by
    simpa using (hasDerivAt_id z).const_mul (-Complex.I)
  have hcomp : HasDerivAt (fun z : ℂ => cerf (-Complex.I * z))
      ((2 / (Real.sqrt π : ℂ)) * Complex.exp (-(-Complex.I * z) ^ 2) * -Complex.I) z :=
    (hasDerivAt_cerf _).comp z hlin
  have hsub : HasDerivAt (fun z : ℂ => 1 - cerf (-Complex.I * z))
      (-((2 / (Real.sqrt π : ℂ)) * Complex.exp (-(-Complex.I * z) ^ 2) * -Complex.I)) z :=
    hcomp.const_sub 1
  have hcancel : Complex.exp (-z ^ 2) * Complex.exp (z ^ 2) = 1 := by
    rw [← Complex.exp_add]
    simp
  have hprod := hgauss.mul hsub
  convert hprod using 1
  simp only [faddeeva, hsq]
  linear_combination (-(2 * Complex.I / (Real.sqrt π : ℂ))) * hcancel

end DDM
