import RequestProject.DDM.Antiderivatives
import RequestProject.DDM.Algebra
import RequestProject.DDM.Exchange

/-!
# Result 1: density with normal drift AND uniform starting point

This file formalizes **Result 1** of the note `notes/closed_forms_sz_st.md`:

> ## Result 1 — Density with normal drift AND uniform starting point (closed form)
>
> Starting point uniform on `[w₁, w₂]` (Ratcliff's `s_z = a(w₂ − w₁)`). Then
>
>     g(t | ν, η, a, [w₁,w₂]) = 1 / ((w₂ − w₁) √(2π t³ S)) · Σ_j (−1)^j I_j
>
> where `I_j = ∫_{w₁}^{w₂} (p_j + q_j w) exp(−A w²/2 + B_j w + C_j) dw` with the coefficient
> table of the note and the closed form `I` given there.

together with the note's remark on the resulting likelihood ("Consequence for the full model
likelihood"), which reduces the seven-parameter likelihood to a single one-dimensional integral
over the non-decision-time window.
-/

noncomputable section

open Real MeasureTheory Set Filter

namespace DDM

/-- The quantity `I_j` of Result 1 of the note `notes/closed_forms_sz_st.md`: the closed form
of `∫_{w₁}^{w₂} (p_j + q_j w) exp(−A w²/2 + B_j w + C_j) dw` given by Lemma L1 with the
coefficient table of Result 1. -/
def Iterm (nu eta a t w₁ w₂ : ℝ) (j : ℕ) : ℝ :=
  L1closed (pcoef a j) (qcoef a j) (Acoef a eta t) (Bcoef nu eta a t j) (Ccoef nu eta a t j)
    w₁ w₂

/-- The density with normal drift and uniform starting point,
`g(t | ν, η, a, [w₁,w₂]) = (1/(w₂−w₁)) ∫_{w₁}^{w₂} g(t | ν, η, a, w) dw`, the left-hand side
of Result 1 of the note `notes/closed_forms_sz_st.md`. -/
def densGEtaSz (nu eta a w₁ w₂ t : ℝ) : ℝ :=
  (1 / (w₂ - w₁)) * ∫ w in w₁..w₂, densGEta nu eta a w t

/-- The `j`-th term of Result 1 of the note `notes/closed_forms_sz_st.md` integrates in `w` to
`I_j`: this is Lemma L5 (the coefficient table) followed by Lemma L1 (the Gaussian moment). -/
theorem result1_term (nu eta a t w₁ w₂ : ℝ) (ha : a ≠ 0) (ht : 0 < t) (j : ℕ) :
    (∫ w in w₁..w₂, r a w j
        * exp ((-nu ^ 2 * t - 2 * nu * a * w + eta ^ 2 * a ^ 2 * w ^ 2) / (2 * Svar eta t)
            - (r a w j) ^ 2 / (2 * t)))
      = Iterm nu eta a t w₁ w₂ j := by
  rw [intervalIntegral.integral_congr (g := fun w => (pcoef a j + qcoef a j * w)
      * exp (-(Acoef a eta t) * w ^ 2 / 2 + Bcoef nu eta a t j * w + Ccoef nu eta a t j))
    (fun w _ => lemma_L5 nu eta a t w ht j)]
  exact lemma_L1 _ _ _ _ _ _ _ (Acoef_pos ha ht)

/-- The density series bound of the note `notes/closed_forms_sz_st.md`: for `w ∈ [0,1]` the
`j`-th term of the common series is bounded by `(j+1) a e^{−j²a²/(2t)}`. -/
theorem densSeries_term_abs_le {a w t : ℝ} (ha : 0 < a) (hw0 : 0 ≤ w) (hw1 : w ≤ 1) (ht : 0 < t)
    (j : ℕ) :
    |(-1 : ℝ) ^ j * r a w j * exp (-(r a w j) ^ 2 / (2 * t))|
      ≤ ((j : ℝ) + 1) * a * exp (-(j : ℝ) ^ 2 * (a ^ 2 / (2 * t))) := by
  have hr0 : 0 ≤ r a w j := r_nonneg ha.le hw0 hw1 j
  have hrle : r a w j ≤ ((j : ℝ) + 1) * a := r_le ha.le hw0 hw1 j
  have hrge : (j : ℝ) * a ≤ r a w j := le_r ha.le hw0 hw1 j
  have habs : |(-1 : ℝ) ^ j * r a w j * exp (-(r a w j) ^ 2 / (2 * t))|
      = r a w j * exp (-(r a w j) ^ 2 / (2 * t)) := by
    rw [abs_mul, abs_mul, abs_pow, abs_neg, abs_one, one_pow, one_mul,
      abs_of_nonneg hr0, abs_of_pos (Real.exp_pos _)]
  rw [habs]
  have hexp : exp (-(r a w j) ^ 2 / (2 * t)) ≤ exp (-(j : ℝ) ^ 2 * (a ^ 2 / (2 * t))) := by
    apply Real.exp_le_exp.mpr
    have hja : (0:ℝ) ≤ (j:ℝ) * a := by positivity
    have hsq : ((j:ℝ) * a) ^ 2 ≤ (r a w j) ^ 2 := by nlinarith
    have h2t : (0:ℝ) < 2 * t := by linarith
    rw [div_le_iff₀ h2t]
    have hrw : -(j : ℝ) ^ 2 * (a ^ 2 / (2 * t)) * (2 * t) = -((j : ℝ) * a) ^ 2 := by
      field_simp
    rw [hrw]
    nlinarith
  nlinarith [Real.exp_pos (-(j : ℝ) ^ 2 * (a ^ 2 / (2 * t))),
    Real.exp_pos (-(r a w j) ^ 2 / (2 * t))]

/-- The normal-drift density `(gη)` of the note `notes/closed_forms_sz_st.md` written with the
`w`-dependent prefactor absorbed into the series, the form in which the `w`-integral of
Result 1 is taken term by term. -/
theorem densGEta_eq_tsum (nu eta a w t : ℝ) :
    densGEta nu eta a w t
      = (∑' j : ℕ, (-1 : ℝ) ^ j * (r a w j
          * exp ((-nu ^ 2 * t - 2 * nu * a * w + eta ^ 2 * a ^ 2 * w ^ 2) / (2 * Svar eta t)
              - (r a w j) ^ 2 / (2 * t)))) / sqrt (2 * π * t ^ 3 * Svar eta t) := by
  simp only [densGEta, densSeries]
  congr 1
  rw [← tsum_mul_left]
  congr 1 with j
  rw [Real.exp_sub, show (-(r a w j) ^ 2 / (2 * t)) = -((r a w j) ^ 2 / (2 * t)) from by ring,
    Real.exp_neg]
  field_simp

/-- **Result 1 of the note `notes/closed_forms_sz_st.md`.**

The first-passage density with normally distributed drift `N(ν, η²)` and starting point
uniform on `[w₁, w₂]` has the closed form

    g(t | ν, η, a, [w₁,w₂]) = 1 / ((w₂ − w₁) √(2π t³ S)) · Σ_j (−1)^j I_j ,

with `I_j` the closed-form Gaussian moment of Lemma L1 evaluated at the coefficient table of
the note. -/
theorem result1 (nu eta a t w₁ w₂ : ℝ) (ha : 0 < a) (ht : 0 < t)
    (hw0 : 0 ≤ w₁) (hw : w₁ < w₂) (hw1 : w₂ ≤ 1) :
    densGEtaSz nu eta a w₁ w₂ t
      = 1 / ((w₂ - w₁) * sqrt (2 * π * t ^ 3 * Svar eta t))
          * ∑' j : ℕ, (-1 : ℝ) ^ j * Iterm nu eta a t w₁ w₂ j := by
  have hSpos : 0 < Svar eta t := Svar_pos ht
  have hle : w₁ ≤ w₂ := hw.le
  set P : ℝ → ℝ := fun w => (-nu ^ 2 * t - 2 * nu * a * w + eta ^ 2 * a ^ 2 * w ^ 2)
    / (2 * Svar eta t) with hPdef
  set g : ℕ → ℝ → ℝ := fun j w => (-1 : ℝ) ^ j * (r a w j * exp (P w - (r a w j) ^ 2 / (2 * t)))
    with hgdef
  set C := exp ((2 * |nu| * a + eta ^ 2 * a ^ 2) / (2 * Svar eta t)) with hCdef
  set M : ℕ → ℝ := fun j => ((j : ℝ) + 1) * a * exp (-(j : ℝ) ^ 2 * (a ^ 2 / (2 * t))) with hMdef
  have hcont : ∀ j, Continuous (g j) := by
    intro j
    simp only [hgdef, hPdef, r]
    split <;> fun_prop
  have hint : ∀ j, IntegrableOn (g j) (Ioc w₁ w₂) := fun j => (hcont j).integrableOn_Ioc
  have hMnonneg : ∀ j, 0 ≤ M j := by
    intro j; simp only [hMdef]; positivity
  have hbound : ∀ j, ∀ w ∈ Ioc w₁ w₂, ‖g j w‖ ≤ C * M j := by
    intro j w hwm
    have h0 : 0 ≤ w := le_trans hw0 hwm.1.le
    have h1 : w ≤ 1 := le_trans hwm.2 hw1
    have hpre : exp (P w) ≤ C := by
      simp only [hPdef, hCdef]
      apply Real.exp_le_exp.mpr
      rw [div_le_div_iff_of_pos_right (by positivity)]
      have h2 : -nu ≤ |nu| := neg_le_abs nu
      have h4 : 0 ≤ |nu| := abs_nonneg nu
      have haw : 0 ≤ a * w := mul_nonneg ha.le h0
      have hA : -2 * nu * a * w ≤ 2 * |nu| * a := by
        have k1 : (-nu) * (a * w) ≤ |nu| * (a * w) := mul_le_mul_of_nonneg_right h2 haw
        have k2 : |nu| * (a * w) ≤ |nu| * a := mul_le_mul_of_nonneg_left (by nlinarith) h4
        nlinarith
      have hw2 : w ^ 2 ≤ 1 := by nlinarith
      have hB : eta ^ 2 * a ^ 2 * w ^ 2 ≤ eta ^ 2 * a ^ 2 := by nlinarith [sq_nonneg (eta * a)]
      have hC0 : -nu ^ 2 * t ≤ 0 := by nlinarith [sq_nonneg nu]
      linarith
    have hser := densSeries_term_abs_le ha h0 h1 ht j
    have hrewrite : g j w
        = ((-1 : ℝ) ^ j * r a w j * exp (-(r a w j) ^ 2 / (2 * t))) * exp (P w) := by
      simp only [hgdef]
      rw [Real.exp_sub, show (-(r a w j) ^ 2 / (2 * t)) = -((r a w j) ^ 2 / (2 * t)) from by ring,
        Real.exp_neg]
      field_simp
    rw [Real.norm_eq_abs, hrewrite, abs_mul, abs_of_pos (Real.exp_pos _)]
    have habs : 0 ≤ |(-1 : ℝ) ^ j * r a w j * exp (-(r a w j) ^ 2 / (2 * t))| := abs_nonneg _
    calc |(-1 : ℝ) ^ j * r a w j * exp (-(r a w j) ^ 2 / (2 * t))| * exp (P w)
        ≤ M j * exp (P w) := by nlinarith [Real.exp_pos (P w)]
      _ ≤ M j * C := by nlinarith [hMnonneg j]
      _ = C * M j := by ring
  have hsummable : Summable fun j => ∫ w in Ioc w₁ w₂, ‖g j w‖ := by
    have hb : Summable fun j : ℕ => (w₂ - w₁) * (C * M j) := by
      apply Summable.mul_left
      apply Summable.mul_left
      simpa [hMdef, mul_comm, mul_assoc, mul_left_comm] using
        ((summable_poly_gaussian (a ^ 2 / (2 * t)) (by positivity)).mul_right a)
    apply Summable.of_nonneg_of_le
      (fun j => integral_nonneg (fun w => norm_nonneg _)) _ hb
    intro j
    have h1 : (∫ w in Ioc w₁ w₂, ‖g j w‖) ≤ ∫ _w in Ioc w₁ w₂, C * M j := by
      refine setIntegral_mono_on ((hint j).norm) ?_ measurableSet_Ioc (hbound j)
      apply integrableOn_const
      · rw [Real.volume_Ioc]; exact ENNReal.ofReal_ne_top
      · exact enorm_ne_top
    have h2 : (∫ _w in Ioc w₁ w₂, C * M j) = (w₂ - w₁) * (C * M j) := by
      rw [MeasureTheory.setIntegral_const, Measure.real, Real.volume_Ioc,
        ENNReal.toReal_ofReal (by linarith : (0:ℝ) ≤ w₂ - w₁), smul_eq_mul]
    linarith [h1, h2.le, h2.ge]
  have hexch := intervalIntegral_tsum_of_summable hle hint hsummable
  have hterm : ∀ j, (∫ w in w₁..w₂, g j w) = (-1 : ℝ) ^ j * Iterm nu eta a t w₁ w₂ j := by
    intro j
    simp only [hgdef]
    rw [intervalIntegral.integral_const_mul]
    congr 1
    exact result1_term nu eta a t w₁ w₂ (ne_of_gt ha) ht j
  simp only [densGEtaSz]
  rw [intervalIntegral.integral_congr (g := fun w => (∑' j : ℕ, g j w)
      / sqrt (2 * π * t ^ 3 * Svar eta t)) (fun w _ => densGEta_eq_tsum nu eta a w t),
    intervalIntegral.integral_div, hexch, tsum_congr hterm]
  have hK : (0:ℝ) < sqrt (2 * π * t ^ 3 * Svar eta t) := by positivity
  field_simp

end DDM
