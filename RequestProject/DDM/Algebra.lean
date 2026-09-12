import RequestProject.DDM.Setup

/-!
# The algebraic lemmas L4, L5, L6 of the note

This file formalizes the remaining symbolic checks of the note `notes/closed_forms_sz_st.md`
(the table of `verify/symbolic_proof.py`):

| check | content |
|---|---|
| L4 | `a²/(2t) − η²a²/(2S) = a²/(2tS) > 0` — the inequality the whole result rests on |
| L5e, L5o | the even-`j` and odd-`j` coefficient tables of Result 1 |
| L6 | the drift integral of the constant-drift density reproduces Blurton Eq. 1 |
-/

noncomputable section

open Real MeasureTheory Set Filter

namespace DDM

/-! ## Lemma L4 -/

/-- **Lemma L4 of the note `notes/closed_forms_sz_st.md`, the identity part.**

    a²/(2t) − η²a²/(2S) = a² (S − η²t) / (2tS) = a²/(2tS),      S = 1 + η² t. -/
theorem lemma_L4_identity (a eta t : ℝ) (ht : 0 < t) :
    a ^ 2 / (2 * t) - eta ^ 2 * a ^ 2 / (2 * Svar eta t) = a ^ 2 / (2 * t * Svar eta t) := by
  have hS : 0 < Svar eta t := Svar_pos ht
  simp only [Svar] at hS ⊢
  field_simp
  ring

/-- **Lemma L4 of the note `notes/closed_forms_sz_st.md`, the positivity part.**

The combined exponent of the `j`-th term of Result 1 is a *decaying* Gaussian in `w` for every
`j` and every `t`, because `a²/(2t) − η²a²/(2S) = a²/(2tS) > 0`. -/
theorem lemma_L4_pos (a eta t : ℝ) (ha : a ≠ 0) (ht : 0 < t) :
    0 < a ^ 2 / (2 * t) - eta ^ 2 * a ^ 2 / (2 * Svar eta t) := by
  rw [lemma_L4_identity a eta t ht]
  have hS : 0 < Svar eta t := Svar_pos ht
  have : 0 < a ^ 2 := by positivity
  positivity

/-! ## Lemma L5: the coefficient tables -/

/-- The coefficient `A = a²/(tS)` of Result 1 of the note `notes/closed_forms_sz_st.md`
(the same for all `j`, always `> 0`). -/
def Acoef (a eta t : ℝ) : ℝ := a ^ 2 / (t * Svar eta t)

/-- The coefficient `p_j` of Result 1 of the note `notes/closed_forms_sz_st.md`:
`p_j = j a` for even `j`, `p_j = (j+1) a` for odd `j`. -/
def pcoef (a : ℝ) (j : ℕ) : ℝ := if Even j then (j : ℝ) * a else ((j : ℝ) + 1) * a

/-- The coefficient `q_j` of Result 1 of the note `notes/closed_forms_sz_st.md`:
`q_j = a` for even `j`, `q_j = −a` for odd `j`. -/
def qcoef (a : ℝ) (j : ℕ) : ℝ := if Even j then a else -a

/-- The coefficient `B_j` of Result 1 of the note `notes/closed_forms_sz_st.md`:
`B_j = −νa/S − j a²/t` for even `j`, `B_j = −νa/S + (j+1) a²/t` for odd `j`. -/
def Bcoef (nu eta a t : ℝ) (j : ℕ) : ℝ :=
  if Even j then -nu * a / Svar eta t - (j : ℝ) * a ^ 2 / t
  else -nu * a / Svar eta t + ((j : ℝ) + 1) * a ^ 2 / t

/-- The coefficient `C_j` of Result 1 of the note `notes/closed_forms_sz_st.md`:
`C_j = −ν²t/(2S) − j²a²/(2t)` for even `j`, `C_j = −ν²t/(2S) − (j+1)²a²/(2t)` for odd `j`. -/
def Ccoef (nu eta a t : ℝ) (j : ℕ) : ℝ :=
  if Even j then -nu ^ 2 * t / (2 * Svar eta t) - (j : ℝ) ^ 2 * a ^ 2 / (2 * t)
  else -nu ^ 2 * t / (2 * Svar eta t) - ((j : ℝ) + 1) ^ 2 * a ^ 2 / (2 * t)

lemma Acoef_pos {a eta t : ℝ} (ha : a ≠ 0) (ht : 0 < t) : 0 < Acoef a eta t := by
  have hS : 0 < Svar eta t := Svar_pos ht
  have : 0 < a ^ 2 := by positivity
  unfold Acoef; positivity

/-- **Lemma L5e of the note `notes/closed_forms_sz_st.md`** (check `L5e` of
`verify/symbolic_proof.py`): the even-`j` coefficient table of Result 1.  For even `j`, the
`w`-dependent part of the `j`-th term of `(gη)` is

    r_j exp[(−ν²t − 2νaw + η²a²w²)/(2S) − r_j²/(2t)] = (p_j + q_j w) exp(−A w²/2 + B_j w + C_j). -/
theorem lemma_L5_even (nu eta a t w : ℝ) (ht : 0 < t) {j : ℕ} (hj : Even j) :
    r a w j * exp ((-nu ^ 2 * t - 2 * nu * a * w + eta ^ 2 * a ^ 2 * w ^ 2) / (2 * Svar eta t)
        - (r a w j) ^ 2 / (2 * t))
      = (pcoef a j + qcoef a j * w)
          * exp (-(Acoef a eta t) * w ^ 2 / 2 + Bcoef nu eta a t j * w + Ccoef nu eta a t j) := by
  have hS : 0 < Svar eta t := Svar_pos ht
  have hr : r a w j = pcoef a j + qcoef a j * w := by
    simp only [r, pcoef, qcoef, if_pos hj]
  rw [hr]
  congr 1
  simp only [Acoef, Bcoef, Ccoef, pcoef, qcoef, if_pos hj]
  simp only [Svar] at hS ⊢
  field_simp
  ring_nf

/-- **Lemma L5o of the note `notes/closed_forms_sz_st.md`** (check `L5o` of
`verify/symbolic_proof.py`): the odd-`j` coefficient table of Result 1. -/
theorem lemma_L5_odd (nu eta a t w : ℝ) (ht : 0 < t) {j : ℕ} (hj : ¬ Even j) :
    r a w j * exp ((-nu ^ 2 * t - 2 * nu * a * w + eta ^ 2 * a ^ 2 * w ^ 2) / (2 * Svar eta t)
        - (r a w j) ^ 2 / (2 * t))
      = (pcoef a j + qcoef a j * w)
          * exp (-(Acoef a eta t) * w ^ 2 / 2 + Bcoef nu eta a t j * w + Ccoef nu eta a t j) := by
  have hS : 0 < Svar eta t := Svar_pos ht
  have hr : r a w j = pcoef a j + qcoef a j * w := by
    simp only [r, pcoef, qcoef, if_neg hj]; ring
  rw [hr]
  congr 1
  simp only [Acoef, Bcoef, Ccoef, pcoef, qcoef, if_neg hj]
  simp only [Svar] at hS ⊢
  field_simp
  ring_nf

/-- The coefficient table of Result 1 of the note `notes/closed_forms_sz_st.md`, both
parities at once (`L5e` and `L5o` combined). -/
theorem lemma_L5 (nu eta a t w : ℝ) (ht : 0 < t) (j : ℕ) :
    r a w j * exp ((-nu ^ 2 * t - 2 * nu * a * w + eta ^ 2 * a ^ 2 * w ^ 2) / (2 * Svar eta t)
        - (r a w j) ^ 2 / (2 * t))
      = (pcoef a j + qcoef a j * w)
          * exp (-(Acoef a eta t) * w ^ 2 / 2 + Bcoef nu eta a t j * w + Ccoef nu eta a t j) := by
  by_cases hj : Even j
  · exact lemma_L5_even nu eta a t w ht hj
  · exact lemma_L5_odd nu eta a t w ht hj

/-! ## Lemma L6: the drift integral -/

/-- The Gaussian integral behind Lemma L6 of the note `notes/closed_forms_sz_st.md`:
averaging `e^{−vc − v²t/2}` over `v ~ N(ν, η²)` gives

    S^{−1/2} exp[(−ν²t − 2νc + η²c²)/(2S)],     S = 1 + η² t. -/
theorem drift_gaussian_average (nu eta c t : ℝ) (heta : 0 < eta) (ht : 0 ≤ t) :
    (∫ v : ℝ, driftPdf nu eta v * exp (-v * c - v ^ 2 * t / 2))
      = exp ((-nu ^ 2 * t - 2 * nu * c + eta ^ 2 * c ^ 2) / (2 * Svar eta t))
          / sqrt (Svar eta t) := by
  have hS : 0 < Svar eta t := by
    have : 0 ≤ eta ^ 2 * t := by positivity
    simp only [Svar]; linarith
  set S := Svar eta t with hSdef
  set b := S / (2 * eta ^ 2) with hb
  have hbpos : 0 < b := by positivity
  set m := (nu - eta ^ 2 * c) / S with hm
  set Q := (-nu ^ 2 * t - 2 * nu * c + eta ^ 2 * c ^ 2) / (2 * S) with hQ
  have hpoint : ∀ v : ℝ, driftPdf nu eta v * exp (-v * c - v ^ 2 * t / 2)
      = (exp Q / (eta * sqrt (2 * π))) * exp (-b * (v - m) ^ 2) := by
    intro v
    simp only [driftPdf]
    rw [div_mul_eq_mul_div, ← Real.exp_add, div_mul_eq_mul_div, ← Real.exp_add]
    congr 1
    congr 1
    simp only [hb, hm, hQ, hSdef, Svar]
    field_simp
    ring
  rw [MeasureTheory.integral_congr_ae (Filter.Eventually.of_forall hpoint)]
  rw [MeasureTheory.integral_const_mul]
  have h1 : (∫ v : ℝ, exp (-b * (v - m) ^ 2)) = sqrt (π / b) := by
    rw [MeasureTheory.integral_sub_right_eq_self (fun x : ℝ => exp (-b * x ^ 2)) m]
    exact integral_gaussian b
  rw [h1]
  have hsq : sqrt (π / b) = eta * sqrt (2 * π) / sqrt S := by
    rw [hb]
    have h : π / (S / (2 * eta ^ 2)) = (eta ^ 2 * (2 * π)) / S := by field_simp
    rw [h, Real.sqrt_div (by positivity), Real.sqrt_mul (by positivity), Real.sqrt_sq heta.le]
  rw [hsq]
  field_simp

/-- **Lemma L6 of the note `notes/closed_forms_sz_st.md`** (check `L6` of
`verify/symbolic_proof.py`): integrating the constant-drift density `(g)` against the
`N(ν, η²)` drift density reproduces the normal-drift density `(gη)` (Blurton et al. 2017,
Eq. 1). -/
theorem lemma_L6 (nu eta a w t : ℝ) (heta : 0 < eta) (ht : 0 < t) :
    (∫ v : ℝ, driftPdf nu eta v * densG v a w t) = densGEta nu eta a w t := by
  have hS : 0 < Svar eta t := Svar_pos ht
  have hpoint : ∀ v : ℝ, driftPdf nu eta v * densG v a w t
      = (densSeries a w t / sqrt (2 * π * t ^ 3))
          * (driftPdf nu eta v * exp (-v * (a * w) - v ^ 2 * t / 2)) := by
    intro v
    simp only [densG]
    ring_nf
  rw [MeasureTheory.integral_congr_ae (Filter.Eventually.of_forall hpoint),
    MeasureTheory.integral_const_mul, drift_gaussian_average nu eta (a * w) t heta ht.le]
  simp only [densGEta]
  rw [show (2 * π * t ^ 3 * Svar eta t) = (2 * π * t ^ 3) * (Svar eta t) by ring,
    Real.sqrt_mul (by positivity)]
  have h1 : eta ^ 2 * (a * w) ^ 2 = eta ^ 2 * a ^ 2 * w ^ 2 := by ring
  rw [h1]
  field_simp
  rw [show (2:ℝ) * π * t ^ 3 * Svar eta t = (2 * π) * (t ^ 3) * (Svar eta t) by ring,
    Real.sqrt_mul (by positivity), Real.sqrt_mul (by positivity)]
  ring

end DDM
