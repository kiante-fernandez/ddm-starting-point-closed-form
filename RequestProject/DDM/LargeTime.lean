import RequestProject.DDM.Faddeeva

/-!
# The Faddeeva bracket of the large-time representation

The note `notes/closed_forms_sz_st.md`, in the section *"Large-time representation of
Result 1"*, writes the starting-point integral of each term of the large-time series as

    Im{ (−i)(√π/(2√κ)) [ e^{κw₂² + μ_k w₂} w(z_k(w₂)) − e^{κw₁² + μ_k w₁} w(z_k(w₁)) ] },
    z_k(w) = √κ w + μ_k/(2√κ),

with `w(z)` the Faddeeva function, and remarks that "for `η = 0` the bracket reduces to
`(e^{μw₂} − e^{μw₁})/μ`".

This file proves both statements as exact identities about the underlying `w`-integral:

* `DDM.integral_cexp_quadratic` — for any complex `κ = √κ²` with `√κ ≠ 0` and any complex `μ`,

      ∫_{w₁}^{w₂} e^{κ w² + μ w} dw
        = (−i)(√π/(2√κ)) [ e^{κ w² + μ w} w(√κ w + μ/(2√κ)) ]_{w₁}^{w₂};

  taking imaginary parts gives the note's formula term by term.
* `DDM.integral_cexp_linear` — the degenerate case `κ = 0` (i.e. `η = 0`) of the note.

The remaining ingredients of the note's large-time formula — the large-time (Ratcliff 1978)
series itself, which the note takes from the literature, and the boundedness of `w(z)` in the
upper half plane which underlies its numerical remark — are not formalized here.
-/

noncomputable section

open Real MeasureTheory Set Filter

namespace DDM

/-- The Faddeeva bracket of the note `notes/closed_forms_sz_st.md`,

    (−i)(√π/(2√κ)) e^{κ w² + μ w} w(√κ w + μ/(2√κ)),

as a function of the (real) relative starting point `w`.  Here `sk` plays the role of `√κ`. -/
def gaussBracket (kappa mu sk : ℂ) (w : ℝ) : ℂ :=
  (-Complex.I) * ((Real.sqrt π : ℂ) / (2 * sk))
    * Complex.exp (kappa * (w : ℂ) ^ 2 + mu * (w : ℂ))
    * faddeeva (sk * (w : ℂ) + mu / (2 * sk))

/-- **The Faddeeva bracket is an antiderivative of `e^{κ w² + μ w}`** in the real variable `w`.
This is the computation behind the large-time representation of Result 1 in the note
`notes/closed_forms_sz_st.md`: the `w`-dependent terms cancel because `2√κ z(w) = 2κ w + μ`. -/
theorem hasDerivAt_gaussBracket (kappa mu sk : ℂ) (hsk : sk ≠ 0) (hk : sk ^ 2 = kappa) (w : ℝ) :
    HasDerivAt (gaussBracket kappa mu sk)
      (Complex.exp (kappa * (w : ℂ) ^ 2 + mu * (w : ℂ))) w := by
  have hpi : ((Real.sqrt π : ℝ) : ℂ) ≠ 0 :=
    Complex.ofReal_ne_zero.mpr (Real.sqrt_ne_zero'.mpr Real.pi_pos)
  have hofReal : HasDerivAt (fun w : ℝ => (w : ℂ)) 1 w := Complex.ofRealCLM.hasDerivAt
  -- the affine argument of the Faddeeva function
  have hz : HasDerivAt (fun w : ℝ => sk * (w : ℂ) + mu / (2 * sk)) sk w := by
    simpa using (hofReal.const_mul sk).add_const (mu / (2 * sk))
  have hW : HasDerivAt (fun w : ℝ => faddeeva (sk * (w : ℂ) + mu / (2 * sk)))
      (sk * (-2 * (sk * (w : ℂ) + mu / (2 * sk))
        * faddeeva (sk * (w : ℂ) + mu / (2 * sk)) + 2 * Complex.I / (Real.sqrt π : ℂ))) w := by
    have h := (hasDerivAt_faddeeva (sk * (w : ℂ) + mu / (2 * sk))).scomp w hz
    simpa [Function.comp, smul_eq_mul] using h
  -- the Gaussian prefactor
  have hquad : HasDerivAt (fun w : ℝ => kappa * (w : ℂ) ^ 2 + mu * (w : ℂ))
      (2 * kappa * (w : ℂ) + mu) w := by
    have h1 : HasDerivAt (fun w : ℝ => (w : ℂ) ^ 2) (2 * (w : ℂ)) w := by
      simpa using hofReal.pow 2
    have h2 := (h1.const_mul kappa).add (hofReal.const_mul mu)
    convert h2 using 1
    ring
  have hE : HasDerivAt (fun w : ℝ => Complex.exp (kappa * (w : ℂ) ^ 2 + mu * (w : ℂ)))
      (Complex.exp (kappa * (w : ℂ) ^ 2 + mu * (w : ℂ)) * (2 * kappa * (w : ℂ) + mu)) w := by
    simpa using hquad.cexp
  have hprod := (hE.mul hW).const_mul ((-Complex.I) * ((Real.sqrt π : ℂ) / (2 * sk)))
  have hprod' := hprod.congr_of_eventuallyEq (f₁ := gaussBracket kappa mu sk)
    (Filter.Eventually.of_forall fun w => by simp only [gaussBracket, Pi.mul_apply]; ring)
  convert hprod' using 1
  subst hk
  field_simp
  ring_nf
  rw [Complex.I_sq]
  ring

/-- **The `w`-integral of the large-time representation** of Result 1 in the note
`notes/closed_forms_sz_st.md`:

    ∫_{w₁}^{w₂} e^{κ w² + μ w} dw
      = (−i)(√π/(2√κ)) [ e^{κ w² + μ w} w(√κ w + μ/(2√κ)) ]_{w₁}^{w₂},

for any complex square root `√κ ≠ 0` of `κ`.  The note's formula is the imaginary part of this
identity, applied to `μ_k = λ + i k π`. -/
theorem integral_cexp_quadratic (kappa mu sk : ℂ) (hsk : sk ≠ 0) (hk : sk ^ 2 = kappa)
    (w₁ w₂ : ℝ) :
    (∫ w in w₁..w₂, Complex.exp (kappa * (w : ℂ) ^ 2 + mu * (w : ℂ)))
      = gaussBracket kappa mu sk w₂ - gaussBracket kappa mu sk w₁ := by
  have hcont : Continuous fun w : ℝ => Complex.exp (kappa * (w : ℂ) ^ 2 + mu * (w : ℂ)) := by
    fun_prop
  exact (intervalIntegral.integral_eq_sub_of_hasDerivAt
    (fun w _ => hasDerivAt_gaussBracket kappa mu sk hsk hk w)
    (hcont.intervalIntegrable w₁ w₂))

/-- **The `η = 0` degeneration** of the bracket, as recorded in the note
`notes/closed_forms_sz_st.md`: for `κ = 0` the `w`-integral reduces to
`(e^{μ w₂} − e^{μ w₁})/μ`. -/
theorem integral_cexp_linear (mu : ℂ) (hmu : mu ≠ 0) (w₁ w₂ : ℝ) :
    (∫ w in w₁..w₂, Complex.exp (mu * (w : ℂ)))
      = (Complex.exp (mu * (w₂ : ℂ)) - Complex.exp (mu * (w₁ : ℂ))) / mu := by
  have hofReal : ∀ w : ℝ, HasDerivAt (fun w : ℝ => (w : ℂ)) 1 w :=
    fun w => Complex.ofRealCLM.hasDerivAt
  have hderiv : ∀ w : ℝ, HasDerivAt (fun w : ℝ => Complex.exp (mu * (w : ℂ)) / mu)
      (Complex.exp (mu * (w : ℂ))) w := by
    intro w
    have h1 : HasDerivAt (fun w : ℝ => mu * (w : ℂ)) mu w := by
      simpa using (hofReal w).const_mul mu
    have h2 : HasDerivAt (fun w : ℝ => Complex.exp (mu * (w : ℂ)))
        (Complex.exp (mu * (w : ℂ)) * mu) w := by
      simpa using h1.cexp
    have h3 := h2.div_const mu
    convert h3 using 1
    field_simp
  have hcont : Continuous fun w : ℝ => Complex.exp (mu * (w : ℂ)) := by fun_prop
  have h := intervalIntegral.integral_eq_sub_of_hasDerivAt (fun w _ => hderiv w)
    (hcont.intervalIntegrable w₁ w₂)
  rw [h]
  ring

end DDM
