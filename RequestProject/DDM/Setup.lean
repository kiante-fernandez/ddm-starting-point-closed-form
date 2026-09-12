import RequestProject.DDM.Normal

/-!
# Setup: the Ratcliff diffusion model series of the note

This file formalizes the **Setup** section of the note `notes/closed_forms_sz_st.md`
("Closed forms for starting-point and non-decision-time variability in the Ratcliff diffusion
model").

The setup of the note is: a Wiener process with unit diffusion coefficient, absorbing barriers
at `0` and `a`, starting point `a·w`, drift `v`; all quantities are lower-barrier quantities.
The note defines

    r_j = j a + a w          (j even)
    r_j = (j+1) a − a w      (j odd)

and the two series it works with,

    g(t | v, a, w)     = (2π t³)^{−1/2} e^{−vaw − v²t/2} Σ_j (−1)^j r_j e^{−r_j²/(2t)}      (g)

    g(t | ν, η, a, w)  = (2π t³ S)^{−1/2} exp[(−ν²t − 2νaw + η²a²w²)/(2S)]
                            Σ_j (−1)^j r_j e^{−r_j²/(2t)},        S := 1 + η² t             (gη)

As in the reference implementation `src/ddm_closed.py` accompanying the note, the summation
index `j` runs over the non-negative integers.

The file also records the elementary bounds `j a ≤ r_j ≤ (j+1) a` valid for `w ∈ [0,1]`, which
are the quantitative form of the note's remark that "in every series the j-th term is
`O(exp(−j²a²/(2t)))`, so summation and integration may be exchanged freely".
-/

noncomputable section

open Real MeasureTheory Set Filter

namespace DDM

/-- The barrier distances `r_j` of the note `notes/closed_forms_sz_st.md`:
`r_j = j a + a w` for even `j` and `r_j = (j+1) a − a w` for odd `j`. -/
def r (a w : ℝ) (j : ℕ) : ℝ := if Even j then j * a + a * w else (j + 1) * a - a * w

/-- `S := 1 + η² t`, the quantity called `S` in the note `notes/closed_forms_sz_st.md`. -/
def Svar (eta t : ℝ) : ℝ := 1 + eta ^ 2 * t

lemma Svar_pos {eta t : ℝ} (ht : 0 < t) : 0 < Svar eta t := by
  have : 0 ≤ eta ^ 2 * t := by positivity
  simp only [Svar]; linarith

/-- Lower bound `j a ≤ r_j` for a relative starting point `w ∈ [0,1]`. -/
lemma le_r {a w : ℝ} (ha : 0 ≤ a) (hw0 : 0 ≤ w) (hw1 : w ≤ 1) (j : ℕ) : (j : ℝ) * a ≤ r a w j := by
  unfold r
  split
  · nlinarith
  · nlinarith

/-- Upper bound `r_j ≤ (j+1) a` for a relative starting point `w ∈ [0,1]`. -/
lemma r_le {a w : ℝ} (ha : 0 ≤ a) (hw0 : 0 ≤ w) (hw1 : w ≤ 1) (j : ℕ) :
    r a w j ≤ ((j : ℝ) + 1) * a := by
  unfold r
  split
  · nlinarith
  · nlinarith

lemma r_nonneg {a w : ℝ} (ha : 0 ≤ a) (hw0 : 0 ≤ w) (hw1 : w ≤ 1) (j : ℕ) : 0 ≤ r a w j := by
  have := le_r ha hw0 hw1 j
  have : (0:ℝ) ≤ (j:ℝ) * a := by positivity
  linarith [le_r ha hw0 hw1 j]

/-- The series `Σ_j (−1)^j r_j e^{−r_j²/(2t)}` common to the two densities `(g)` and `(gη)`
of the note `notes/closed_forms_sz_st.md`. -/
def densSeries (a w t : ℝ) : ℝ :=
  ∑' j : ℕ, (-1 : ℝ) ^ j * r a w j * exp (-(r a w j) ^ 2 / (2 * t))

/-- The constant-drift first-passage density `(g)` of the note `notes/closed_forms_sz_st.md`:

    g(t | v, a, w) = (2π t³)^{−1/2} e^{−vaw − v²t/2} Σ_j (−1)^j r_j e^{−r_j²/(2t)}. -/
def densG (v a w t : ℝ) : ℝ :=
  exp (-v * a * w - v ^ 2 * t / 2) * densSeries a w t / sqrt (2 * π * t ^ 3)

/-- The first-passage density with normally distributed drift `N(ν, η²)`, equation `(gη)` of the
note `notes/closed_forms_sz_st.md` (Horrocks & Thompson 2004; Blurton et al. 2017 Eq. 1):

    g(t | ν, η, a, w) = (2π t³ S)^{−1/2} exp[(−ν²t − 2νaw + η²a²w²)/(2S)]
                          Σ_j (−1)^j r_j e^{−r_j²/(2t)}. -/
def densGEta (nu eta a w t : ℝ) : ℝ :=
  exp ((-nu ^ 2 * t - 2 * nu * a * w + eta ^ 2 * a ^ 2 * w ^ 2) / (2 * Svar eta t))
    * densSeries a w t / sqrt (2 * π * t ^ 3 * Svar eta t)

/-- The `N(ν, η²)` drift density, the across-trial drift distribution of the note
`notes/closed_forms_sz_st.md`. -/
def driftPdf (nu eta v : ℝ) : ℝ := exp (-(v - nu) ^ 2 / (2 * eta ^ 2)) / (eta * sqrt (2 * π))

end DDM
