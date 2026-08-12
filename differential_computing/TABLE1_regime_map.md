# Table 1 — The Regime Map (framework paper spine)

**Symbols.**
`θ` parameter · `∇C` target = gradient of the *noisy* device landscape ·
`T` evolution time · `Γ=1/T₂` dephasing rate ·
`δ` control setpoint-error (std) · `ε` FD step · `Δt` control time-slice ·
`N` shots · `ε_g` target gradient RMSE · `m` generator terms an extensive `θ` touches ·
`M` Nyquist truncation order (# shift pairs) · `v̄` typical `|v_j|` ·
`v_j(t)=∂u_j/∂θ` · `A(t)=Σ_j v_j(t) H_j` (tangent) · `diam=λ_max−λ_min` ·
**`K = (1/2πħ)∫₀ᵀ diam(A(t)) dt`** (Nyquist bandwidth — a compile-time static
analysis over the tangent's spectral diameter; `ħ=1`; time-independent case
`K=(T/2π)diam(A)`).
Cells: **✓** holds · **✗** fails · **✓†** numerically holds, proof open · **†** open ·
superscript **ˢ** = stochastic Nyquist variant.

| Axis | Kick PSR | Nyquist | FD |
|---|:--:|:--:|:--:|
| **— requirements —** | | | |
| generator req. | `H²=I` **& separately synth.** (e.g. Pauli) | `diam(A)<∞` (any bounded Hermitian) | none |
| hardware req. | digital op + gate zone | amplitude headroom `s₀=1/4K` | none |
| fine param, `Δt→0` | ✓ | ✗ `(s₀∝1/Δt)` | ✓ |
| **— cost —** | | | |
| executions / grad | `O(m)` pairs | `O(M) / O(1)ˢ` | `O(1)` |
| variance | `O(T²v̄²)` | `O(K²)ˢ` | `O(1/ε²)` |
| **shots @ extensive `θ`** | `O(m²T²v̄²·e^{2ΓT}/ε_g²)` | `O(K²·e^{2ΓT}/ε_g²)` | `m`-indep, **floored** |
| compilation | analog waveforms + gates (incl. transport) | waveforms only | waveforms only |
| **— guarantees —** | | | |
| bias | `0` | `O(K/M) / 0ˢ` | `O(ε²)+O(δ/ε)`, floor `O(δ^{2/3})` |
| unbiased on noisy `∇C` | ✓ | ✓† | ✗ |
| coherent-error suppr. | ✓† `O(η²)` | † | ✗ |

*Caption.* Kick `O(T²v̄²)` assumes the `|v_j|≈O(1)` normalization (`v̄`). Nyquist
variance `O(K²)` is the **stochastic** form `(Σ|w_n|)²∼K²`; the deterministic
truncation splits the budget over `Σw_n²` and is higher by the order factor `M`.
FD cells are all at **free `ε`** (the knob); its optimized floor lives in the bias
row (`ε*∼δ^{1/3}⇒O(δ^{2/3})`), which is also why "no shots reach `ε_g<δ^{2/3}`".

---

### The extensive-`θ` crossover (headline — F3a verifies)

Same-dimension shot counts: **kick `∼ m²T²v̄²/ε_g²`** (budget split over `m` branch
pairs, variances add) vs **Nyquist `∼ K²/ε_g²`** with `K=(1/2π)∫diam(A)dt`. The
generator's `O(m)` branch count *follows from* the separately-synthesizable
condition (the transformation splits the sum term-by-term); Nyquist folds the
whole sum into one tangent direction. The verdict is `diam(A)` vs `m·v̄`:

`diam(Σ_j v_j H_j)` is **subadditive** (`≤ Σ_j|v_j|diam(H_j)`), so:

| tangent structure | `diam(A)` (verified) | `K` | Nyquist shots | vs kick `m²T²v̄²` |
|---|:--:|:--:|:--:|:--:|
| uniform `ΣZ_j` / same-sign | `2mv̄` (extensive) | `∝m` | `∝m²` | wins by `∼π²` const |
| frustrated ZZ / overlapping | `<2Σ|v_j|` | `<m` | `<m²` | wins by more |
| telescoping `Σ(Z_j−Z_{j+1})` | `4` (**O(1)**, all `m`) | `O(T)` | **`m`-indep** | wins by `∼m²` |

So on **shots**, extensive `θ` favors **Nyquist** (always `≤` kick, and
`m`-independent when the tangent spectral diameter is *subextensive*). Kick's wins
lie elsewhere — coherent-error robustness, fine-grained `Δt` params, and a single
**large-`|v|`** tangent (`|v|>π ⇒ K>T ⇒ O(T²)<O(K²)`). Sign-alternation on
*independent single-body* terms does **not** help (`diam` still `2mv̄`);
subextensivity needs *cancelling/telescoping* structure.

### `generator req.` — boson / fermion

Kick needs a **finite, few-level equidistant spectrum** for an exact finite shift
rule; `H²=I` (Pauli strings, single Majoranas) is the ±1 case, and the generator
must be **separately synthesizable** as a gate.
- **Fermion**: number `n_i` (`{0,1}`) and hopping (`{-1,0,1}`) are equidistant with
  few levels → work via the *generalized* multi-term shift rule (not the 2-term
  kick). ✓ with more terms.
- **Boson**: `n=a†a` (`0,1,2,…`, unbounded) → no finite shift rule → kick ✗. Nyquist
  needs `diam(A)<∞`, so it too needs a photon-number cutoff; with one it works.

### `coherent-error suppr.` — meaning & status (open)

Coherent/slow errors = *systematic* miscalibration / drift (gate over-rotation,
control drift), **not** shot noise. "Suppression" = the estimator's difference
cancels them common-mode (an echo).
- **Kick** = ✓†: for a symmetric ±kick a common over-rotation is `O(η²)` (measured
  slope 2.00, `coherent_error_check.py`) — the ± branches share the *same* base at
  the *same* parameter, so the shift sits at the response extremum. The *general*
  claim rests on Leng Lemma 3.3's strength (proof open).
- **FD** = ✗: a miscalibrated step scales the estimate `∝(1+η)` → `O(η)` (slope 1.00).
- **Nyquist** = † (open): a *multiplicative* shift miscalibration is `O(η)` (slope
  0.99, like FD — its `±s` branches sit at *different* operating points `θ±s`). But
  an *additive, waveform-independent* slow error is common to the `±s` pair and
  plausibly cancels like the kick's ± branches — **not yet checked**. Do not print
  ✗ until the additive case is resolved.

### `δ`-amplification when `ε ≈ δ`  (`figures/delta_amplification.png`)

FD differences two shifted setpoints, each with error `δ`, so the `δ` term is
`(δ/ε)·|C'|` — amplified by `1/ε`. As `ε→δ` it reaches `O(|∇C|)`: at a steep
near-inflection point (`∇C=1.07`, `δ=0.02`) FD error is `0.75≈0.7|∇C|` at `ε=δ`
(sign unreliable), doubling per halving of `ε`. Kick/Nyquist are ε-free: `δ` is
only the operating-point offset `δ·|C''|=O(δ)`, **flat in `ε`** (here `6e-4`,
1000× smaller).

---

**Open cells** (FRAMEWORK_OUTLINE §Open questions): (i) Nyquist `unbiased on noisy
∇C` = ✓† (numerics `noisy_nyquist_vs_fd_kick`; proof pending, blocker #? band-limit
of the dephased cost in `s`); (ii) `coherent-error suppr.` kick ✓† / Nyquist †
(additive-slow-error check pending, blockers #3/#6). Verified elsewhere: FD
`δ^{2/3}` floor (measured `δ^{0.70}`), extensive-`θ` `diam(A)` scaling (table above).
