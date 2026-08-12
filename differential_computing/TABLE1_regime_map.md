# Table 1 — The Regime Map (framework paper spine)

**Symbols.**
`θ` parameter · `∇C` target = gradient of the *noisy* device landscape ·
`T` evolution time · `Γ=1/T₂` dephasing rate ·
`δ` control setpoint-error (std) · `ε` FD step · `Δt` control time-slice ·
`N` shots · `ε_g` target gradient RMSE · `m` generator terms an extensive `θ` touches ·
`M` Nyquist truncation order (# shift pairs) · `v̄` typical `|v_j|` · `‖v‖₁=Σ_j|v_j|` ·
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
| variance (per shot) | `O(T²‖v‖₁²)·(1−f₊f₋)` | `O((2πK)²)=O(T²diam(A)²)ˢ` | `O(1/ε²)` |
| **shots @ extensive `θ`** | `O(T²‖v‖₁²·e^{2ΓT}/ε_g²)` | `O(T²diam(A)²·e^{2ΓT}/ε_g²)` | `m`-indep, **floored** |
| compilation | analog waveforms + gates (incl. transport) | waveforms only | waveforms only |
| **— guarantees —** | | | |
| bias | `0` | `O(K/M) / 0ˢ` | `O(ε²)+O(δ/ε)`, floor `O(δ^{2/3})` |
| unbiased on noisy `∇C` | ✓ | ✓† | ✗ |
| coherent-error suppr. | ✓† `O(η²)` | † | ✗ |

*Caption.* **Kick and Nyquist are the same L1 functional.** The stochastic-Nyquist
per-shot variance is the squared L1 norm of the weights `w_n=(2K/π)(−1)ⁿ/(n+½)²`:
`Σ|w_n|=(2K/π)·Σ_{n∈ℤ}(n+½)⁻²=(2K/π)π²=2πK`, so `Var≤(2πK)²=(T·diam(A))²`. Kick's
is `(T‖v‖₁)²(2−2f₊f₋)` measuring both ± branches; sampling the branch sign instead
(1 exec/sample) gives exactly `(2T‖v‖₁)²` — **identical to Nyquist**. Kick's only
structural discount is the both-branches factor `(1−f₊f₋)`: O(1), landscape-
dependent, `m`-independent. (Deterministic Nyquist adds an order-`M` budget-split
penalty over `Σw_n²`; the stochastic form above is the fair one.) FD cells are at
**free `ε`**; its floor lives in the bias row (`ε*∼δ^{1/3}⇒O(δ^{2/3})`).

---

### Kick vs Nyquist on shots — same L1 functional; NEITHER universally wins (F3)

Both cost `∼ (T·L1)²·e^{2ΓT}/ε_g²`; same L1 functional, **no `π²` advantage** (the
`π` in `K` and the `π²` in `Σ|w|=2πK` cancel). Two independent axes decide:

- **structural** `ρ = diam(A)/‖v‖₁ ∈ (0,2]` (Pauli, `diam(H_j)=2`). Nyquist pays
  `diam(A)`; a **folding** kick combines commuting/same-axis involutions
  (`cφX+sφZ` *is* an involution; `Σ(Z_j−Z_{j+1})` folds to `Z₀−Z_last`), so for any
  **foldable** tangent `ρ=2` (equal L1s). `ρ<2` only for **non-foldable
  subextensive-diameter** tangents (exotic, non-commuting frustrated structure).
- **landscape** `f₊f₋≈⟨O⟩²`. Kick's `±` branches are **co-located** (same `θ`, kick
  sign flips), so its paired single-shot 2nd moment is `2−2f₊f₋` — a **discount**
  when correlated/polarized (`f₊f₋>0`), a **penalty** when anti-correlated
  (`f₊f₋<0`). Nyquist's `±s` sit at far-apart points `θ±s` → no such term.

Ratio `kick/Nyquist = 4(1−f₊f₋)/ρ²`, so **kick wins iff `ρ > 2√(1−f₊f₋)`**:

| regime | condition | winner | why |
|---|:--:|:--:|---|
| aligned/foldable, correlated | `ρ=2, f₊f₋>0` | **kick** | co-located `±` → 2nd moment `2−2f₊f₋` small |
| aligned/foldable, anti-corr. | `ρ=2, f₊f₋<0` | **Nyquist** | kick's both-branches becomes a penalty |
| non-foldable subextensive | `ρ<2` | **Nyquist** | `diam(A)<‖v‖₁·diam(H)`; one combined shift |

Verified (`regime_kick_vs_nyquist.py`, `H=θZ+X`, `A=Z`, `ρ=2`): kick wins near
polarized `⟨Z⟩` (`f₊f₋>0`), loses near the equator — landscape-dependent, O(1). So
the generic (foldable) case is a **wash**; kick is favored at correlated/polarized
points (e.g. near optimization minima), Nyquist at high-entropy points or for the
exotic non-foldable subextensive tangents. Kick's *durable* wins are elsewhere:
coherent-error robustness, fine-grained `Δt`.

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
