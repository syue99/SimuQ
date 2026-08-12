# Table 1 — The Regime Map (framework paper spine)

**Symbols.**
`θ` parameter · `∇C` target = gradient of the *noisy* device landscape ·
`T` evolution time · `Γ=1/T₂` dephasing rate ·
`δ` control setpoint-error (std) · `ε` FD step · `Δt` control time-slice ·
`N` shots · `ε_g` target gradient RMSE · `m` generator terms an extensive `θ` touches ·
`M` Nyquist truncation order (# shift pairs) ·
`v_j=∂u_j/∂θ` · `A=Σ_j v_j H_j` (tangent) · `diam(A)=λ_max−λ_min` ·
`K=(T/2π)·diam(A)` (Nyquist bandwidth).
Cells: **✓** holds · **✗** fails · **✓†** numerically holds, proof open · **?** open.
Nyquist has two variants — deterministic (order `M`) and **stochasticˢ** (samples
shifts); cells read `det / stochˢ` where they differ.

| Axis | Kick PSR | Nyquist | FD |
|---|:--:|:--:|:--:|
| generator req. | `H²=I` (e.g. Pauli) | `diam(A)<∞` (any bounded Hermitian) | none |
| primitive | native + digital gate | native op. | native op. |
| compilation | analog waveforms + gates (incl. transport) | waveforms only | waveforms only |
| executions / grad | `O(m)` | `O(M) / O(1)ˢ` | `O(1)` |
| bias | `0` | `O(K/M) / 0ˢ` | `O(δ^{2/3})` |
| variance | `O(T²)` | `O(K²)` | `O(1/ε²)` |
| extensive `θ` (`m` terms) | `O(m)` exec | `O(1)` dir, `K∝m` | `O(1)` |
| fine param, `Δt→0` | ✓ | ✗ `(s∝1/Δt)` | ✓ |
| unbiased on noisy `∇C` | ✓ | ✓† | ✗ |
| slow/coherent-error suppr. | ✓ | ? | ✗ |
| **shots `N→ε_g`** | `O(T²e^{2ΓT}/ε_g²)` | `O(K²e^{2ΓT}/ε_g²)` | `O(e^{2ΓT}/ε_g²)` |
| **δ-robust (no floor)** | ✓ | ✓ | ✗ `(ε_g≳δ^{2/3})` |

---

### Bandwidth estimate & the `T`-vs-`K` regime

For a Pauli-type tangent the terms give `diam(A)=2·Σ_j|v_j|`, so
`K = (T/π)·Σ_j|v_j|`. Single direction (`|v|≈1`): `K≈T/π≈0.32T`. Extensive `θ`
over `m` commuting terms: `K≈mT/π`. Since Nyquist variance is `O(K²)` and kick is
`O(T²)`, the ratio `K/T = Σ|v_j|/π` selects the strategy:

| regime | when | favored | why |
|---|---|:--:|---|
| `T ≫ K` | few terms / small tangent (`m ≲ π`) | **Nyquist** | `O(K²)` variance small; also works on non-Pauli |
| `K ≫ T` | extensive `θ` (`m ≳ π`) or large tangent | **Kick** | Nyquist `O(K²)∝m²` blows up — extensive `θ` "trips" it |

So yes: an extensive `θ` (`diam(A)∝m` ⇒ `K∝mT`) pushes into `K≫T`, where
Nyquist's one-direction execution saving is cancelled by `O(m²)` shot variance and
kick wins (when the generator is `H²=I`).

### `generator req.` — boson / fermion

Kick needs a **finite, few-level equidistant spectrum** so a finite shift rule is
exact; `H²=I` (Pauli strings, single Majoranas) is the canonical ±1 case.
- **Fermion**: number `n_i` (spec `{0,1}`) and hopping (`{-1,0,1}`) are equidistant
  with few levels → work via the *generalized* multi-term shift rule (not the bare
  2-term kick). ✓ with more terms.
- **Boson**: `n=a†a` (spec `0,1,2,…`, unbounded) → no finite shift rule → kick ✗.
  Nyquist needs `diam(A)<∞`, so it too needs a photon-number cutoff; with a cutoff
  it works (kick still does not, unless the cutoff spectrum is equidistant).

### `slow/coherent-error suppr.` — what it means

Coherent/slow errors = *systematic*, slowly-drifting or miscalibration errors
(control drift, gate over-rotation) — **not** random shot noise. "Suppression" =
the estimator's difference cancels such common-mode errors (an echo).
- **Kick**: `f⁻−f⁺` differ only by the ±kick sign on the *same* base evolution and
  the *same* kick gate → coherent errors are common-mode → cancel. ✓
- **FD**: `θ±ε` use *different* pulses → the errors are not common-mode → no
  structural cancellation. ✗
- **Nyquist**: `±s` are the same waveform family (only the shift sign flips), so
  they *plausibly* cancel like the kick — **unverified (?)**.

### `δ`-amplification when `ε ≈ δ`  (`figures/delta_amplification.png`)

FD differences two shifted setpoints, each with error `δ`, so the `δ` term is
`(δ/ε)·|C'|` — amplified by `1/ε`. As `ε→δ` it reaches `O(|∇C|)`: at a
steep near-inflection point (`∇C=1.07`, `δ=0.02`) FD error is `0.75 ≈ 0.7|∇C|`
at `ε=δ` (sign unreliable), and grows as `1/ε` (doubles per halving of `ε`).
Kick/Nyquist are ε-free: `δ` is only the operating-point offset `δ·|C''| = O(δ)`,
**flat in `ε`** (here `6e-4`, 1000× smaller). δ amplified in FD, kept `O(δ)` in PSR.

---

**Open cells** (blocking, FRAMEWORK_OUTLINE §Open questions): Nyquist `unbiased on
noisy ∇C` = ✓† (numerics `noisy_nyquist_vs_fd_kick`; proof pending) and
`slow-error suppr.` = ? (check pending). FD `δ^{2/3}` floor verified (measured
`δ^{0.70}`, `δ∈[0.005,0.15]`).
