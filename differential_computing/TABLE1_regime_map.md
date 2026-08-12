# Table 1 — The Regime Map (DiffSimuQ framework paper spine)

Three sound differentiation strategies for `H(t)=H_c+Σ_j u_j(t;θ)H_j`, characterized
by axis. **Columns**: Kick PSR (arXiv:2210.15812) · Nyquist waveform shift
(arXiv:2207.01587) · FD (finite-difference baseline). Two cells are genuinely
**OPEN** (blocking questions in FRAMEWORK_OUTLINE.md §"Open questions").

Notation: `K = (T/2π)·diam(A)`, `A = Σ_j v_j H_j`, `v_j = ∂u_j/∂θ` (the tangent);
`diam = λ_max−λ_min`. `δ` = control-resolution setpoint error (std). `Γ = 1/T2`
(dephasing). `ε_g` = target gradient RMSE. `m` = number of Pauli terms an
extensive parameter touches. Branch/execution count is de-emphasized: with the
clean compiled structure the compiler cost is negligible — **shots are the
currency**.

---

## The map

| Axis | **Kick PSR** | **Nyquist** | **FD (baseline)** |
|---|---|---|---|
| **generator requirement** | Pauli-like (involution / few-eigenvalue) **and** separately synthesizable as a gate | derivative direction lies in the tunable control span — any Hermitian `H_j`; no Pauli / involution / separate-synth | none (black-box cost) |
| **hardware requirement** | one digital op (native CZ/1q gate) + gate zone & transport | amplitude headroom for the shifts `\|s\|≤(N+½)/2K` on existing channels | none (re-run at a shifted parameter) |
| **executions / gradient dir.** *(minor — compiler ~free)* | 2 per term × τ-samples (2 per term with deterministic-τ quadrature) | 2N (deterministic truncation) **or** 1 per stochastic sample | 2 per parameter |
| **bias** | none (unbiased) | deterministic: truncation, **algebraic ∼K/N**; stochastic: **none** (unbiased) | `O(ε²)` truncation **+** `δ/ε` control → irreducible floor |
| **variance (noise scale)** | `O(1)` / branch → per-grad `Var=4T²/N` (**∝ T**) | `Var ∝ K²/N` (**noise scale ∝ K**); grows with extensive-sum `diam(A)` | `Var ∝ 1/(ε²N)` (**∝ 1/ε**); blows up as `ε→0` |
| **extensive parameter** (touches `m` terms) | `m` branch-pairs; shots split → `Var ∝ m²` | ONE tangent direction, but `K ∝ diam(A) ∝ m` → `Var ∝ m²` | 2 execs, but curvature/attenuation degrade the floor |
| **fine-grained param, Δt→0** | fine (kick angle ⟂ Δt) | **shifts diverge** `s_0=1/4K ∝ 1/Δt` → amplitude blows up; unsuited | fine (ε ⟂ Δt) |
| **noise: unbiased for the *noisy* landscape** | **yes** (proven — Lindblad signed-kick shift identity, holds for mixed ρ) | **OPEN (proof).** Numerically **yes** — converges to `∇C_noisy` past the FD floor (`noisy_nyquist_vs_fd_kick`) | **no** (δ/ε floor; targets the attenuated landscape with bias) |
| **noise: slow / coherent-error suppression** | branch-symmetric (±kick same family → common drift cancels in `f⁻−f⁺`) | **OPEN, check.** ±s are the same waveform family — plausibly cancels; unverified | none |
| **compilation cost** | zones + transport + gate synthesis (clean, ~free) | **waveform synthesis only** (shifted amplitudes on existing channels; no zone/gate) | waveform synthesis only |

---

## Shot complexity — the error-robustness spine (big-O)

Shots `N` to reach gradient RMSE `ε_g`, and its scaling in the **control error δ**
and **dephasing Γ**. This is the axis that decides deployment.

| | **Kick PSR** | **Nyquist** | **FD (baseline)** |
|---|---|---|---|
| **base (δ=0, Γ small)** | `O(T² / ε_g²)` | `O(K² / ε_g²)` *(stochastic; det.: `O(K³/ε_g³)` with truncation, or floored at `K/N` for fixed N)* | `O(1 / ε_g²)` |
| **control error δ** | `O(T²/ε_g²)` — **no δ-floor** (δ is 2nd-order operating-point only) | `O(K²/ε_g²)` — **no δ-floor** (ε-free; smallest shift `1/4K = O(1)`, so `δ/s = O(δ)`) | **floored: `ε_g ≳ O(δ^{2/3})`** — *no shot count reaches below* (`ε*∼δ^{1/3}` balances `ε²`-truncation vs `δ/ε`) |
| **dephasing Γ=1/T2** | `× O(e^{2ΓT})` (signal attenuation), **still unbiased** | `× O(e^{2ΓT})`, **still unbiased** | `× O(e^{2ΓT})` **and** the δ-floor persists (with Γ-damped curvature) |

**Reading it.**
- **δ separates the classes.** Kick and Nyquist are ε-free, so the control error
  is only a second-order operating-point shift — `O(1/ε_g²)` shots reach *any*
  target. FD needs two shifted setpoints, so δ enters as `δ/ε` and imposes an
  irreducible floor `ε_g ≳ δ^{2/3}`: **no number of shots crosses it.** Both
  confirmed empirically: (i) at high shots the oracle-FD floor scales `∝ δ^{0.70}`
  (measured, vs predicted `2/3`; δ swept 0.005→0.15); (ii) `noisy_nyquist_vs_fd_kick.png`
  — oracle-FD flattens at ~0.05 while kick and Nyquist ride `N^{-1/2}` straight
  through it (`T/T2*=0.5, r=0.02`).
- **Γ costs shots but not bias.** Dephasing attenuates the target gradient
  `g_Γ ∼ e^{-ΓT} g_0`, so *relative*-error shot count grows `∝ e^{2ΓT}` for **all
  three** — it is signal attenuation, not a method artifact. The differentiator
  is that kick/Nyquist stay **unbiased** under it (they estimate the true noisy
  gradient), while FD keeps its δ-floor on top.
- **Kick vs Nyquist is `T` vs `K`.** Same `1/ε_g²` shot law; the coefficient is
  `T²` (kick) vs `K² = (T·diam A/2π)²` (Nyquist). For a single-generator direction
  `K ∼ T/π`, comparable; for an **extensive** parameter `diam(A) ∝ m` so Nyquist's
  `K² ∝ m²` — it pays in variance exactly what it saves in executions. **Kick wins
  variance; Nyquist wins generality/capability (non-Pauli) and compile cost.**

---

## The two OPEN cells (blocking, FRAMEWORK_OUTLINE §Open questions)

1. **Nyquist unbiasedness for the noisy landscape** — kick is proven (superoperator
   Duhamel + signed-kick commutator insertion); the Nyquist analogue (is the
   *dephased* cost still band-limited in the shift `s` with the same `K`, so the
   truncated/stochastic sum stays unbiased for `∇C_noisy`?) needs its own
   derivation. **Numerical evidence says yes** (this work); a proof turns two
   theorems into one and strengthens §4.5.
2. **Nyquist slow/coherent-error suppression** — the `±s` executions are the same
   waveform family, so slow drift *should* cancel in `J[u+sv]−J[u−sv]` the way the
   ±kick cancels for kick PSR. Unverified — a `check` (numerical + argument) that
   fills the last row's middle cell.

---

*Supporting numerics: noiseless accuracy-vs-executions `nyquist_vs_fd_kick.{json,png}`;
noisy (δ + Γ + shots) `noisy_nyquist_vs_fd_kick.{json,png}`; infra `nyquist_shift.py`,
tests `test_nyquist_shift.py`.*
