# Section 6 Data — Answers to SEC6_DATA_RUN_GUIDE (2026-08-13)

Answering the guide question-by-question. The opening section fixes a scope
misunderstanding in the guide's framing; it changes what "compiled pipeline",
"emulator", "T4", and "compiled spot checks" mean for every downstream item.

---

## 0. METHODOLOGY / SCOPE (read first) — what actually produces these results

**We generate all Section-6 results at the SimuQ *Hamiltonian* level with a QuTiP
noise-model emulation. We do NOT run the physical `pulse → Hamiltonian → QuTiP`
reconstruction.** That physical path (AAIS `rydberg2d` compile → PulseDSL/AWG
waveforms → ledger reconstruction back to a Hamiltonian) exists in the repo but is
**not the instrument for Section 6**. Concretely, the pipeline we use is:

```
H(θ,t) = H_c + Σ_j u_j(θ,t) H_j      (SimuQ Parametrized_Hamiltonian, ideal operators)
   │  differentiate at the Hamiltonian level
   │    kick-PSR  → observable_program_generator  (τ-split + ±kick H_j)         [PSR]
   │    Nyquist   → nyquist_shift                 (shift waveform by s·∂u/∂θ)   [NSR]
   │    FD        → central difference of ⟨O⟩(θ)
   ▼  emulate the dynamics + noise
NoisyQuTiPRunner  =  QuTiP mesolve of each segment's TIHamiltonian with Lindblad
   collapse operators from noise_model.py  (T2* dephasing, gate/kick error, and a
   control setpoint error δ applied as a parameter offset).  ⟨O⟩ = Tr(Oρ).
```

So, mapping the guide's language to what we run:

| Guide term | What it is here |
|---|---|
| "compiled pipeline + emulator of Sec 5.5" | **SimuQ Hamiltonian program + `NoisyQuTiPRunner` (QuTiP mesolve)**. No pulse layer. |
| "calibrated channel model (T4)" | **`noise_model.py`** applied at the *segment/Hamiltonian* level: T2* dephasing on evolution segments, a discrete gate-error channel on kick segments, and δ as a control-setpoint offset. It is a *literature-calibrated* model, not a device calibration file (see P0-B). |
| "compiled spot checks" (P0-A.5, P1-C iv) | **Not available at this stage** — they require the pulse-level compile, and (importantly) `rydberg2d` only realizes ZZ + single-qubit terms, so the all-to-all XX/YY pool used in F3 is not even AAIS-representable. These items are deferred / reframed below, not run. |

**Consequence for the guide's standing rule** "NOTHING may be computed at
Hamiltonian level unless a figure explicitly says so": under the corrected scope,
**everything in Sec 6 is Hamiltonian-level**. Each figure caption should say so and
name the noise model. The honest statement is: *Sec 6 evaluates the differentiation
semantics + estimator behaviour under a calibrated Hamiltonian-level noise model;
pulse-level compiled validation is future work (the ledger round-trip of Sec 4/Tab 3
is the only place the pulse layer is exercised).*

---

## P0-A — answers before burning seeds

### A.1  ρ definition and its relation to the paper's χ

Exact formula used in `phase_who_wins_3panel` / `phase_shots_kick_vs_nyquist`:

```
ρ = diam(A) / Σ_j |v_j|,   A = Σ_j v_j H_j (tangent),  v_j = ∂u_j/∂θ,  diam = λ_max − λ_min.
```

The paper's compressibility `χ = Ω / D1` with `Ω = diam(A)` and
`D1 = Σ_j |v_j|·diam(H_j)`. For a **Pauli alphabet** `diam(H_j) = 2` for every term,
so `D1 = 2·Σ_j|v_j|` and therefore

```
χ = diam(A) / (2 Σ|v_j|) = ρ / 2   ⇒   ρ = 2χ   (exact for Pauli generators).
```

**Confirmed** on three test tangents (6 qubits), ratio ρ/χ = 2.000 in every case:

| tangent | diam(A) | Σ\|v\| | D1=2Σ\|v\| | ρ=diam/Σ\|v\| | χ=diam/D1 | ρ/χ |
|---|--:|--:|--:|--:|--:|--:|
| uniform ZZ chain | 10.00 | 5.0 | 10.0 | 2.000 | 1.000 | 2.000 |
| sign-alternating ZZ chain | 10.00 | 5.0 | 10.0 | 2.000 | 1.000 | 2.000 |
| Heisenberg chain (XX+YY+ZZ) | 14.97 | 15.0 | 30.0 | 0.998 | 0.499 | 2.000 |
| Heisenberg single bond | 4.00 | 3.0 | 6.0 | 1.333 | 0.667 | 2.000 |

**Recommendation:** relabel F3 to the single symbol **χ = ρ/2 ∈ (0,1]** (colorbar,
annotations, boundary). The kick/PSR-wins boundary `ρ > 2√var` becomes
**`χ > √var`**; the aligned/foldable extreme `ρ=2` becomes `χ=1`.

**Anomaly worth flagging (a finding, not a bug):** *sign-alternating ZZ has the SAME
diameter as uniform ZZ (χ=1)* — sign flips on **commuting** terms do not reduce the
spectral diameter (each ZZ can independently reach ±1). Subextensivity (χ<1) requires
**non-commuting** cancellation (Heisenberg χ≈0.5). If the paper's intuition was
"sign-alternating → compressible", that is false for commuting alphabets; state the
condition as non-commutativity, not sign structure.

*(computation: `sec6_rho_chi.py` → `figures/sec6_rho_chi.json`.)*

### A.2  The `⟨σ⟩=1.37` annotation

It is the **mean per-generator branch shot-standard-deviation**
`σ_j = √((1−f₊,j²)+(1−f₋,j²))`, averaged over the pool and the sampled states — i.e.
the *kick/PSR co-located ± branch shot noise* at the operating point. It **does**
affect the who-wins boundary: the PSR-vs-NSR ratio is `∝ σ²/χ²`-ish, so `σ` sets how
far the boundary sits (polarized states → small σ → PSR region grows; high-entropy
→ σ→√2 → PSR region shrinks). **It must stay in the caption**, stated as "high-entropy
operating point, ⟨σ⟩≈1.37 — the PSR-conservative regime; polarized states enlarge the
PSR region." It is *not* a noise/normalization literal to drop.

### A.3  Per-iteration budget + optimizer (F-loop)

**Agree with the proposal** — vanilla GD, fixed learning rate, `B = 1000` quantum
executions **per gradient** (split per component by each method's own accounting:
FD = 2 execs/component, PSR = its branch draws, NSR = singleton draws), 60 iterations,
20 seeds; same optimizer/schedule/`B` for all methods; only FD's ε grid varies.

Two things I will fix before running (not deviations, just under-specified):
- **Learning rate**: pick ONE `η` from the *noiseless* gradient's descent on the
  program (a value that converges cleanly noiselessly), then freeze it for all
  methods and seeds. Document `η`.
- **`θ*` and the objective**: `θ* = argmin` of the *emulated noisy* landscape `C_noisy`
  (grid + polish), and the plotted quantity is `C_noisy(θ_t) − C_noisy(θ*)`. This is the
  device objective (no "rescale"/oracle framing), per the guide.

### A.4  Which θ are differentiated (and single-qubit insertions)

Benchmark programs are Hamiltonian-level; the running example is **TFIM**
`H(θ) = J·Σ_i Z_iZ_{i+1} + g·Σ_i X_i`, differentiated w.r.t. **J (coupling, generator
ZᵢZᵢ₊₁ — two-qubit)** and **g (transverse field, generator Xᵢ — single-qubit)**. The
foldable-panel program (P1-C) adds a Heisenberg/XY tangent.

- At the **Hamiltonian level**, differentiating `g` *does* use a single-qubit kick
  generator `Xᵢ`. So the semantics exercise single-qubit insertions.
- At the **pulse-compiled level** — which we are *not* running — the single-qubit
  "insertion into the gate/1q zone" realization is therefore **not exercised** by any
  Section-6 run. **Honesty note for Table `tab:realization` first two rows:** those
  rows describe the compiled realization of 1q/2q insertions; Section 6 validates them
  only at the Hamiltonian level, not through the pulse layer. State this explicitly.

### A.5  F3 compiled-pipeline spot-check feasibility

**Not feasible as written, for two independent reasons — this is a finding:**
1. **We are not running the pulse compile for Sec 6** (Section 0). There is no
   compiled instrument to spot-check against.
2. Even if we did, the F3 pool is **all-to-all two-local {XX,YY,ZZ}**, and the
   `rydberg2d` AAIS realizes **ZZ + single-qubit** interactions only — XX/YY two-local
   tangents are **not AAIS-representable**, so the general and Heisenberg panels
   cannot be compiled at all. Only the ZZ-only (favor-PSR) panel is even in scope for a
   future compiled anchor.
Additionally, **NSR has no compiled lowering** yet (the compiler was built for the
kick; Nyquist waveform shifts are not in the AAIS/tweezer_mapper path).

**Proposal:** drop the "compiled spot-check overlay" from F3; caption states F3 is
Hamiltonian-level under the T4 noise model. If a compiled anchor is wanted later, the
only tractable one is a **ZZ-only, PSR-only** point (e.g. TFIM-J on a chain, small n),
and it validates the *ledger round-trip* (Tab 3), not the who-wins verdict. I can set
that up as a separate item — flag if you want it.

---

## P0-B — T4 noise model table

**Reframe:** T4 is our **literature-calibrated Hamiltonian-level model**
(`noise_model.py`), not a device calibration file. I can produce `T4.csv` + a rendered
table listing every channel, its rate, its (literature) source, and which schedule
segments it applies to (the three prices of Sec 5.2: dressed T2* on interaction
segments; long-clock decoherence on 1q/halt/transport; gate infidelity on kick
insertions), with the **θ-independence assumption flagged**. **Uncertainty to resolve
with you:** the *provenance strings* (which calibration each rate comes from) — the
code has the rates (T2*, 1q/2q gate error, leakage) but not documented citations. I
will fill sources from the values already in `noise_model.py` + the Evered et al.
budget referenced in the outline, and mark any I cannot source as `TBD-citation`.
**Held pending your confirmation of the numbers** before I emit T4.csv as the single
source of truth.

---

## P1 / P2 — status (all Hamiltonian-level under T4; ✅ = delivered)

Each delivered item has script + PNG/PDF + JSON + a data note; 20 seeds median+IQR;
execution-normalized; T/T2* in caption; T4 best-guess values flagged.

- **P0-B (T4)** ✅ — `sec6_T4_noise_table.py` → `T4.csv` + `sec6_T4_noise_table.png`.
  All channels θ-independent (flagged). Values best-guess pending calibration.
- **P1-A (F6, floor + amplification)** ✅ — `build_F6.py` → `F6_floor_amplification.*`.
  TFIM θ·Z0Z1+ΣX, T/T2*=0.15. Panel L: PSR (symmetric kick) & NSR (stochastic) ride
  `N^{-1/2}` to `∇C_noisy` (exact fine-FD target, logged), crossing below FD's δ/ε floor
  (FD frozen at ε*=0.25 tuned once at N=1e4). Panel R: FD V-shape, PSR/NSR flat.
  **Two findings surfaced (in the data note):** (i) the PSR estimator was
  bootstrap-resampling the τ-pool → spurious variance floor (fixed: use all pool
  samples). (ii) **T4's kick gate error biases raw PSR by ~0.028** (the kick is a
  digital op with its own error; NSR is immune) — a real *pro-NSR* result, but a
  Sec-5.2 gate-infidelity point, so it is **excluded from F6** (F6 = dephasing + δ).
- **P1-B (F-loop)** ⏳ running — `build_Floop.py`, TFIM P=4. Modeling documented in the
  data note (FD = real noisy secant + δ + shots; PSR/NSR = unbiased ∇C_noisy + shot
  noise σ=2T√(P/B), the L1-functional model — PSR≈NSR for single-Pauli ZZ). θ* via
  Nelder-Mead. Added `nsteps` to `NoisyQuTiPRunner` (backward-compat) for stiff mesolves.
- **P1-C (F3)** ✅ — `phase_who_wins_3panel.py` relabeled (PSR/NSR, χ=ρ/2, Hamiltonian-
  level under T4, ⟨σ⟩ kept, no compiled overlay).
- **P2-A (Fig 1)** ✅ — `build_fig1.py` → `fig1_intro_trap.*`, single-column, T/T2*=0.5.
- **P2-B (systems 6.5)** — **blocked at Hamiltonian level**: compile-scaling and the
  NSR-vs-PSR compile asymmetry require the compiler, and **NSR has no compiled lowering**
  (A.5). PSR C7 exists; NSR compile numbers unavailable until lowering is built. Not run.

### Net uncertainties to discuss
1. **T4 provenance** — confirm the noise rates + their sources so T4.csv can be the
   single source of truth (P0-B).
2. **δ value** — is 0.02 the intended control-noise magnitude (P1-A/P0-B)?
3. **Compiled anchors / NSR compile numbers** — out of scope at Hamiltonian level;
   confirm we drop them from Sec 6 (A.5, P2-B) or schedule NSR lowering as separate work.
4. **η, θ*** — I will fix these from the noiseless program before the F-loop run (A.3).
