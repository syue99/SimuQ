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

**Confirmed** on the test tangents (6 qubits). For **single-Pauli** generators `ρ/χ =
2.000` exactly; the general `D1 = Σ_j|v_j|·diam(H_j)` uses each generator's own diameter
(the telescoping row below has diam(H_j)=4, so `ρ≠2χ` there):

| tangent | diam(A) | Σ\|v\| | D1 | ρ=diam/Σ\|v\| | χ=diam/D1 | ρ/χ |
|---|--:|--:|--:|--:|--:|--:|
| uniform ZZ chain | 10.00 | 5.0 | 10.0 | 2.000 | 1.000 | 2.000 |
| sign-alternating ZZ chain | 10.00 | 5.0 | 10.0 | 2.000 | 1.000 | 2.000 |
| Heisenberg chain (XX+YY+ZZ) | 14.97 | 15.0 | 30.0 | 0.998 | 0.499 | 2.000 |
| Heisenberg single bond | 4.00 | 3.0 | 6.0 | 1.333 | 0.667 | 2.000 |
| **telescoping Σ(Zⱼ−Zⱼ₊₁)** (commuting) | **4.00** | **5.0** | **20.0** | 0.800 | **0.200** | 4.000 |

**Recommendation:** relabel F3 to the single symbol **χ = ρ/2 ∈ (0,1]** for single-Pauli
tangents (colorbar, annotations, boundary). The kick/PSR-wins boundary `ρ > 2√var`
becomes **`χ > √var`**; the aligned/foldable extreme `ρ=2` becomes `χ=1`.

**The compression condition (CORRECTED per SEC6_FOLLOWUP C1 — my earlier
"non-commutativity" claim was wrong).** Compression (small χ) = **failure of joint
extremizability** of the weighted sum `Σ_j v_j H_j` (its joint value cannot reach
`Σ_j|v_j|·diam(H_j)`). Reachable by EITHER mechanism, independently:
- **(i) shared-support cancellation within a COMMUTING family.** The telescoping tangent
  `Σ_j (Z_j − Z_{j+1})` is fully commuting yet `χ = O(1/m)` (row 5: it telescopes to
  `Z_0−Z_m`, diam(A)=4, but D1 = Σ_j diam(Z_j−Z_{j+1}) = 4m → χ = 1/m; measured 0.200 at
  m=5). Commuting families **can** compress.
- **(ii) anticommuting / non-commuting contraction** — Heisenberg bonds (χ≈0.5); `X_a`
  with `Z_aZ_b`.

Sign flips alone do **not** compress: sign-alternating ZZ on a chain is still jointly
extremizable (each bond value is independently ±1) ⇒ χ=1, identical to uniform ZZ. **Do
not** write "requires non-commutativity" in any caption; state it as joint-extremizability
failure. (The paper's companion table already encodes this; do not edit it.)

*(computation: `sec6_rho_chi.py` → `figures/sec6_rho_chi.json`; telescoping row added.)*

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

Two things I fixed before running (not deviations, just under-specified):
- **Learning rate**: one `η`=0.25 frozen for all methods and seeds (converges cleanly on
  the noiseless descent). All seeds share the start `θ0 + N(0,0.08)` jitter.
- **`θ*` and the objective**: `θ* = argmin` of the *emulated noisy* cost, plotted quantity
  `C_noisy(θ_t) − C_noisy(θ*)` — the device objective, no rescale/oracle framing.

**Two modeling points forced by making the loop honest (per SEC6_FOLLOWUP C2):**
- **Cost observable & shot model.** `O = (1/P)Σ_i Z_iZ_{i+1} ∈ [-1,1]` (mean bond parity).
  All bonds are DIAGONAL, so one Z-basis shot draws a bitstring giving every bond at once:
  the shot model samples basis states from `diag ρ` and averages — the correct finite-shot
  model for a summed diagonal cost. (A naive single-`[-1,1]` binomial on `⟨ΣZZ⟩∈[-P,P]`
  saturates the clip and yields identically-zero gradients — a bug I caught and fixed.)
- **Interior minimum via an amplitude regularizer.** The raw `⟨O⟩` minimum sits at the
  coupling box edge (couplings want to grow), where NSR's Nyquist shifts exceed the
  amplitude limit. I descend `C = ⟨O⟩ + (λ/2)|θ|²`, `λ=0.3` — a physical amplitude prior
  giving an interior `θ*` (‖θ*‖∞≈1.43). `∇reg = λθ` is an EXACT classical add-on to the
  SAMPLED `∇⟨O⟩` (consumes no shots, identical for every method), so the estimator
  semantics stay pristine. NSR's `n≥1` tail shifts still clip near `θ*` — that is NSR's
  amplitude-headroom (certificate) cost, and it is on-narrative.

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

- **P0-B (T4)** ⏸ HELD (Q1) — `sec6_T4_noise_table.py` → `T4.csv` + `.png` exist as
  best-guess/provisional, but per FOLLOWUP Q1 **T4.csv is not final** until Fred confirms
  the rates + δ=0.02 + provenance strings. All channels θ-independent (flagged).
- **P1-A (F6, floor + amplification)** ✅ **F6_REVISION applied** — `build_F6.py` →
  `F6_floor_amplification.*` + `_caption.txt`. TFIM θ·Z0Z1+ΣX, **both panels T/T2*=0.15**
  (right panel is the 0.15 rebuild, A3). Error vs **∇C_noisy** (noisy gradient, stated on
  both axes, A1). x = **total executions for one gradient estimate** with per-method
  accounting in the note (A2). No "raw"/"oracle" (A4/B1). **Panel L:** PSR & NSR ride
  `N^{-1/2}` — fitted **N^−0.49 / N^−0.48** (B4) — to ∇C_noisy; FD frozen at ε*=0.25
  saturates at the **predicted δ/ε floor 0.025** (B5). **B2 disclosure:** faint
  *PSR + gate channel* series floors at PSR's own **≈0.028** kick-gate bias (NSR immune) —
  the gate channel is excluded from the headline but shown, forwarded to Sec 6.3 in the
  caption only (D5). **Panel R:** FD V, both arms, over the predicted δ/ε floor curve (C1);
  sign-flip markers, peak **25%** wrong-sign (C2); PSR/NSR flat = "no step size"; ε=δ and the
  **usable-ε window [0.058,0.703]** marked (C3, same δ/definition as Fig 1, wider here because
  θ0 is smooth: C''≈0.03). Real estimators, no surrogates (D2); 20 seeds median+IQR (D3);
  Hamiltonian-level under T4 (D4); δ/rates provisional/Q1-pending (D1).
- **C3 (gate-bias cell)** ✅ NEW — `build_gate_bias.py` → `gate_bias.*`. Raw-PSR bias vs
  ∇C_noisy at 0.5×/1×/2× the T4 2q rate: **−0.020 / −0.028 / −0.039**, scaling ~√ε_gate
  (coherent-dominated); NSR ≡ 0 (no inserted op). FINDING: standard and short (symmetric)
  kicks give an IDENTICAL bias — the T4 gate error is a fixed post-kick Z-channel, not
  echoed by kick symmetry; the digital price is intrinsic to inserting the op. This is
  Sec-6.3's complementary-failure-modes entry (PSR pays the digital price; NSR pays the
  certificate scale). Error bars = shot std at N=1e4.
- **P1-B (F-loop, REAL estimators)** ✅ — `build_Floop_real.py` → `F_loop_real.*` (C2
  compliant; surrogate `build_Floop.py` retired). TFIM P=4 (5q), cost `⟨(1/P)ΣZZ⟩+λ/2|θ|²`
  (λ=0.3, interior θ*), diagonal-readout shot model. **PSR** = real kick branches through
  the noisy runner INCLUDING the T4 kick gate error → carries its ~0.028 digital bias in
  the loop; **NSR** = real stochastic Nyquist sampler (gate-immune; tail-shift clip near
  θ* = its headroom cost); **FD** = real noisy secant + δ + shots (only the FD ε-grid uses
  the permitted side-variant form). B=1000/grad, η=0.25, 60 iters, 20 seeds, θ0+N(0,0.08)
  shared start. Measured cost ≈2.1 s/PSR-grad, 1.7 s/NSR-grad → full run ~2 h wall (20×60
  at P=4; feasible, no reduction needed — reported per C2).
  **Result (final median C−C*):** NSR **0.0027** < PSR **0.0037** < FD 1× **0.0084** <
  FD 3× **0.0337** < FD 0.3× **0.0795**. Both unbiased estimators reach the shot floor,
  ~2–20× below every FD ε; no FD step saves it (0.3× too noisy — 519 uphill steps across
  the descent — 3× biased, 1× still floored above). PSR sits marginally above NSR = its
  ~0.028 digital gate bias carried *inside the loop* (within the shot band at B=1000; the
  C3 cell isolates it cleanly at fixed θ). Unbiased-vs-FD separation is clear by ~35% of
  the execution budget. θ*=[-0.057,-1.429,1.429,0.057], C*=0.311.
- **P1-C (F3)** ✅ — `phase_who_wins_3panel.py` relabeled (PSR/NSR, χ=ρ/2, Hamiltonian-
  level under T4, ⟨σ⟩ kept, no compiled overlay).
- **P2-A (Fig 1)** ✅ — `build_fig1.py` → `fig1_intro_trap.*` + `fig1_intro_trap_caption.txt`.
  Single-column, self-contained (own Hamiltonian-level landscape H=θZ0+X0, `⟨Z0⟩_noisy`;
  does not read the shared `landscape_device_data.json`, so Fig 3 is untouched). FIG1_REVISION
  **+ REV 2** applied. **The R2/R5.4 tension is RESOLVED by R8's grid sweep** (T×θ*×ε_min,
  T/T2*=0.5 fixed): the flip condition is ε>λ/2 (half the ripple period), independent of
  anchor, while secant magnitude peaks at the steep mid-flank — so a mid-flank anchor with
  ε_min bumped above λ/2 flips all three *with margin*. Sweep found 41 passing configs;
  chosen **NON-MARGINAL (REV3 R9) T=12, θ*=1.290, ε=0.18/0.25/0.32**: secants +0.77/+1.44/+0.65
  all wrong-signed; anchor **25%** of period off the extremum (margin, not on-the-line);
  |slope| 4.35 = **61%** of max; no collisions. Tangent slope = analytic ∇C_noisy = −4.35.
  **REV 2 R7:** caption migrated in-figure — muted info line (program·instrument·regime),
  per-secant ε labels, **PSR/NSR vocabulary only**, **no section refs inside the image**;
  ~2-line mini caption (REV3 R12 text) delivered in the sidecar + JSON.
  **Small-ε δ-floor (answers "what if we shrink ε", REV3 R10):** shown + said, honestly.
  Shown — (i) a **noise cone** at ε=0.03≈δ (mean −4.01 vs true −4.35, σ 1.66, reaching
  near-flat/wrong-sign) = the δ/ε scatter near the floor; (ii) a **"step floor ε≳δ"** δ-wide
  bracket = ε cannot be set below the control resolution δ. Bounded (ε-window sweep [δ,λ/2]):
  best ε=0.056 (RMSE/|g|=0.28), usable window [0.026,0.072] ≈34% of [δ,λ/2]. **HONESTY:** FD is
  *not* trapped from below at this anchor — a usable ε exists; Fig 1 asserts only that the shift
  rules need **no ε** (precise, safe) and defers FD's quantitative defeat to Sec 6.2/F6-R. δ=0.02
  is Q1-pending — cone geometry depends on it (re-render if δ changes). Fan noise-model bug fixed
  (divide by NOMINAL 2ε, not the drifted separation).
- **P2-B / Track P (systems, R0)** — reframed by FOLLOWUP R0 as **Track P** (pulse
  generation to EVIDENCE the Sec-5 compiler claims, not a physics result): lower a
  Nyquist-shifted TFIM through the SAME AAIS→pulse→ledger path and report lines-of-glue +
  compiler components modified. Days-scale systems task; **not run autonomously** (flagged
  for scheduling). NSR still has no compiled lowering, which is exactly what Track P builds.

---

## SEC6_FOLLOWUP (2026-08-13) — resolutions

- **R0 two-track** — Track S = all Sec-6 physics at the Hamiltonian level (as run); Track P
  = pulse-gen artifact to evidence Sec-5 compiler claims (lines-of-glue for NSR lowering).
  Claims kept separate; every caption says "Hamiltonian-level under T4." Track P scheduled,
  not run.
- **C1 compressibility (CORRECTED)** — my "requires non-commutativity" was wrong. Condition
  = failure of joint extremizability; commuting families CAN compress (telescoping
  `Σ(Z_j−Z_{j+1})` → χ=0.200=1/m, added as row 5 of `sec6_rho_chi`). A.1 rewritten; no
  caption will say "non-commutativity."
- **C2 F-loop surrogate (FIXED)** — replaced by `build_Floop_real.py` with the REAL sampled
  estimators (PSR incl. gate error, NSR stochastic sampler, FD real secant). Cost measured
  and feasible (~78 min); no reduction needed. Surrogate retired.
- **C3 gate-bias placement (DONE)** — measured at 0.5×/1×/2× rate (`build_gate_bias.py`),
  moved to the Sec-6.3 complementary-failure-modes prose + data note.
- **R1 accepted** — χ relabel + boundary `χ>√var` + extremes `χ=1`; ⟨σ⟩=1.37 stays in F3;
  A.4 realization-row honesty note; F3 compiled overlay dropped.

### Q-items — HELD for Fred (why each is held)

These are **held because the FOLLOWUP explicitly filed them as Q-items** = "still need
confirmation from Fred before you act." They are not blocked on capability (I can execute
all three); they are blocked on a **decision that is the author's to make**, and acting
first would either bake in an unconfirmed number or spend a figure slot Fred may want
differently. Concretely:

- **Q1 — T4 rates + δ=0.02 + provenance.** Held because T4.csv is meant to be the paper's
  **single source of truth** for the noise model, and I only have *best-guess* rates +
  *undocumented* provenance (which calibration each rate traces to). FOLLOWUP C says "emit
  T4.csv only after confirmation." Emitting it now would give a paper artifact a false air
  of authority. *What unblocks it:* Fred confirms the numbers (T2*, 1q/2q gate error,
  leakage, δ) and the citation for each. The provisional CSV/PNG exist and are flagged; I
  flip them to final on his word — minutes of work.
- **Q2 — F3 device-alphabet foldable panel** (X_a + Z_aZ_b anticommute → χ→1/√2 *inside*
  the device alphabet). Held because it is a **figure-composition decision**: FOLLOWUP Q2
  lists three mutually-exclusive options (add as a 4th panel / replace panel (c) / leave F3
  as is). The *physics* is settled and cheap to compute; *where it goes* changes F3's
  layout and narrative, which is the author's call. *What unblocks it:* Fred picks one of
  the three; I build it same-day (it reuses `sec6_rho_chi`/`phase_who_wins_3panel`).
- **Q3 — F-loop `T/T2*=0.5` stressor variant.** Held because FOLLOWUP Q3 says "decide after
  the 0.15 run lands." **That precondition is now met** — the 0.15 run landed
  (`F_loop_real.*`). So Q3 is **now actionable**; it is held only on Fred's go/appendix
  decision, not on any dependency. *Cost if greenlit:* one more ~2 h run at the same P/seeds
  with `T2 = T/0.5` (a one-line change), producing an appendix stressor figure.
