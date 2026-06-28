# DiffSimuQ — Project Summary

_Last updated: 2026-06 (current session)_

DiffSimuQ is a full-stack **automatic-differentiation system for analog quantum
programs** — parameter-shift-rule (PSR) gradients for pulse/analog Hamiltonian
evolution, compiled to neutral-atom tweezer arrays, with a faithful noise model
and a benchmark comparing PSR against finite difference (FD). It implements and
extends **Algorithm 1 of Leng et al. 2022** (arXiv:2210.15812, "Differentiable
Analog Quantum Computing for Optimization and Control"), and is the gradient
engine for **Inverse Quantum Simulation** (IQS, arXiv:2601.12239).

Paper framing: _"Quantum Engineers Meet Automatic Differentiation."_

---

## 1. Architecture (compilation stack)

| Layer | Component | Status |
|---|---|---|
| **1 — PSR branch generator** | `observable_program_generator.py`, `combine_gradient.py` | ✅ validated |
| **2 — QuTiP simulation** | `qutip_sequential.py`, `noisy_qutip.py` | ✅ PSR vs FD < 1e-3 |
| **3 — Tweezer compilation** | `tweezer_mapper.py` (zone arch, solver improvements, kick compilation) | ✅ seg-norm < 0.02 |
| **4 — PulseLedger + verify** | `pulse_ledger.py`, `verify_compilation.py` | ✅ multi-layer 0.06% |
| **5 — DSL / AWG bridge** | op-tree IR → PulseDSL → AWG | ✅ structure; shapes deferred |

**Gradient formula (Algorithm 1):**
`∂L/∂v = (T/b) Σ_k Σ_j (∂u_j/∂v)(τ_k) · (p̃⁻_j − p̃⁺_j)`

**DSL op-tree bridge (2026-06)** — the PulseDSL scheduler is now *declarative*
(`SEQ`/`PARA`/`COMB`/`RUN`); the old imperative bridge emitted nothing. New path:
- `pulse_tree.py` — DSL-agnostic IR (`Seq`/`Para`/`PlayNode`/`CombNode`/`AodNode`/
  `DelayNode`); `flatten()` reproduces the old flat op list exactly.
- `tweezer_mapper.map_hlist_tree` — emits the tree **natively** with the
  **position-segmented PARA** rule (AOD = hard barrier; same-position plays
  concurrent; conflicting positions serialize). Decoupled from verify.
- `physical_channels.to_physical` — consolidates per-qubit logical channels onto
  **5 fixed physical channels** (TRANSPORT_AOD / ADDR_DET / ADDR_RABI /
  DRESSING_AOM / GATE_AOM); per-atom control = RF **tones** (`COMB`), not wires.
- `COMB` instruction added to the PulseDSL repo (multi-tone on one device);
  `Pulse.frequency` field added.

**TD Hamiltonian support (2026-04)** — `td_hamiltonian.py`, `td_psr.py`,
`compile_td`/`run_td`; simulator path validated (<2% vs FD), hardware path scoped.

---

## 2. Noise infrastructure (the PSR-vs-FD benchmark engine)

Built in QuTiP (the right tool: small-n, fully-non-Clifford analog evolution has
no structure for Stim / stabilizer-rank simulators).

- `noise_model.py` — `NoiseModel(T1, T2, pauli_rates, leakage_rate,
  gate_error_1q/2q, gate_coherent_frac)`. All decoherence is Lindblad collapse
  ops integrated by `mesolve` over each segment's **real duration** →
  duration-scaled, fair between FD (1 segment) and PSR (3 segments).
- `noisy_qutip.py` — `NoisyQuTiPRunner(noise, kick_dephases=False)`. Density-matrix
  evolution; post-selected leakage via conditional no-jump Liouvillian; gate error
  on the kick segment.
- `shot_sampling.py` — ±1 binomial shot noise.

**Physical noise model (neutral-atom platform, user-pinned):**
- **Avalanche loss** — cryo-suppressed (BBR ↓30×) + post-selected out (whole-array
  loss is detectable).
- **Single-atom dressing leakage** (only |1⟩ dressed) — `Γ ≈ (Ω/2Δ)²/τ_Ryd ≈
  2.5e-5/µs` → ~10⁻⁵/shot → **negligible**, confirmed.
- **T2\*** — ground clock ~ms (negligible over µs evolution); ground–Rydberg ~20µs
  but only via admixture and **strobing-suppressed** → benign.
- **Gate error** — kick gate, Z-type (Doppler/T2\*-dominated, NOT flip), anchored
  to **Evered et al. 2026** (arXiv:2604.25987): 99.9% 2q / 99.99% 1q.

**★ Key modeling correction (this session):** the PSR **kick is a gate**
(clock-state rotation / Rydberg gate), **not** a dressed evolution, so the
dressing-T2\* does **not** act during it. `kick_dephases=False` is now the default.
This reversed several earlier conclusions (see §4).

---

## 3. PSR vs FD — the scientific core

### Methodology (locked in, after corrections)
Always compare with **same start + same seed, paired, averaged over multiple
starts, at equal shot budget**. Two earlier mistakes were corrected:
1. **Kick-dephasing artifact** (dephasing wrongly applied to the kick gate).
2. **Unpaired / single-start comparison.**

### Corrected findings
| Question | Result |
|---|---|
| **Is PSR's gradient correct?** | **Yes — exact.** At free shots, PSR == FD to ≤1e-4 every start (H₂ and MaxCut). The difference is purely **variance**, zero bias. |
| **At equal budget under noise?** | **PSR beats FD** (H₂ fair paired: 13/15; shot-budget 5.6× better at low shots; dephasing 8/8 at T/T2\*=1). Advantage largest at **low shots / high noise / sharp landscapes**, shrinking to a tie at very high shot budgets. |
| **FD's only edge** | A **tuned ε** on **smooth, high-shot** landscapes, and **many-term (large M) parameters**. The natural *small* ε always fails (1/ε variance). PSR needs no ε. |
| **Local optima (MaxCut)** | Non-convex → neither reaches the true max cut 4; gradient precision does **not** help escape (exploration does). Coarse FD can even win via accidental exploration. **Basin-hopping** (clean descent + jumps + keep-best) decouples exploration → PSR finds better optima (3.84 vs 3.75). |

### The U-shaped ε-dilemma (FD's fundamental tradeoff)
- Small ε → 1/(2ε) variance → orbits the minimum.
- Large ε → secant bias → settles **confidently to a wrong point**.
- A tuned sweet spot exists on smooth landscapes, but PSR (no ε) is tuning-free
  and more precise. On **sharp** landscapes the window closes (no ε works).

### Chain-rule scaling (the honest cost caveat)
PSR cost per gradient component ∝ **M·n_sample** (M = terms a parameter affects);
FD is always **2 evaluations**. At equal budget, PSR's gradient RMSE grows ~linearly
with M (0.017→0.172 for M=1→6), FD's is flat → **crossover at M≈4–6**. So the
"saves shots" claim **dies for many-term parameters on smooth landscapes**;
it survives for **sparse (local) controls** or **sharp landscapes** (FD forced
to small ε regardless of M).

---

## 4. Decoherence — the corrected verdict

After fixing the kick-dephasing model **and** the unpaired comparison:
- PSR's gradient is **exact**; under noise at equal budget **PSR beats FD**.
- The earlier "FD beats PSR under decoherence", "rescale can't help", and the
  elaborate "short-kick decoherence fix" were **all artifacts** of the two
  mistakes — now **superseded**.
- `short_kick` mode (`observable_program_generator(..., short_kick=True)`) remains
  as an **exact** reformulation (f₊ branch `[−Hj, π/4]` instead of `[Hj, 7π/4]`,
  Pauli kicks are 2π-periodic) — a minor gate-time reduction, **moot for
  decoherence** under the faithful (kick=gate) model.

---

## 5. Analytical conclusions (where PSR actually wins)

PSR's advantage is **physics-tied, not a generic optimization speedup**: it needs
a **quantum (classically-hard) cost** and a **sharp landscape from long analog
evolution**. On smooth/classically-cheap landscapes FD uses large ε and the
advantage vanishes.

**Direction-resolution floors** (smallest gradient an estimator can still point):
```
FD floor :  |g| ~ σ/(2ε√N)      PSR floor:  |g| ~ √M·σ/√N
window where PSR uniquely resolves direction:  FD/PSR = 1/(2ε√M)
```
- **|g| > FD_floor**: both work, no PSR edge.
- **PSR_floor < |g| < FD_floor** (the window): **PSR uniquely makes progress** —
  the high-precision **"last mile"** to a ground state / target. Width grows as
  the landscape sharpens (longer evolution = more intractable) — the scaling
  works *for* PSR here, if control stays local (M bounded).
- **|g| < PSR_floor** (true **barren plateau**): both fail; gradient descent dies →
  random search/jumps → **FD's cheaper evaluation wins** (PSR precision wasted).

**Defensible claim:** _On sharp analog landscapes with local controls, PSR lowers
the directionless-gradient threshold by 1/(2ε√M) vs FD, extending useful descent
into a window of small gradients FD can't resolve — the high-precision final
approach to a quantum target (e.g. IQS / analog ground-state prep). It does not
save shots in general (chain-rule) nor cure barren plateaus._

---

## 6. Reproductions of Leng et al. 2022 (their flagship examples)

Their public code (github.com/YilingQiao/diffquantum) implements only MaxCut and
has **no explicit noise model** (the paper's FD failure was on noisy IBM hardware).
We reproduce both under an **explicit, controlled** noise budget.

- **H₂ VQE** (Fig 2b): minimize ⟨H_H2⟩ (5-term, E₀=−1.8355), analog ansatz reaches
  E₀. FD ε=0.01 **stalls** (gap 1.70 — their shot-noise finding); PSR (and tuned FD)
  converge. `h2_vqe_psr_vs_fd.py`.
- **MaxCut QAOA** (Fig 2d): 4-cycle, max cut **= 4** (true optimum; 3.87 is a
  local-optimum plateau, not the answer). FD ε=0.01 stalls at the |++++⟩ baseline;
  PSR/tuned-FD climb. `maxcut_psr_vs_fd.py`.

---

## 7. Figures (`differential_computing/figures/`)

| File | Shows |
|---|---|
| `all_comparisons_summary.png` | one-page results table — PSR wins/ties every regime |
| `fair_protocol_comparison.png` | shot-budget + dephasing, **fair paired** — PSR wins, biggest at low shots / high noise |
| `convergence_curves.png` | actual trajectories — PSR descends faster on H₂; MaxCut local-optima dominated |
| `optimization_loop_epsilon.png` | the **U-shaped ε-dilemma** (PSR pins, FD orbits/biases) |
| `h2_vqe_psr_vs_fd.png`, `maxcut_psr_vs_fd.png` | reproductions — small-ε FD stalls |
| `maxcut_basinhop.png` | basin-hopping — precise PSR finds better optima |
| `chain_rule_scaling.png` | PSR's shot advantage erodes with M (terms/parameter) |

---

## 8. Status & open threads

**Complete:** error-model + benchmark infrastructure; both Leng et al. examples
reproduced; fair paired protocol; figure set; analytical regime map.

**Test suite:** 215+ passing (`conda run -n qec_pg python -m pytest
differential_computing/tests/`). Study scripts in `tests/` (not pytest — run
directly).

**Deferred / next:**
- DSL/AWG: real per-channel pulse shapes, AOD 2-D fx/fy encoding, AWG comb
  summation, n_sample>1 emission.
- TD hardware path: `play_wf` op, per-group solves, cross-group placement.
- "Perfect PSR landscape" figure (sharp, small-feature-scale — the last-mile /
  IQS regime) as a candidate headline.
- IQS application demo (gradient-descend analog controls to match target
  properties) — the natural home for the last-mile advantage.

**Conventions:** `conda activate qec_pg`; local SimuQ fork in `src/` (don't pip
install public); PulseDSL at `/Users/syue99/research/RISC-Q/PulseDSL/src/DSL/`;
test code in `.py` first, then paste into notebooks; commit only after tests pass;
never `git add -A`.

**Honest one-liner:** _PSR's gradient is exact and, fairly compared, beats FD at
equal shot budget under noise — decisively in the low-shot / high-noise /
sharp-landscape regime. It is not a generic optimization speedup: its value is
precision-of-convergence for analog quantum control/simulation (the last mile),
it does not save shots for many-term parameters, and it does not cure barren
plateaus._
