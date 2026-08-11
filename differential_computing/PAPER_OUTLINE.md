# DiffSimuQ — ASPLOS Paper Structure (DEVICE-TARGET pivot, 2026-08-11)

Supersedes the 2026-07-02 outline. **The rescale / light-cone transfer map is
GONE from this paper** — it lives in the companion ML-conference paper
(ideal-target Hamiltonian learning). This paper is purely **device-target**:
raw compiled PSR computes the *exact* gradient of the deployed noisy program,
which is what device optimization / quantum control actually wants. Removing
rescale vacated the old §5.2 headline; three things move up to fill it
(marked ⇧ PROMOTED below).

Target: ASPLOS 2027 Fall deadline **Sept 9, 2026**. ~12 pages + refs.
Rapid-review reality: the first 2 pages must sell PL+systems in ASPLOS-native
terms; quantum physics minimal and downstream.

Reference leverage (do NOT re-derive / re-explain, cite):
- **Leng et al., NeurIPS'22 (arXiv:2210.15812)** — analog PSR foundation.
  Cite for Algorithm-1 kicks and the coherent-robustness Lemma 3.3 (unitary
  error only). **Our device-exactness theorem (§5.2) is the complementary
  NON-unitary/Lindblad result their Lemma 3.3 does not cover**; our §3 is the
  PL formalization (semantics + soundness) their paper lacks. Their H2 VQE
  (Fig 2b) is the experiment we reproduce and extend with explicit noise (Fig 7).
- **SimuQ, POPL** — AAIS abstraction + solver. Cite for Hamiltonian
  compilation; our compiler contributions are the *differentiable* additions
  (branch generation, kick lowering, zones/transport, ledger, verification).

Framing rails (post-pivot — enforce everywhere):
- Raw PSR is **UNBIASED for ∇C_noisy** (the device's own gradient), not
  "attenuated". "Attenuation" only exists relative to the *ideal* noise-free
  gradient — that gap is the companion paper's problem, a one-line pointer here.
- NO "O(1) correction" pillar, NO rescale figures, NO light-cone vocabulary.
- FD fails on-device for TWO independent reasons, both shown: (i) truncation on
  sharp landscapes (wrong sign), and (ii) the **δ/ε control-resolution floor**
  (real hardware can't set ε arbitrarily small; setpoint error δ~N(0,r) is
  amplified by 1/ε, a γ-independent O(1) bias). PSR is ε-free → immune to both.

---

## §1 Introduction (1.5 pp)

Three pitch pillars, in order:
1. **Certifiable gradients as a compiler artifact.** "Gradients are a program
   construct, not a hardware feature." Three routes on analog hardware:
   autodiff-a-simulator (intractable at scale), finite differences on-device
   (fails *silently* — wrong sign under truncation, and a δ/ε control floor no
   number of shots removes), or a language-level differentiation transformation
   preserved through compilation. The compiled transformation computes the
   **exact gradient of the deployed noisy program** — sign- and
   magnitude-correct for the device's own cost, by theorem (§5.2), at every
   shot budget. Backprop+XLA analogy.
2. **Hardware–program fit.** The analog device natively *is* the program:
   constant pulse depth T vs O(n·T²/ε) Trotter gates *per branch per landscape
   point* digitally; at NISQ fidelities the digital route can't run it; QEC
   fixes fidelity at polynomially more physical resources on a far timeline
   (credit QEC explicitly).
3. **Codesign twist.** The derivative of a pure-analog program is an
   analog–digital *hybrid* program — the transformation derives the machine's
   minimal digital ISA (one Pauli kick = one native CZ + virtual frames), which
   the cryo platform provides at 99.95%.

Contributions (each maps to a section + figure):
- Differentiable semantics + sound code transformation for analog observable
  programs (§3, Fig 2).
- **Device-exactness theorem:** raw compiled PSR is an unbiased estimator of the
  deployed program's own (noisy, Lindblad) gradient — the non-unitary
  complement to Leng et al.'s coherent-only robustness (§5.2). ⇧ PROMOTED
- Full-stack compiler from differentiated programs to neutral-atom pulse
  schedules, with an honest verification IR (§4, Fig 5, Tab 3).
- Hybrid-ISA observation + CZ-kick lowering with branch-symmetric error
  cancellation (§4.3).
- Noise-regime evaluation: FD's twin silent failures (truncation + δ/ε control
  floor) vs certifiable compiled PSR, and an end-to-end device-optimization loop
  where PSR descends and FD stalls (§5, Figs 3/6/7).
- Compile-at-scale + resource comparison vs digital emulation and vs exact
  simulation (§5.5, Figs 8/9).

**Fig 1: system overview** — program → diff transform → branches → tweezer
compilation → ledger/verify → pulses; one horizontal stack. `fig_architecture.py`.

## §2 Background & Motivation (1.5 pp)

- 2.1 Analog quantum programs in 3 paragraphs (H(θ,t) programs; cite SimuQ; no
  gate-model detour).
- 2.2 Why gradients: VQE / Hamiltonian learning / quantum control. Why you
  cannot autodiff the device (no tape) and cannot afford to autodiff a simulator.
- 2.3 The FD trap, shown not told: **Fig 3** (device-target, `landscape_device.py`
  → `build_paper_figs.fig3`). Sharp noisy landscape: even ε=0.15 secants are
  wrong-sign (truncation); panel B is the fixed-budget bias–variance U with the
  δ/ε control floor on the left arm and truncation on the right, raw PSR flat far
  below. Both FD failure modes in one figure.
- 2.4 Resource asymmetry: **Fig 4 = resource pillar** (`fig_resource_pillar.py`
  → `figR`): analog-native constant pulse depth vs O(n²T²/ε) digital Trotter 2q
  gates per branch, NISQ gate-error wall. Replaces the old Tab 2 (analytic,
  cheap; keep Tab 2 only if a reviewer wants exact per-gradient counts).
- **Tab 1: the three gradient routes** (rows: sim-autodiff / device-FD /
  compiled PSR; cols: scalability, bias, sign-safety, tuning knobs). Compiled-PSR
  bias cell = **"unbiased (device gradient)"**. NEW, conceptual, 5 lines.

## §3 Differentiable Analog Programs (2 pp) — mostly EXISTS in draft §2

- 3.1 Syntax + denotational semantics (condense draft §2.2–2.3; resolve the
  YIKAI/PENGYU inline TODOs about classical-function grammar).
- 3.2 Differential semantics + the code transformation (draft §2.4).
  **Fig 2: the transformation on a concrete program** — source vs transformed
  side-by-side with the τ-draw and ±kick highlighted. `fig_code_transform.py`.
- 3.3 Soundness (ideal semantics): the transformation computes the exact
  gradient of the *ideal* program; theorem statements (draft §2.5), proofs →
  appendix. (The *device*/noisy exactness is the separate §5.2 theorem — forward
  pointer here.)
- 3.4 The hybrid-class observation: input is pure-analog; output needs exactly
  one discrete primitive (Pauli kick). One paragraph — sets up §4.3.
- 3.5 Sample complexity: Chernoff bound theorem (draft §3.1.2–3.1.3), stated;
  proof → appendix.

## §4 Compiling Differentiated Programs to Atoms (2.5 pp)

- 4.1 AQAM + standard-form theorems (draft §3.1.1, condensed); solver via SimuQ.
- 4.2 Zoned compilation: interaction/gate/idle zones, AOD transport, position
  state machine. **Fig 5: zone architecture + one branch's schedule timeline**
  (positions + channels over the 3 segments). `fig_zone_schedule.py`.
- 4.3 Kick lowering: exp(−iφZZ) = phase·(virtual-Z)²·CP(−4φ); PSR angles ⇒ one
  native CZ on BOTH branches; branch-symmetric error cancels in f₋−f₊. Inline
  equation block, no figure.
- 4.4 The PulseLedger IR + verification methodology: reconstruct H per segment
  from meta-parameters, compare norms + end-to-end gradients. **Tab 3:
  verification round-trip** (1q/2q/3q/multi-layer: seg norms, gradient errors —
  refreshed via `tab3_verification.py`; 1q 8.9e-4/0.86%, 2q 2.2e-4/0.03%, 3q
  2.4e-5, multi-layer PSR-semantics 0.00%; kick seg-norm 0 everywhere).
- 4.5 Physical channels + AWG: 5 fixed modulator lines, COMB multi-tone; one
  paragraph + pointer to Fig 5's channel rows.

## §5 Evaluation (3 pp)

- 5.1 Methodology: literature-calibrated noise model (T2*, leakage
  post-selected, gate-error budget per Evered et al.); simulation as *validation
  proxy*; deterministic-τ for bias studies. Half page.

- 5.2 **The device-exactness theorem — the headline.** ⇧ PROMOTED (was an
  appendix lemma).
  - Statement: for θ-independent Markovian (Lindblad) noise, raw compiled PSR is
    an **unbiased** estimator of ∇C_noisy, the gradient of the device's own
    (mixed-state) landscape — sign- AND magnitude-correct for the deployed
    program, at any shot budget.
  - Proof in main text (short): the ±kick shift identity
    K₊ρK₊† − K₋ρK₋† = −i[H_j, ρ] is **algebraic** — it holds for any mixed ρ, so
    it commutes through the dissipator; the estimator's expectation is exactly
    ∂C_noisy/∂θ. This is precisely the regime Leng et al. Lemma 3.3 (unitary
    error) leaves open. ~½ page.
  - **Fig 6: estimator quality** (two panels; `build_paper_figs.fig5`+`fig6`,
    consider merging into one 2-panel figure).
    (a) accuracy vs control resolution r: FD's δ/ε penalty is control-resolution,
        **γ-independent** (two decoherence levels overlap); raw PSR ~1e-4.
    (b) finite shots: raw PSR → ∇C_noisy as ~N^{-1/2}; oracle-FD floors at the
        δ/ε bias no shots can remove.

- 5.3 **δ/ε control-resolution failure (the "no arbitrarily-small ε" argument).**
  ⇧ PROMOTED (was implicit in captions). Short subsection: on real hardware the
  step ε is floored by control resolution r and the setpoint carries error
  δ~N(0,r); the FD estimate inherits a δ/ε≈O(1) bias that is independent of the
  decoherence rate and survives infinite shots. PSR carries no ε, so it is
  structurally immune. Points at Fig 3B (floor arm) and Fig 6a. ~⅓ page.

- 5.4 **Device optimization loop — the application payoff.** ⇧ PROMOTED (was
  supplement-only). End-to-end: minimize a noisy cost by gradient descent, PSR
  vs FD at the SAME per-step shot budget, same start/lr. **Fig 7: H2 VQE under
  decoherence** (`h2_vqe_psr_vs_fd.py` — reproduces Leng et al. Fig 2b and adds
  an explicit noise budget): PSR converges to E0; FD stalls / steps in the wrong
  direction as the gradient shrinks. This is the natural device-target result —
  you are minimizing the machine's real cost, which is exactly what raw PSR is
  exact for. (Backups: `optimization_loop_demo.py`, `vqe_noisy_comparison.py`.)

- 5.5 **Systems scaling.** **Fig 8: compile at scale** (C7, `compile_scaling`):
  compile time to n=12, branches linear, pulse depth constant. **Fig 9: the cost
  wall** (`cost_wall.py` → `build_paper_figs.fig8`, correction line dropped):
  wall-clock of exact noisy simulation (exponential, dashed beyond feasible) vs
  differentiable compilation (flat/linear), shaded intractable region — the
  toolchain runs where the simulator cannot.

## §6 Discussion & Limitations (0.5 pp)

Honesty block (reframed): raw PSR is unbiased for the **device's own** gradient
under θ-independent Markovian noise — the right target for optimization, control,
and QAOA on the deployed machine. Recovering the *ideal* noise-free gradient
(Hamiltonian learning / inverse simulation) is a distinct problem solved by a
local transfer-map correction — **companion paper, one-line pointer** (arXiv ok
for dual-submission integrity). Other limits: θ-independent noise assumption,
kick discretization error (bounded, §4.3), QEC eventually shifts the resource
comparison, hardware demonstration is future work (cryo platform).

## §7 Related Work (0.5 pp)

Leng et al. NeurIPS'22 (analog PSR; coherent-only robustness; uncompiled — we
add PL formalization, compilation, and the non-unitary exactness theorem), SimuQ
POPL (non-differentiable compilation), digital PSR / PennyLane / stochastic PSR
(Banchi–Crooks), pulse-level stacks (Qiskit Pulse, Pulser), Hamiltonian learning
/ IQS application pull, our companion transfer-map paper (ideal-target).

## §8 Conclusion (0.25 pp)

---

## Master figure/table list (device-target)

| # | Content | Source | Status |
|---|---------|--------|--------|
| Fig 1 | Full-stack overview, program→atoms | `fig_architecture.py` | EXISTS (PNG) — **needs PDF into paper_fig/** |
| Fig 2 | Code transformation, source vs transformed | `fig_code_transform.py` | EXISTS (PNG) — **needs PDF** |
| Fig 3 | FD trap: sharp noisy landscape + secants + RMSE-vs-ε U (device+δ) | `build_paper_figs.fig3` | **DONE** |
| Fig 4 | Resource pillar: analog-native vs digital Trotter per branch | `figR` | **DONE** (replaces Tab 2) |
| Fig 5 | Zone architecture + branch schedule timeline + channels | `fig_zone_schedule.py` | EXISTS (PNG) — **needs PDF** |
| Fig 6 | Estimator quality: (a) accuracy vs r, (b) finite-shot → ∇C_noisy | `build_paper_figs.fig5`+`fig6` | **DONE** (consider merging to 1 fig) |
| Fig 7 | H2 VQE descent under noise: PSR converges, FD stalls | `h2_vqe_psr_vs_fd.py` | data EXISTS — **promote + ACM replot** |
| Fig 8 | Compile at scale (C7): time, branches, constant depth | `build_paper_figs.fig7` | **DONE** |
| Fig 9 | Cost wall: exact-sim wall-clock vs toolchain cost (no correction line) | `build_paper_figs.fig8` | **DONE** |
| Tab 1 | Three gradient routes (concept; PSR = unbiased device grad) | — | TO WRITE (no compute) |
| Tab 2 | Trotter-vs-analog exact per-gradient counts | C7 + formula | OPTIONAL (Fig 4 covers it) |
| Tab 3 | Verification round-trip (seg norms, gradient errors, multi-layer) | `tab3_verification.py` | **DONE** (refreshed; cache `figures/tab3_verification.json`) |
| Supp. | Rescale/ideal-target, other opt loops, plateau, generality, ZNE, leakage | companion paper + supplement | per locked scoping |

**Companion figures now OWNED by the ML paper** (`build_ml_paper_figs.py` →
`paper_fig_ml/`): shots_scaling, decomposition, bias_vs_n, gate_error. Not in
this paper.

Consistency pass before submission: every figure labeled in **T/T2\*** (not
absolute T); regime story stated (0.15 headline / 0.5 for the trap + estimator
figures); no "correction"/"rescale"/"light-cone" strings anywhere in the main
text.

## Immediate to-do (post-pivot, no big sims except Tab 3)
1. Schematics Fig 1/2/5 → emit PDF into `paper_fig/` (add save-PDF to the three
   `fig_*.py` scripts or fold into `build_paper_figs.py`).
2. Fig 7: ACM-style replot of `h2_vqe_psr_vs_fd` into `paper_fig/` (device-target
   caption: track the real noisy cost; PSR descends, FD stalls).
3. ~~Tab 3: re-run for 1q/2q/3q/multi-layer~~ DONE (`tab3_verification.py`).
4. Prose: write the §5.2 theorem+proof and the §5.3 δ/ε subsection.
5. Tab 1 (concept) + optional Tab 2 collapse into Fig 4.
