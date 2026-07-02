# DiffSimuQ — ASPLOS Paper Structure (locked pitch, 2026-07-02)

Target: ASPLOS 2027 Fall deadline **Sept 9, 2026**. ~12 pages + refs.
Rapid-review reality: the first 2 pages must sell PL+systems in ASPLOS-native
terms; quantum physics minimal and downstream. NO light-cone vocabulary/figures
(P2 material); rescale appears only as an O(1)-cost black-box correction.

Reference leverage (do NOT re-derive / re-explain, cite):
- **Leng et al., NeurIPS'22 (arXiv:2210.15812)** — analog PSR foundation.
  Cite for Algorithm-1 kicks and the coherent-robustness Lemma 3.3; position
  our noise results as the complementary NON-unitary regime, and our §3 as the
  PL formalization (semantics + soundness) their paper lacks.
- **SimuQ, POPL** — AAIS abstraction + solver. Cite for the Hamiltonian
  compilation machinery; our compiler contributions are the *differentiable*
  additions on top (branch generation, kick lowering, zones/transport, ledger,
  verification). Saves ~1.5 pages of background.

---

## §1 Introduction (1.5 pp)

The three pitch pillars, in order:
1. **Certifiable gradients as a compiler artifact.** "Gradients are a program
   construct, not a hardware feature." Three gradient routes on analog quantum
   hardware: autodiff-a-simulator (intractable at scale), finite differences
   on-device (fails *silently* — wrong sign at every feasible ε, ~20% bias
   floor at ∞ shots), or a language-level differentiation transformation
   preserved through compilation (exact semantics, descent-safe under noise by
   lemma, 0.5–1% after an O(1) tuning-free correction). Backprop+XLA analogy.
2. **Hardware–program fit.** The analog device natively *is* the program:
   constant pulse depth T vs O(n·T²/ε) Trotter gates *per branch per landscape
   point* digitally; at NISQ fidelities the digital route can't run it; QEC
   fixes fidelity at polynomially more physical resources on a far timeline
   (credit QEC's long-term potential explicitly).
3. **Codesign twist.** The derivative of a pure-analog program is an
   analog–digital *hybrid* program — the transformation derives the machine's
   minimal digital ISA (one Pauli kick = one native CZ + virtual frames), which
   the cryo platform provides at 99.95% (costing the estimator +0.4pp).

Contributions list (each bullet maps to a section + figure):
- Differentiable semantics + sound code transformation for analog observable
  programs (§3, Fig 2)
- Full-stack compiler from differentiated programs to neutral-atom pulse
  schedules, with an honest verification IR (§4, Fig 4, Tab 3)
- The hybrid-ISA observation + CZ-kick lowering with branch-symmetric error
  cancellation (§4.3)
- Noise-regime evaluation: FD's silent failure vs certifiable compiled PSR;
  size-robust accuracy at realistic gate fidelity (§5, Figs 3/5/6)
- Compile-at-scale + resource comparison vs digital emulation (§5.4, Fig 7,
  Tab 2)

**Fig 1 (A1): system overview** — program → diff transform → branches →
tweezer compilation → ledger/verify → pulses; one horizontal stack. REDRAW.

## §2 Background & Motivation (1.5 pp)

- 2.1 Analog quantum programs in 3 paragraphs (what H(θ,t) programs are; cite
  SimuQ for the machine abstraction; no gate-model detour).
- 2.2 Why gradients: VQE / Hamiltonian learning / control; why you cannot
  autodiff the device (no tape) and cannot afford to autodiff a simulator.
- 2.3 The FD trap, shown not told: **Fig 3 = landscape_and_distance_noisy**
  (EXISTS; polish pass: T/T2* labels) — both FD arms wrong-sign.
- 2.4 Resource asymmetry: **Tab 2 = Trotter-vs-analog gate counts** (CHEAP
  BUILD from C7 data + 2nd-order Trotter formula; per branch and per full
  gradient) + one sentence QEC overhead/timeline (honesty rail).
- **Tab 1: the three gradient routes** (rows: sim-autodiff / device-FD /
  compiled PSR; cols: scalability, bias, sign-safety, tuning knobs). NEW,
  conceptual, 5 lines.

## §3 Differentiable Analog Programs (2 pp) — mostly EXISTS in draft §2

- 3.1 Syntax + denotational semantics (condense draft §2.2–2.3; resolve the
  YIKAI/PENGYU inline TODOs about classical-function grammar).
- 3.2 Differential semantics + the code transformation (draft §2.4).
  **Fig 2 (A2): the transformation on a concrete program** — source and
  transformed program side-by-side with the τ-draw and ±kick highlighted.
  REDRAW (code-style figure, not physics).
- 3.3 Soundness: differential logic theorem statements (draft §2.5); proofs →
  appendix/extended version.
- 3.4 The hybrid-class observation: input program is pure-analog; output
  program needs exactly one discrete primitive (Pauli kick). One paragraph —
  sets up §4.3.
- 3.5 Sample complexity: Chernoff bound theorem (draft §3.1.2–3.1.3), stated;
  proof → appendix.

## §4 Compiling Differentiated Programs to Atoms (2.5 pp)

- 4.1 AQAM + standard-form theorems (draft §3.1.1, condensed); solver via
  SimuQ cite.
- 4.2 Zoned compilation: interaction/gate/idle zones, AOD transport, position
  state machine. **Fig 4 (A3): zone architecture + one branch's schedule
  timeline** (positions + channels over the 3 segments). REDRAW.
- 4.3 Kick lowering: exp(−iφZZ) = phase·(virtual-Z)²·CP(−4φ); PSR angles ⇒ one
  native CZ on BOTH branches; branch-symmetric error cancels in f₋−f₊. Small
  inline equation block, no figure needed.
- 4.4 The PulseLedger IR + verification methodology: reconstruct H per segment
  from meta-parameters, compare norms + end-to-end gradients. **Tab 3:
  verification round-trip** (1q/2q/3q/multi-layer: seg norms, gradient errors —
  numbers EXIST).
- 4.5 Physical channels + AWG: 5 fixed modulator lines, COMB multi-tone; one
  paragraph + pointer to Fig 4's channel rows.

## §5 Evaluation (2.5 pp)

- 5.1 Methodology: literature-calibrated noise model (T2*, leakage
  post-selected, gate error budget per Evered et al.); simulation as
  *validation proxy*; deterministic-τ for bias studies. Half page.
- 5.2 **Estimator quality — the headline.**
  - Lindblad-PSR lemma (statement): raw compiled PSR = unbiased estimator of
    the device's own landscape gradient ⇒ shares FD's attenuation exactly,
    sign-safe always; proof sketch → appendix.
  - **Fig 5: shots_scaling + decomposition** (EXISTS; needs regime-consistency
    replot decision): only rescaled PSR converges ~N^{-1/2}; FD-best and raw
    PSR floor; shots kill variance, only the correction kills bias.
  - **Fig 6: bias_scaling_relative + gate-error variants** (EXISTS): flat
    ~0.6–0.75% vs FD's flat ~18–20%, n=3–7, local+extensive; 99.95% cryo CZ
    costs ≤0.4pp. One sentence on the O(1) correction ("constant-size
    subsystem, independent of qubit count; derived in the extended version").
- 5.3 Robustness corners (short): plateau behavior, generality sweep,
  gate-error mechanism (branch cancellation). Text + citations to our
  supplement figures.
- 5.4 **Systems scaling.** **Fig 7: compile_scaling (C7)** (EXISTS): compile
  time to n=12, branches linear, pulse depth constant. **Fig 8 (F2, TO BUILD):
  the cost wall** — wall-clock of exact noisy simulation (exponential, dashed
  beyond feasible) vs compilation + correction cost (flat/linear), shaded
  intractable region: the toolchain runs where the simulator cannot.

## §6 Discussion & Limitations (0.5 pp)

Honesty block: raw PSR is attenuated (sign-safe, not unbiased — only the
correction restores magnitude); correction assumes θ-independent Markovian
noise + moderate gradients; QEC'd digital machines eventually change the
resource comparison; hardware demonstration is future work (cryo platform).
Pointer to extended version / P2 for the correction's derivation.

## §7 Related Work (0.5 pp)

Leng et al. NeurIPS'22 (analog PSR; coherent-only robustness; uncompiled),
SimuQ POPL (non-differentiable compilation), digital PSR / PennyLane /
stochastic PSR (Banchi–Crooks), pulse-level stacks (Qiskit Pulse, Pulser),
Hamiltonian learning / IQS application pull.

## §8 Conclusion (0.25 pp)

---

## Master figure/table list

| # | Content | Status |
|---|---------|--------|
| Fig 1 (A1) | Full-stack overview, program→atoms | **REDRAW** (SimuQ reuse not allowed) |
| Fig 2 (A2) | Code transformation, source vs transformed program | **REDRAW** |
| Fig 3 | FD trap: noisy landscape + secants + error-vs-ε (landscape_and_distance_noisy) | EXISTS — polish (T/T2* labels) |
| Fig 4 (A3) | Zone architecture + branch schedule timeline + channels | **REDRAW** |
| Fig 5 | Shots scaling + bias/variance decomposition (before/after correction) | EXISTS — regime-consistency replot decision |
| Fig 6 | Relative bias vs n + 99.95%/99.9% CZ variants | EXISTS (bias_scaling_relative, bias_scaling_gate_error) |
| Fig 7 | Compile at scale (C7): time, branches, constant depth | EXISTS |
| Fig 8 (F2) | Cost wall: exact-sim wall-clock vs toolchain cost vs n | **TO BUILD** (cheap, mostly timing runs) |
| Tab 1 | Three gradient routes (concept) | TO WRITE (no compute) |
| Tab 2 | Trotter-vs-analog resources per branch / per gradient | **TO BUILD** (cheap, from C7 + formula) |
| Tab 3 | Verification round-trip (seg norms, gradient errors, multi-layer) | numbers EXIST |
| Supp. | Optimization loops (H2/MaxCut), plateau, generality, ZNE, leakage | EXIST (supplement only, per locked scoping) |

Consistency pass before submission: every figure labeled in **T/T2\*** (not
absolute T); one operating regime story (0.15 headline / 0.5 for the trap
figure, stated explicitly); Fig 5 may need a re-run to match (user decision —
expensive, discuss first).
