# DiffSimuQ: Project Summary and Implementation Plan

## What We Have Built (as of 2026-04-23)

DiffSimuQ is a full-stack automatic differentiation system for analog quantum
programs, targeting neutral-atom tweezer arrays. 128 tests passing.

### Layer 1 — Differentiable Quantum Program Generator

- `observable_program_generator.py` generates PSR branches from `Parametrized_Hamiltonian`.
- `combine_gradient.py` assembles the stochastic gradient estimate.
- Gradient formula: `d<O>/dθ = T * E_τ [ Σ_j (du_j/dθ)(τ) * (f+ - f-) ]`

### Layer 2 — QuTiP Simulation Backend

- `qutip_sequential.py` chains sesolve for multi-segment H_lists.
- `diffQCProvider.run(backend="qutip")` computes gradient end-to-end.
- **Validated**: PSR matches finite differences to <0.001% on 2-3 qubit models.

### Layer 3 — Tweezer Compilation (SimuQ → Schedule Ops)

- `tweezer_mapper.py` maps H_list branches to hardware schedule op dicts.
- **Zone architecture**: interaction / gate / idle zones with position state machine.
- **Solver improvements**:
  - Body-count filter: single-qubit targets use detuning+rabi only (no dressing/ZZ)
  - Side-effect equations: dressing Z terms visible to detuning for compensation
  - fix_time=True: exact coefficient matching
  - Idle qubit placement: spectator qubits placed far away (1000 μm)
- **Kick compilation**: PSR kick (seg 1) mapped directly to channels (Z→detuning,
  X/Y→rabi, ZZ→gate zone + return). Not through solver boxes.
- **Channel layout** (n qubits):
  - ch[0..n-1] detuning, ch[n..2n-1] rabi, ch[2n] dressing UV, ch[2n+1] ZZ gate, ch[2n+2] AOD

### Layer 4 — PulseLedger + Honest Verification

- Ledger stores **solver's H** (not pre-compilation H):
  - Dressing: `_build_dressing_H(o_coef)` from solver params
  - ZZ: `_build_zz_H(J, q0, q1)`
  - Kick: original Hj stored directly
  - Detuning/rabi: meta-params reconstructed via rydberg2d conventions
- **Verify reports**:
  - Per-segment `||H_compiled - H_target||` norm (identity-free)
  - Gradient error (ground truth vs compiled)
- **Multi-layer verify**: `verify_multilayer_compilation()` checks compiled gradient
  across ALL layers combined (not just one layer).

### Layer 5 — DSL + AWG Bridge

- `to_pulsedsl_simple()` emits sigmatukey pulses to PulseDSL channels.
- `schedule.return_pulse_sequence_by_channel()` extracts per-channel sequences.
- `compile_gates()` generates AWG waveform samples via gatedict.
- FIFO drain thread for PulseDSL MMIO named pipe.

### Verification Results (n_sample=1, tol=0.1)

| Case | Seg 0,2 norm | Seg 1 norm | Gradient error |
|------|-------------|------------|----------------|
| 1q (x·Z₀+X₀) | 0.018 | 0.00 | 8.9e-4 |
| 2q (sin(2x)·(Z₀Z₁+X₀+X₁)) | 0.001 | 0.00 | 1.5e-4 |
| 3q (same, 3 atoms) | 0.005 | 0.00 | 3.2e-4 |
| Multi-layer (2-layer dressing) | — | — | 0.06% (all compiled) |

### Test Coverage: 128 tests

- Gradient pipeline (46), tweezer mapper (24), diffQC provider (18),
  pulse ledger (9), single-trial walkthrough (19), multi-layer compiled (1),
  DSL demo script, organized notebook.

---

## What Remains

### Your side (DSL + AWG refinement)
- **n_sample > 1 DSL emission**: emit separate schedules per PSR branch
- **DSL pulse scheduling**: real PulseDSL setup (parallel ops, smarter scheduling)
- **AWG compile_gates**: correct pulse shapes per channel type (detuning, rabi,
  dressing, ZZ, AOD — each has different physical pulse shape)

### Optional (code/architecture improvements)
- **Dressing-vs-gate heuristic**: solver still uses dressing for clean sparse pairs;
  should prefer gate ZZ when only one pair has interaction
- **Solver offset regularization**: `offset = sqrt(1e5 * tol / sqrt(nvar))` could be
  tuned for tighter convergence
- **QuTiP solver tolerance**: sesolve defaults (rtol=1e-3) limit PSR/FD agreement
  at high n_sample

---

## Implementation Plan (Next Month)

### Week 1: Paper writing — Sections 1-2
- Write introduction (analog AD motivation, IQS connection)
- Revise Section 2 (theory) with PL-friendly preliminaries:
  - What a Hamiltonian is, parameterized evolution, analog vs digital
  - Concrete single-qubit example of parameter shift
  - Why we want derivatives (bridge to applications)

### Week 2: Paper writing — Sections 3-4 (systems)
- Section 3: System architecture diagram, zone compilation, PulseLedger IR,
  verification methodology, channel layout
- Section 4: Solver improvements (body-count filter, side-effects, fix_time,
  idle placement), kick compilation, multi-layer compilation

### Week 3: Evaluation benchmarks + DSL refinement
- Run convergence plots: gradient error vs n_sample (1q, 2q, 3q)
- Compilation scaling: schedule ops vs qubit count
- DSL refinement: real pulse shapes per channel, n_sample>1 emission
- AWG waveform generation for all channel types

### Week 4: Paper writing — Sections 5-7 + figures
- Section 5: Evaluation results, convergence plots, compilation overhead,
  multi-layer verification, IQS gradient-descent demo
- Section 6: Related work (SimuQ, PennyLane, IQS)
- Section 7: Conclusion
- Key figures: architecture diagram, zone positions, PulseLedger timeline,
  convergence plot, scaling plot, IQS demo

### Key Figures Needed
1. System architecture diagram (full stack: H → PSR → compile → DSL → AWG)
2. Zone architecture with atom positions at each stage
3. PulseLedger timeline (positions + pulses + channels over time)
4. Gradient convergence: PSR error vs n_sample
5. Compilation quality: norm error vs qubit count
6. Multi-layer demo: compiled vs original gradient

---

## Connection to IQS (arXiv:2601.12239)

DiffSimuQ is the gradient engine for Inverse Quantum Simulation on analog
hardware. IQS does gradient-based optimization to discover parent Hamiltonians
from target properties. DiffSimuQ provides: PSR gradients on Rydberg hardware,
full-stack compilation with verified correctness, multi-layer evolution support.

The two-layer dressing example demonstrates the core IQS workflow:
parameterize one interaction layer, keep others fixed, gradient-descend to
discover target properties — all compiled to hardware schedules with 0.06%
gradient accuracy.
