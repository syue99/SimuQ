# Session Summary (2026-04-22 to 2026-04-23)

## Overview

Reviewed the compilation pipeline end-to-end with 1→2→3 atom single-trial
walkthrough. Found and fixed four bugs in the solver ↔ TweezerMapper ↔ verify
chain. Final results: all cases have gradient error < 1e-3 and per-segment
Hamiltonian norm < 0.02.

## Bugs Found and Fixed

### 1. Ledger was cheating
**Problem**: `_dressing_ops` stored the pre-compilation `evaluated_H` (original
PSR Hamiltonian) in the ledger. Verify read it back trivially — never tested
the solver's decomposition.

**Fix**: `_build_dressing_H(o_coef)` computes dressing H from the solver's
`o_coef + sol_gvars`. `_build_zz_H(J, q0, q1)` builds `J·Z_iZ_j`. Verify
sums all channel H's per segment instead of using stored H as complete answer.

### 2. Solver activated wrong channels for single-qubit H
**Problem**: For `H = x·Z₀ + X₀`, the solver activated dressing and ZZ because
dressing's `n₀n₁` expands to include `Z₀` sub-terms matching the target.

**Fix**: Body-count filter in `solver.py:build_eqs` — 1-body original target
terms only match 1-body instructions (detuning/rabi). Side-effect terms
(coeff=0, from dressing activation) still allow multi-body instructions so
the solver can balance dressing Z side-effects with detuning compensation.

### 3. Kick segment used wrong Hamiltonian
**Problem**: `map_hlist` applied solver's boxes (designed for `H_eval`) to ALL
segments including the kick (segment 1). The kick has a different Hamiltonian
`Hj` (e.g., `Z₀`, `X₀`, `Z₀Z₁`).

**Fix**: `_map_kick_segment(Hj, duration, t_cursor)` maps Hj directly to
channels: `Z→detuning(d=2*coeff)`, `X/Y→rabi(Ω=2*coeff)`, `ZZ→gate zone +
AOD return`. Stores original Hj in ledger as "kick" entry.

### 4. ZZ suppressed by dressing in TweezerMapper
**Problem**: `map_evaluated_H` skipped ZZ when dressing was present
(`has_dressing` flag). But the solver uses both dressing AND ZZ together
(`dressing Z₀Z₁ = -7.07` + `ZZ = +7.98` = `target 0.91`). Suppressing ZZ
left only dressing's `-7.07` in the ledger.

**Fix**: Removed the `has_dressing` suppression. Both dressing and ZZ are
emitted and recorded in the ledger.

## Other Changes

- **fix_time=True**: Solver fixes `t_solver = T` so coefficients match target
  directly (no time-coefficient trade-off).
- **Switch pruning**: Keeps single-qubit instructions when dressing is active.
- **Skip redundant AOD**: `_positions_match()` avoids AOD when atoms already
  at target positions.
- **Verify norm diagnostics**: `norm_check()` reports per-segment
  `||H_compiled - H_target||` with identity removed (global phase).
- **Rydberg2d conventions**: Fixed `_reconstruct_H_from_entry` to use correct
  conventions: detuning `H = -d*(I-Z)/2`, rabi `H = Ω/2*(cosφ·X - sinφ·Y)`.
- **to_pulsedsl_simple()**: Cell-8-style DSL emission with sigmatukey pulses.
- **Module cache clearing**: Notebook setup cell clears all cached modules.

## Final Results (127 tests, n_sample=1)

| Case | Seg 0,2 norm | Seg 1 norm | Gradient error |
|------|-------------|------------|----------------|
| 1q (x·Z₀ + X₀) | 0.018 | 0.00 | 8.9e-4 |
| 2q (sin(2x)·(Z₀Z₁+X₀+X₁)) | 0.001 | 0.00 | 1.5e-4 |
| 3q (same H, 3 atoms) | 0.005 | 0.00 | 3.2e-4 |

Note: n_sample=1 gives near-exact results because (1) all terms share the
same x-dependence, (2) short T=0.5, (3) QuTiP = infinite repetitions. On
hardware with finite shots and diverse parameter dependence, more samples needed.

## Files Modified

| File | Changes |
|------|---------|
| `src/simuq/solver.py` | body-count filter, side-effect equation fix, fix_time, switch pruning |
| `src/simuq/braket/diffQC_provider.py` | to_pulsedsl_simple, fix_time, verify norm diagnostics |
| `differential_computing/tweezer_mapper.py` | ledger honesty, kick compilation, ZZ unsuppressed, position skip |
| `differential_computing/verify_compilation.py` | norm_check, identity removal, kick handling, rydberg2d conventions |
| `differential_computing/tests/test_single_trial.py` | 1→2→3 atom walkthrough tests |
| `differential_computing/tests/test_tweezer_mapper.py` | Updated for new API |
| `differential_computing/tests/test_diffQC_provider.py` | Updated fake provider |
| `differential_computing/tests/test_pulse_ledger.py` | Updated verify assertions |
| `differential_computing/tests/dsl_schedule_demo.py` | DSL rendering script |
| `differential_computing/tests/single_trial_walkthrough.ipynb` | Walkthrough notebook |

## Commits

- `880a2ab` — Ledger honesty, solver channel selection, kick compilation, verify diagnostics
- `6872001` — Fix 2q/3q: allow dressing side-effects in solver, remove ZZ suppression
