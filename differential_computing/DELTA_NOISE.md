# DELTA_NOISE.md — the setpoint rule as implemented (P0-0)

2026-09-04. Figures 8 (`tests/build_F6.py`) and 9 (`tests/build_Floop_trajectory.py`) are
regenerated under the rule below. Fig 2 follows.

## The rule (owner's ruling, 2026-09-04): a draw per *change* of programmed value

r = 0.02, zero-mean Gaussian, in coefficient units. **A draw δ is taken whenever the
programmed coefficient value changes, and held for every shot until it changes again.**
Times (τ, segment boundaries) and the inserted gate carry no δ. The reference ∇C_device is
the shot-free gradient at nominal θ₀ (`build_F6.py:145`, central difference h = 10⁻³).

| estimator | values dialed per estimate | draws | code |
|---|---|---|---|
| FD | θ₀ + ε/2 and θ₀ − ε/2 | **2** (one per probe, N/2 shots each) | `fd_est`, `build_F6.py:354-356` |
| PSR | the source coefficient u, for all 2m branches | **1**, shared by all 96 branches | `_psr_setpoints` → `_branch_at`, `build_F6.py:239-248` (branch expectations at θ₀+δ from a 7-point grid over ±3r, `:190-200`) |
| NSR | u + σh_κ·v, a different (κ, σ) on (almost) every execution | **one per execution** | `_nsr_deltas`, `build_F6.py:264-271` |

Selected by `DELTA_MODEL=per_change` (`build_F6.py:106-109`, the default).

This differs from the handover's P0-0 NSR rule ("one draw per distinct (κ, σ) actually
executed", floor Ω̄r|f′|/√3). The owner ruled that rule wrong for this device: NSR's
stochastic sampler re-dials a new value on each execution, so it gets a new draw each time,
whereas FD holds each probe and PSR holds the source value. The handover's clause "if NSR
does not floor, δ is being redrawn per execution … an implementation bug" is therefore
overridden by design, not by accident. Both alternative models remain selectable as
diagnostics: `per_value` (the handover's rule) and `per_programming` (a draw per execution
for every estimator, including PSR's branches).

Seeds: each repetition s uses one stream `default_rng(1000 + s)` (`:415`; `500 + s` for the
inset, `:454`); δ is drawn from it before the shots, so δ and shot noise are independent
variables but not separately seeded. δ seeds are the same across the six series, i.e.
series are paired by repetition.

## Why the operating point moved (1.940 → 1.757)

Under any frozen-draw rule PSR returns the exact gradient at θ₀+δ, so its exposure is
|f″|·r at first order. At θ₀ = 1.940, f″ = 10.1 and that is 0.203 = 53% of |f′| — PSR floors
*above* FD, and NSR (with f″ this large, C′ at its shifted setpoints is ~2, not 0.385)
would floor at 39% under the handover's rule. Nothing about δ fixes that; it is the
curvature of the point C.3's rule selected. The new rule takes the C″ = 0 crossing inside
the window that keeps M = 5 (θ₀ = 1.757, |f′| = 1.449), where every estimator's
displacement is second order and the comparison isolates what each does with the draw.
Longer T (steeper landscape) was checked and rejected: ε* shrinks only as T^{−2/3} while the
second-order exposures grow with T; at T = 10 nothing improves.

## Measured floors at θ₀ = 1.757 (RMSE at N = 10⁶, 100 seeds, per-change rule)

| series | RMSE | % of \|f′\| | set by |
|---|---|---|---|
| FD @ ε* = 0.208 | 0.180 | 12.4% | **δ/ε**: √2·r\|f′\|/ε* = 0.197 alone; truncation 0.123 is secondary |
| FD @ ε = 0.05 fixed | 0.817 | 56% | δ/ε at an untuned step (0.82 predicted) |
| PSR | 0.028 | 1.9% | second-order displacement (√3/2)\|f‴\|r² = 0.024 |
| PSR + gate | 0.024 | 1.7% | same; exact gate bias 0.0037 is negligible here |
| NSR M=∞ | 0.0095 | 0.7% | **no floor** (slope −0.51); residual δ-blur bias r²f‴/2 = 0.014 masked by the 24-mode sampler's −0.012 truncation, see NUMBERS.md |
| NSR M=5 (truncated, plotted) | 0.0102 | 0.7% | truncation −0.0148 cancelled by δ-blur +0.0137 at this point (net −0.0007) |
| NSR M=5 (rejection, reported) | 0.0092 | 0.6% | |

First-order PSR displacement |f″|r = 0.0001. The handover's NSR floor formula gives 0.167
here; it does not apply under the ruling.

## Diagnostics (not paper figures) — same θ₀ = 1.757, seeds, grid; RMSE at N = 10⁶

| series | paper (per-change, r = 0.02) | δ off (`DELTA_R=0`) | handover's per-setpoint rule (`per_value`) |
|---|---|---|---|
| FD @ ε* | 0.180 (ε* = 0.208) | 0.058 (ε* retunes to 0.141; truncation only) | 0.180 (identical: same rule for FD) |
| FD @ ε = 0.05 | 0.817 | 0.027 | 0.817 |
| PSR | 0.028 | 0.008 (slope −0.50, no floor) | 0.028 (identical: same rule for PSR) |
| PSR + gate | 0.024 | 0.008 | 0.024 |
| NSR M=∞ | 0.0095 (slope −0.51) | 0.016 (the 24-mode sampler's own truncation −0.012, no δ-blur to cancel it) | **0.078** (slope −0.20; exact Ω̄-weighted sum predicts 0.079, leading order Ω̄r\|f′\|/√3 = 0.167) |
| NSR M=5 | 0.010 | 0.018 (truncation 0.0148) | 0.077 |

Reading: δ is three quarters of FD's floor (0.180 → 0.058) and all of PSR's (0.028 → 0.008);
NSR is the only estimator whose δ exposure depends on the rule — none under per-change,
5.4% of |f′| under the handover's per-setpoint rule. Caches:
`figures/F6_floor_amplification_diag_delta_off.json`, `…_diag_per_value.json`. The
previous paper build (per_programming, θ₀ = 1.940, commit 3e0338c) is kept as
`figures/F6_diag_per_programming_th1940.json`.

## Figure 9 (`tests/build_Floop_trajectory.py`) — the same rule, per optimizer step

One gradient estimate = one optimizer step; a fresh `Dial` (`build_Floop_trajectory.py:158`)
per estimate holds one draw per coefficient and redraws a coefficient's draw only when its
programmed value changes, so every draw is redrawn between steps.

| estimator | programs per step | draws | code |
|---|---|---|---|
| FD | 4 probes (θ ± ε/2·e_ℓ) | each probe redraws the coefficient it moves, keeps the other's | `fd_grad_C` `:184` |
| PSR | residual measurement at θ, then 64 + 128 branches, all dialing θ | **one draw vector** for the whole step: the estimate is ∇C(θ+δ) | `iqs_grad` `:302`, `_obs_grads_psr` `:214` |
| NSR | residual at θ, then per coefficient ℓ the shifted programs θ + s·e_ℓ, a new (κ, σ) on almost every execution | shifted coefficient: fresh draw whenever (κ, σ) differs from the previous execution's; other coefficient held; 3-point stencil at each shift, quadratic in δ | `_obs_grads_nsr` `:244` |

Seeds: `default_rng(seed·131 + 17)` per seed (`:319`) serves shots and draws; δ and shot
noise are independent variables, not separately seeded. Optimizer: η_t = η₀/(1 + t/20)
(`:326`), reported iterate = tail average (`tail_avg`, `:458`).

**What the rule does in a loop.** The IQS residual r is measured at one held setpoint, so
that program takes one draw and both shift rules inherit the displacement ∇C(θ+δ)
through r; NSR's per-execution averaging only cleans its ∇⟨O⟩ factor, which multiplies
r → 0 at the optimum. So in Fig 9 NSR cannot beat PSR on δ (Fig 8 is where that shows);
under a fixed step both jitter by r·√(ημ/2) per axis (0.017 on the stiff axis here), and
that jitter is what the decaying step and the tail average remove. FD's setpoint term
√2·r|∇C|/ε vanishes at the optimum; what remains for FD is truncation bias ε²C‴/(24 μ_soft)
plus shot variance ∝ 1/ε, and neither averages away.

Floors (median ‖θ̄₅₀ − θ*‖, 20 seeds, B = 6000, T = 2.5): PSR 0.012, NSR 0.011, FD ε = 0.1
(oracle) 0.024, FD 0.5: 0.097, FD 0.05: 0.039. δ off was not rerun for this figure (the
per-step audit at θ* with δ off gives the shift rules 0.016 under a fixed step).
