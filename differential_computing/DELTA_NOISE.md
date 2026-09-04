# DELTA_NOISE.md — the setpoint rule as implemented (P0-0)

2026-09-04. Figure 8 (`tests/build_F6.py`) is the first figure regenerated under
the rule. Figs 2 and 9 follow and will be appended here.

## The rule, per estimator

`r = 0.02` (coefficient units), zero-mean Gaussian, **one draw per programmed
setpoint, frozen over every shot taken at that setpoint within one gradient
estimate, redrawn for the next estimate**. δ applies to programmed coefficients
only — never to τ, segment boundaries, or the inserted gate (that is ε_ins).

| estimator | draws per estimate | code |
|---|---|---|
| FD | 2 (one per probe θ ± ε/2) | `build_F6.py` `fd_est()` — `dp, dm = rng.normal(0, R_CTRL), rng.normal(0, R_CTRL)` once per call |
| NSR | one per distinct (κ, σ) actually programmed | `_setpoint_table()` returns `rng.normal(0, R_CTRL, 2*n_modes)`, indexed `2κ + [σ>0]`; used by `nsr_est`, `nsr_trunc_est`, `nsr_rej_est` |
| PSR | **1**, shared by all 2m branches | `psr_est()` / `psr_gate_est()` — `d = rng.normal(0, R_CTRL)` once, then `_branch_at(d, ...)` |

Reference `∇C_device` is evaluated at nominal θ, δ-free and shot-free. δ is drawn
from the same per-estimate `default_rng(1000+seed)` stream as the shots but
before them, so δ and shot noise are independent draws; seeds are unchanged from
the pre-δ build.

**PSR needed real work, not a model.** Every PSR branch programs the source
coefficient, so one shared δ makes the estimate the exact device gradient *at
θ₀+δ*. Realizing that requires the 2m branch expectations at the dialed
coefficient, so `_psr_branches()` is now evaluated on a 7-point grid over
±3r and interpolated per estimate (`_branch_at`). The alternative — asserting
the estimator equals the shifted exact gradient plus shot noise — would have
made PSR's floor a model of itself rather than a simulation, which is not
something a floor claim should rest on.

## Floors, predicted vs measured (Fig. 8, θ₀ = 1.940)

Measured floor = RMSE at N = 10⁶, 100 seeds.

| series | predicted | measured | verdict |
|---|---|---|---|
| PSR | \|f″\|·r = **0.2028** | **0.2178** | confirmed (residual is 3rd order) |
| PSR + gate | same + ε_ins bias 0.0138 | **0.2179** | displacement dominates the gate bias |
| NSR M=∞ | leading order Ω̄r\|f′\|/√3 = 0.0445; **exact-F′ 0.1506** | **0.1496** | confirmed against the exact form (0.7%) |
| NSR M=5 | as above | **0.1495** | same |
| FD @ ε*=0.252 | B.6.4 analytic 0.0663; MC floor 0.1467 | **0.1390** | confirmed against the MC floor |
| FD @ ε=0.05 | MC floor 0.2693 | **0.2687** | confirmed |

### The leading-order NSR prediction does not hold at this point

P0-0 predicts RMS ≈ Ω̄·r·|f′|/√3 = 0.0445 by taking F′(h_κ) ≈ f′(θ₀). The exact
setpoint contribution for this sampler is

    RMS = r · sqrt( Σ_{κ,σ} [ p_κ · Ω̄ · F′(σh_κ) / 2 ]² )

which evaluates to **0.1506** here, against a measured 0.1496. The reason is that
F′ at the probed shifts is *not* f′(θ₀): at κ=0 it is −1.42 and +1.16 against
f′(θ₀) = −0.385, and the shifts run out to h₅ = 1.73 on a landscape with
f″ = 10.1. So the 3.4× gap is the approximation, not the implementation.
Script: `scratchpad/nsr_floor_check.py` (recomputable; F′ from the same mesolve
landscape).

**This is the diagnostic P0-0 asked for, and it passes**: δ is not averaging
away. If it were being redrawn per execution, NSR would keep descending past
10⁵ instead of flattening at 0.15.

## δ off (diagnostic, not a paper figure)

The pre-δ build of the same script is commit `871e1b5`'s cache: PSR 0.0095,
NSR 0.0107, PSR+gate 0.0167, FD 0.1519, FD fixed 0.1752 at N = 10⁶ (note that
build used code-ε, so its FD numbers are at a different physical step — see
NUMBERS.md on the convention change). All three estimators moved in the
predicted direction: FD's floor is essentially unchanged (it was already
δ-limited), while PSR and NSR acquired floors where they previously had none.
