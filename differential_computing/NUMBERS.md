# NUMBERS.md — old → new, per figure

Old = what the paper text says (handover §3 registry) unless marked otherwise.
Script and commit for each entry are given per section. Nothing here edits the paper.

## Figure 8 — `tests/build_F6.py`, run 2026-09-04 (this commit)

Cache: `figures/F6_floor_amplification.json`. Figure: `paper_fig_3/figs/F6.{pdf,png}`
(copy in `paper_fig_2/`). Setpoint rule: **per change** (owner's ruling, see
DELTA_NOISE.md). Operating point changed — see "Deviations" at the end of this section.

### Unchanged, confirmed

| quantity | value |
|---|---|
| Ω̄ = 2T | 10.0 |
| T, T₂* (T/T₂* = 0.15) | 5.0 µs, 33.3 µs |
| r (setpoint std) | 0.02 |
| ε_ins (PSR+gate series) | 10⁻³ (2q), 10⁻⁴ (1q) |
| M cap | 5 |
| excluded mass p_out | 0.0337 → **3.4%** (analytic); shot inflation 1.035 |
| N grid, seeds | 10²…10⁶ (9 points), 100 reps/point |
| inset grid | 24 points, geometric over [0.02, 1.2], N = 10⁴, 100 seeds |
| FD fixed step | ε = 0.05 |
| PSR branches | 2m = 96 (m = 48; τ-quadrature bias 0.0016 at m = 48) |

### Moved — text edits needed

| quantity | old (text) | new | note |
|---|---|---|---|
| θ₀ | **1.940** | **1.757** | C″ = 0 crossing in the M = 5 window [1.728, 2.042) |
| ∇C_device(θ₀) | −0.385 | **−1.449** | |
| C″(θ₀) | +10.14 (cache) | **+0.006** | by construction |
| C‴(θ₀) | +22.7 (cache) | **+68.3** (h = 0.05; +65.2 at h = 0.08) | |
| FD ε* (tuned at N = 10⁴) | **0.17** | **0.208** | paper convention θ ± ε/2, ÷ε; B.6.4 gives ε*_analytic = 0.217 |
| PSR tail slope | N^−0.48 | **N^−0.33** (R² 0.94) | bends: PSR reaches its floor inside the window (N ≳ 10⁵) |
| NSR M=∞ tail slope | N^−0.50 | **N^−0.51** (R² 0.997) | no floor through 10⁶ |
| PSR RMSE at N = 10⁶ | — | **0.0282** (1.9% of \|f′\|) | floor = second-order setpoint displacement, see below |
| PSR+gate RMSE at 10⁶ | 0.0167 (δ-off cache) | **0.0242** (1.7%) | same floor; the gate bias here is only 0.0037 |
| PSR exact gate bias | 0.0138 (cache, θ₀=1.94) | **0.0037** | operating-point dependent |
| NSR M=∞ RMSE at 10⁶ | 0.0107 (δ-off cache) | **0.0095** (0.7%) | no setpoint floor under the per-change rule |
| NSR M=5 RMSE at 10⁶ | 0.0159 (δ-off cache) | **0.0102** (0.7%) | truncation −0.0148 cancelled by δ-blur +0.0137 (coincidence, see below) |
| NSR M=5 rejection variant at 10⁶ | — | 0.0092 | reported, not plotted |
| FD @ ε* RMSE at 10⁶ | 0.152 (δ-off cache) | **0.180** (12.4%) | MC-predicted shot-free floor 0.197 |
| FD @ ε=0.05 RMSE at 10⁶ | 0.175 (δ-off cache) | **0.817** (56%) | √2·r\|f′\|/ε = 0.82 |
| PSR+gate vs FD ε* at 10⁶ | "an order of magnitude below" | **7.5×** below (PSR plain 6.4×) | |
| NSR vs FD ε* at 10⁶ | — | **19×** below (M=∞), 18× (M=5) | |
| FD-beats-both crossover | — | none in grid (FD ≥ min(PSR,NSR) from N = 100) | |
| usable-ε window (RMSE/\|f′\| < 0.5, sign err < 5%) | [0.169, 0.412] (cache) | **[0.058, 0.493]** | |
| wrong-sign FD steps (≥ 20% of seeds) | — | ε ≤ 0.024 and ε ≥ 1.00 | inset × marks |
| PSR / NSR flat lines in inset (N = 10⁴) | — | 0.084 / 0.093 | |

### P0-0 expected consequences — computed and confirmed at θ₀ = 1.757

| quantity | predicted | measured |
|---|---|---|
| FD setpoint term √2·r\|f′\|/ε at ε* = 0.208 | 0.197 | FD floor 0.180 (MC shot-free 0.197) |
| FD truncation ε*²\|f‴\|/24 | 0.123 | |
| B.6.4 closed form 0.60\|f‴\|^{1/3}(\|f′\|r)^{2/3} | **0.231** | vs 0.180 measured: the Taylor truncation term overestimates (ε*·ω ≈ 1.6 for this landscape's ~7.7 rad/unit tone) |
| FD common-mode displacement \|f″\|r/√2 | 0.0001 | vanishes at C″ = 0 (this is why B.6.4 is now the right form, if not the right value) |
| PSR first-order displacement \|f″\|·r | **0.0001 (0.0% of \|f′\|)** | not the floor |
| PSR second-order displacement, RMS (√3/2)\|f‴\|r² | **0.0237** | PSR+gate 0.0242, PSR 0.0282 (100-rep scatter) |
| PSR second-order displacement, mean bias r²f‴/2 | 0.0137 | |
| NSR setpoint floor (per-change rule) | none (averages as N^−1/2) | 0.0095 at 10⁶, slope −0.51 |
| NSR δ-blur bias r²f‴/2 (per-execution δ smooths the landscape) | +0.0137 | cancels the −0.0148 truncation bias of M = 5 to −0.0007; cancels the simulator's 24-mode sampler truncation −0.0120 to +0.0021 for "M=∞" |
| handover's NSR floor Ω̄·r\|f′\|/√3 (per-setpoint rule) | 0.167 | not applicable under the ruling; measured in the per_value diagnostic (DELTA_NOISE.md) |
| cone lower bound ε = √2·r | 0.028 | |

**Caveat for Fred (NSR "no floor").** Under the per-change rule NSR's only residual is the
δ-blur bias r²f‴/2 = 0.014 (0.9% of |f′|), the same second-order term PSR carries as a mean.
At N = 10⁶ shot noise is 0.0095, so a true M=∞ estimator would show sqrt(0.014² + 0.0095²)
≈ 0.017 at the last point, not 0.0095; the simulator's 24-mode sampler hides it because its
own truncation (−0.012) happens to cancel the blur at this θ₀. NSR still beats FD by ≥ 10×
and PSR by ~1.5× at 10⁶ either way. If Fred wants the last point strictly faithful, the
sampler's MAXN can be raised (a rerun, ~30 min); I did not do this unasked.

### Consistency checks (handover §3)

2. PSR+gate floor ≥ 10× below FD ε* floor at N = 10⁶? **No: 7.5×.** NSR is 19× below.
6. δ audit: δ on → FD floors (0.180), NSR does not floor (slope −0.51 to 10⁶), PSR floors
   at its second-order displacement (0.024–0.028), not at |f″|r (≈ 0). δ off and the
   handover's per-setpoint rule: diagnostic runs, reported in DELTA_NOISE.md.

### Deviations from the handover, for Fred

1. **Operating point moved 1.940 → 1.757** (handover §5: "no change to operating points").
   Reason: under any frozen-δ rule PSR's exposure is |f″|r, and at 1.940 (f″ = 10.1) that is
   53% of |f′| — PSR floors above FD, inverting §6.1. C.3's selection rule ("the point
   maximizing FD's predicted floor") selects for large f″. New rule: the stationary point of
   ∇C (C″ = 0) inside the window that keeps M = 5, so Ω̄, T, T₂*, M, p_out, ε_ins, N grid,
   seeds are all unchanged. Steepness (longer T) is not a lever: ε* only shrinks as T^{−2/3}
   while NSR's and PSR's second-order exposures grow with T (checked at T = 10).
2. **NSR setpoint rule** differs from P0-0's "one draw per distinct (κ,σ)": owner's ruling,
   DELTA_NOISE.md. The handover's "if NSR does not floor, that is a bug" clause is overridden.
3. **No B.6.4 curve in the inset** (P0-5): owner's call; the closed form is 0.231 against
   0.180 measured even at C″ = 0, so drawing it would misstate agreement.
4. ε on the figure is in the paper's θ ± ε/2, ÷ε convention (was θ ± ε, ÷2ε in earlier
   builds), so the text's ε* = 0.17 and any older ε label are not comparable.
