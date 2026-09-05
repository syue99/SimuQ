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

## Figure 9 — `tests/build_Floop_trajectory.py`, run 2026-09-04 (this commit)

Cache: `figures/F_loop_curves.npz` (raw iterates), `F_loop_meta.json`, `F_loop_trajectory.json`.
Figure: `paper_fig_3/figs/F_loop.{pdf,png}` (copy + two-panel `F_loop_full` in `paper_fig_2/`).
Setpoint rule: per change (DELTA_NOISE.md). Two estimator bugs fixed, the landscape (T) and
the optimizer (decaying step, tail-averaged iterate) changed — all owner's rulings, see
"Deviations". Pre-fix cache kept as `figures/F_loop_*_v1_prefix.*`.

### Unchanged, confirmed

| quantity | value |
|---|---|
| θ* | (1, 1) |
| box | [0.2, 1.4]² (constrains the iterate only; programs are never clipped) |
| w | 0.25 |
| seeds | 20 |
| τ-samples (PSR) | 32 |
| executions per gradient | 6000 for FD; 4800 for the shift rules as accounted (see note) |
| tolerance | 0.03 (= 1.5 r) |
| T/T₂* | 0.15 |
| r | 0.02 |

### Moved — text edits needed

| quantity | old (text) | new | note |
|---|---|---|---|
| T, T₂* | 0.8 µs, 5.33 µs | **2.5 µs, 16.7 µs** | landscape scan (below) |
| start | (0.802, 1.251) | **(1.010, 0.680)** | auto: 0.32 from θ*, mostly along the stiff axis |
| steps run / drawn | 100 / 50 | **50 / 50** | |
| step size | η = 0.403 fixed | **η_t = η₀/(1 + t/20), η₀ = 0.064** (= 1.4/μ_stiff) | SGD schedule |
| reported iterate | θ_t | **θ̄_t = mean(θ_s, s ∈ [⌈t/2⌉, t])** (tail average) | y-label ‖θ̄_t − θ*‖ |
| μ_soft, μ_stiff, κ | 0.31, 3.48, 11 | **0.54, 21.9, 41** | |
| C‴ along soft | 1.24 | **18.1** | |
| a*, b* | 0.221, 0.388 | **0.597, −1.331** | |
| FD arms (paper convention θ ± ε/2) | 0.15 / 0.7 / 0.04 (builder's θ ± ε ÷ 2ε = 0.3 / 1.4 / 0.08) | **0.1 (best) / 0.5 (too large) / 0.05 (too small)** | oracle grid {0.1, 0.15, 0.2, 0.3, 0.5} → 0.1 |
| converged (median of ‖θ̄_t − θ*‖ holds 0.03 for 5 steps, first such step) | PSR 10, NSR 34 | **PSR 3, NSR 4**; both then hold to step 50 | |
| FD | never | oracle ε = 0.1: **holds from step 24** (0.024 at 50); ε = 0.5: **never** (0.097); ε = 0.05: **never** (0.039) | |
| median ‖θ̄₅₀ − θ*‖ | — | PSR **0.012**, NSR **0.011**, FD 0.1: 0.024, FD 0.5: 0.097, FD 0.05: 0.039 | raw iterates: 0.014 / 0.017 / 0.022 / 0.096 / 0.030 |
| IQR at 50 | — | [0.010, 0.014] / [0.007, 0.022] / [0.022, 0.030] / [0.095, 0.099] / [0.012, 0.133] | |
| seeds inside tolerance at 50 | 0.80 / 0.80 / 0.35 / 0 / 0.25 | **0.90 / 0.90 / 0.75 / 0 / 0.45** | |
| mean offset of θ̄₅₀ from θ* (bias) | — | PSR 0.003, NSR 0.007, FD 0.1: 0.004, FD 0.5: **0.097**, FD 0.05: **0.051** | the two deployable steps are biased; averaging cannot help them |
| C(θ̄₅₀) − C* median | — | 8·10⁻⁵ / 2.3·10⁻⁴ / 4.1·10⁻⁴ / 0.060 / 4.0·10⁻⁴ | |

### Estimator audit (shot-free, δ-free; printed at setup, asserted < 3%)
PSR reproduces ∇⟨O⟩ to 0.1% on both coefficients (was 50% low on θ₂). NSR to 1.5% (the
14-mode series truncation of the sampler; was 30–100% off with clipped shifts).

### Budget note
`iqs_grad` charges the residual measurement twice (`ngrad = B − 2·n_res`) although one
program yields both observables, so the shift rules use 4800 executions per gradient
against FD's 6000. Pre-existing; conservative for the shift rules; not changed.

### Why the raw fixed-step loop stalls (the reason for the schedule and the average)
At T = 2.5 with a fixed η = 1.4/μ_stiff, the raw iterate of an unbiased estimator settles
in a noise ball of median radius 0.025 (PSR) / 0.029 (NSR), of which the shared setpoint
draw contributes 0.017 (η·H·δ along the stiff axis, r·√(ημ_stiff/2)), shot jitter
0.008 ⊕ 0.006, bias ≤ 0.011 (not significant). Neither shots nor bias, so a budget change
does not fix it; the SGD remedies do, and only for unbiased estimators: with δ off the
floor is 0.016. (Cache of that fixed-step run: commit 8757aca.)

### Deviations from the handover, for Fred
1. **T moved 0.8 → 2.5** (T/T₂* unchanged). Under the rule with the fixed estimators,
   T = 0.8 gives PSR 0.042, NSR 0.044, FD-best 0.038, FD-small 0.042 at step 50 (20 seeds;
   cache `figures/F_loop_*_diag_T08.*`) — shot-noise jitter above tolerance for every
   method, nothing separates. The audit over T ∈ {0.8, 1.2, 1.6, 2.0, 2.5, 3.0} picks 2.5
   (shift rules 0.02, FD's best 0.05 per step); 2.0 is degenerate (κ = 500), 3.0 biases
   PSR's τ-quadrature at M = 32.
2. **Decaying step and tail-averaged iterate** (§5 "no optimizer hyperparameters"): the
   raw fixed-step iterate cannot converge to θ* under per-step noise (noise ball above),
   and the paper's claim is convergence to θ*. Both are textbook SGD; both leave a biased
   estimator's offset untouched, which is exactly what separates FD's deployable steps.
3. **Two estimator bugs fixed**: PSR read only the first of the two per-term program sets
   for θ₂ (X₀+X₁), halving that gradient component; NSR's Nyquist shifts (±0.98, ±2.9, …
   in θ₁ at T = 0.8) and FD's probes were clipped into the plotting box. The old
   "34 steps" for NSR is that artefact.
4. FD ε labels moved to the paper's convention.

## Figure 2 — `tests/build_fig1.py`, run 2026-09-05 (this commit)

Cache: `figures/fig1_intro_data.json` (version 2). Figure: `figures/fig1_intro_trap.{pdf,png}`,
copies in `paper_fig_2/` and `paper_fig_3/figs/`. Hamiltonian-level, T/T₂* = 0.5 (kept by ruling).

### Unchanged, confirmed

| quantity | value |
|---|---|
| H, observable | θZ₀ + X₀, ⟨Z₀⟩ |
| T, T₂*, regime | 12 µs, 24 µs, 0.5 |
| anchor θ* | 1.290 (the R8 sweep re-selects it; 41 configs pass) |
| ∇C_device(θ*) | −4.347 (61% of the window's max slope 7.13) |
| physical probes | θ* ± 0.18, ± 0.25, ± 0.32 |
| r | 0.02 |

### Moved — text edits needed

| quantity | old (text) | new | note |
|---|---|---|---|
| secant ε labels | 0.18 / 0.25 / 0.32 | **0.36 / 0.50 / 0.64** | same probes, paper's θ ± ε/2 convention (as Figs 8, 9) |
| drawn secant slopes | +0.77 / +1.44 / +0.65 (δ-free) | **+0.70 / +1.39 / +0.53** (seed-0 δ draws per probe, P0-0) | wrong-signed in 97% / 100% / 100% of 2000 draws |
| purple cone | shot-noise fan at ε = 0.03 (builder conv.), ±1.8σ, N = 4000 | **analytic setpoint cone S(ε) = √2·r\|f′\|/ε at ε = √2·r = 0.028**, ±1σ = \|∇C\| | δ only, no shots; 15% wrong sign at that step (measured) |
| step-floor marker | 0.04 wide, "ε ≳ δ" | **removed** (owner: illustration only) | |
| title strip | "Hamiltonian-level, T4 noise" | "Hamiltonian-level" (owner: no ε*/RMSE line and no step-floor bracket on the figure; those numbers are for the caption) | |
| notation | C_noisy, ∇C_noisy | **C_device, ∇C_device** | v11 name, as Fig 8 |

### B.6.4 at the anchor (for the caption)

| quantity | value |
|---|---|
| f′, f″, f‴ | −4.347, −2.18, +1645 |
| ε*_analytic = (24\|f′\|r/\|f‴\|)^{1/3} | **0.108** |
| B.6.4 floor 0.60\|f‴\|^{1/3}(\|f′\|r)^{2/3} | 1.39 = **32% of \|f′\|** |
| shot-free Monte-Carlo FD floor (2000 draws per ε, 25 steps in [0.02, 0.8]) | best ε = **0.108**, RMSE = **28% of \|f′\|** (this is the annotated number) |
| common-mode term \|f″\|r/√2 | 0.03 — negligible here (near the steepest point), so B.6.4 and MC agree to 15% |
| cone step √2·r | 0.028 |

The old sidecar's "usable window [0.026, 0.072]" and "best ε = 0.056 (28%)" were at N = 4000
with shot noise in the builder's convention; the new numbers are shot-free in the paper's.

## Figures 10 and 14 — `tests/build_F_select_balanced.py`, `tests/build_F_regimes.py --replot`, 2026-09-05 (this commit)

Replots from the existing caches (`figures/F_select_balanced_data.json`,
`figures/F_regimes_data.json`); no sweep rerun. Fig 14(a) arrays == Fig 10 arrays (max |ΔZ|
= 2.2e-16, asserted by the builder).

### Unchanged, confirmed (all reproduce the text exactly)

| quantity | value |
|---|---|
| plane | p ∈ [1,10], q ∈ [1,35], 350 cells, 6 seeds, 11,550 tangents (7q TFIM device alphabet, 35 terms) |
| NSR measured share (general) | **42.3%** |
| selector agreement L1 / AC / AC+margin (general) | 57.7 / 71.4 / **88.3%** |
| forfeit, general (AC+margin, over disagreeing cells) | median **1.07×**, max **1.26×** |
| ties | **41** (L1 certificate exact ties) → **6** (after AC + margin) |
| NSR chosen by the selector (general) | 46.0% |
| aligned family | share 27.7%, agreement 92.6%, max forfeit 1.39×, NSR chosen on 20.9% (margin gain 72.3% → 92.6%) |
| Heisenberg family | share 94.3%, agreement 96.0%, max forfeit 2.17× |
| γ₀ | 1.86 |
| Table 4 row (L1 certificate, published builder) | forfeit max 5.76×, median over disagreeing cells 1.35×, divergent on 36.0% |
| star (2,1) | measured 10^+0.012 (PSR side, near tie) |

### Moved — figure changes and text edits

| item | old | new |
|---|---|---|
| Fig 10 colour | absolute "executions to target (best strategy)", 10¹–10⁴ gray bands + winner washes | **log₁₀(N_NSR/N_PSR)**, mean over seeds, diverging green (NSR wins) / white / blue (PSR wins), limits ±0.8 (data −0.76 … +0.28); colorbar "log₁₀(N_NSR / N_PSR)". Target-free, as G.1 says. |
| Fig 10 overlays | solid black measured crossing only (no certificate line: the L1 certificate never crosses this plane) | solid black = measured crossing; **dashed orange = compiler's selector** Ω̄_AC + margin γ(q) = min(1, 1.86/√q) (G.3.1), from Fig 14's cache |
| Fig 14 style | two-tone winner fill, green dashed selector | same diverging fill and colorbar as Fig 10, black solid (1.0 pt) / orange dashed |
| circle "global-θ rewrite" | (1, 2) | **(1, 3)** — θ·(Z₀Z₁+X₀+X₁) has one coefficient over three terms. Its measured ratio on the plane's model is **10^−0.24 (NSR side)**; the plane cell (1,3) averages 10^−0.02 |
| caption "both sit on the crossing at (2,1)" | | true for the star (+0.02); the rewrite is on the NSR side (see SELECTOR_FACTOR2.md) |
| caption "gold line = compiler's choice" | | the selector line is orange dashed in both figures ("gold" in the caption is acceptable, or say "orange dashed") |

Consistency checks: 3 (factor 2, near-tie at (2,1)) — holds, SELECTOR_FACTOR2.md; 4
(Fig 14(a) == Fig 10) — holds.

## Figures 7 and 13 — `tests/build_branch_anatomy.py` (REBUILD=1), `tests/build_F_waveform.py` (unchanged), 2026-09-05

Owner's ruling: the 5 µm transit lane is the real schedule. Fig 7 is now extracted from
the same mapper configuration as Fig 13 (`transit_dy = 5 µm`, `TweezerMapper._transit_legs`:
lift → travel → drop per relocation). Both figures now come from the same nine-segment
schedule; the previous Fig 7 compile (direct move) is kept as
`figures/branch_anatomy_data_v1_direct.json`.

| quantity | old (Fig 7 / Table 3 / App E) | new (= Fig 13 / Table 5) |
|---|---|---|
| segments | 5 | **9** (ev · lift · travel · drop · CZ · lift · travel · drop · ev) |
| τ | 2.085 µs | 2.085 µs |
| move legs | 52.4 / 52.3 µs | **57.07 / 57.07 µs** each = 2.344 (lift, +5 µm) + 52.381 (travel, 100 µm) + 2.344 (drop) |
| CZ | 696 ns | 696 ns |
| total | 110.5 = 5.0 ev + 104.8 move + 0.70 gate | **119.83 = 5.0 ev + 114.14 move + 0.70 gate** (transport 95.2%) |
| Fig 7 label "AOD moves pair (52 µs)" | | "(57.1 µs)" |
| R_CZ | 2.5 µm | 2.5 µm |
| NSR branch (Fig 13) | 5.000 µs | 5.000 µs |

Fig 13 itself is unchanged (its compile already had the lane); its caption's "same
schedule, different accounting" must go — see TEXT_CHANGES.md.

## Figure 12 — `tests/build_F_cycle.py`, 2026-09-05
"the window schedule (see fig:schedules)" → "the window schedule (Figure 13)". Box label
"~1–10 ms" for the operation window kept **pending Fred's answer** (App F says the window is
"at most hundreds of microseconds"; the branch here is 120 µs, so a 1–10 ms operation phase
means many branches per cycle, or the label is wrong).

## Figure 15(b) / Table 6 — `tests/compile_curves.py` (no change), checks from `figures/sec6_compile_timing.json`

| quantity | value |
|---|---|
| n | 10 / 30 / 100 / 300 / 1000 |
| source compile (s) | 0.0165 / 0.0751 / 0.634 / 5.654 / 59.61 |
| PSR branch (ms) | 0.544 / 1.760 / 6.483 / 27.03 / 174.97 |
| NSR branch (ms) | 0.00229 / 0.00579 / 0.01733 / 0.05323 / 0.19106 |
| ratio at n = 1000, unrounded | **915.7×** (the figure prints "916×"); Table 6's rounded 175 / 0.19 = 921 |
| check 1 | 174.97 ms / 59.61 s = **0.293%** (6.3's "0.3%" ✓); 0.191 ms / 59.61 s = **3.2·10⁻⁴ %** ✓ |
| check 5, exponents (log-log fit, all 5 points) | source **1.80**, PSR **1.24**, NSR **0.96** (text 1.8 / 1.3 / 1.0 ✓ at one decimal: PSR 1.24 rounds to 1.2, not 1.3; n ≥ 100 fits give 1.97 / 1.43 / 1.04) |
| platform string | `macOS-26.5-arm64-arm-64bit` is real: `platform.platform()` recorded in `F_scale_data.json` meta |

Table 6 fix so the annotation and the table agree at displayed precision: print NSR at
n = 1000 as **0.191 ms** (then 175 / 0.191 = 916) — or quote the ratio as "≈ 9×10²".

## Figure 6 (lowering stack) — no builder in this repo
The figure is authored in the paper source (TikZ/tex). Handover default applied: **drop the
instance strip**, let Figs 7 and 13 carry the instance; Fred confirms.

## Figure 11 (P1-3) — attribution
Not a plot task. The figure file needs the "reproduced from [ref] (CC BY 4.0)" text once
the bib entry exists; if it is the lab's own apparatus figure it may need replacing with a
schematic for anonymity. Fred's call.
