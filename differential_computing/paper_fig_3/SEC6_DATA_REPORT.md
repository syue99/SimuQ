# SEC6_DATA_REPORT — data + figure handover (6-Evaluation.tex v11; v2 2026-09-05)

2026-08-26; v2 2026-09-04/05: every P0 item of the plot-redo handover applied (F6,
F_loop, F_select/F_regimes, fig1, F_cycle, branch_anatomy). Old → new numbers in
`../NUMBERS.md`, text lines in `../TEXT_CHANGES.md`, the setpoint rule in
`../DELTA_NOISE.md`, the C_PSR factor-2 audit in `../SELECTOR_FACTOR2.md`. Emulator = QuTiP under the Appendix C.3 device model, measured
against ∇C_device (the noisy landscape). No rescale / corrected-estimator
series anywhere. Deliverables: `figs/F6.pdf`, `figs/F_loop.pdf`,
`figs/F_select.pdf`, `figs/F_scale_app.pdf` (this folder), this report.

## ⚠ Contradictions with the current text / draft (read first)

1. **D1: source compile at 10³ qubits is 59.6 s, not ~10 s.** The draft
   strip's "~10 s" does not match the cached measurement
   (`F_scale_data.json`, specialized 1D chain, n=1000, median of 3 reps:
   59.61 s on this machine). The ~10 s figure matches n≈400–500
   (n=500 → 15.2 s). Fix the strip or the prose; the measurement stands.
2. **A1 dependency: no App E number exists for s_max.** Used
   s_max = θ0 (2× coupling headroom: √2 in Rabi with J∝Ω²/Δ), decided
   with the paper owner 2026-08-26, marked PROVISIONAL like δ. v2: θ0 =
   1.757 was chosen inside the window that keeps M = 5, so A1/A2 stand.
   If App E lands a different headroom, F6 and the A1/A2/A4 numbers re-render.
3. **A4(a): the Lemma D.5 bound is ~25× loose here.** Exact truncation
   bias of NSR M=5 is 0.0148 vs bound 0.368 — the alternating (−1)^κ tail
   cancels; the bound is a triangle-inequality worst case. v2 adds a twist:
   under the per-execution setpoint draw NSR estimates the δ-blurred
   landscape, whose bias r²f‴/2 = +0.0137 cancels the truncation to
   −0.0007 at this θ0, so the plotted M=5 series shows no floor through
   10⁶. That is a coincidence of the point, not a property of the cap.
4. **C4: the TFIM star sits ON the measured crossing of the balanced
   plane** (measured log₁₀ ratio +0.01 ≈ a tie at (P=2, k=1)), not clearly
   inside the PSR region as on the old 15%-NSR plane. If 6.3 prose says
   "the running instance is a PSR case", soften to "on the boundary; its
   global-θ rewrite (also ratio ≈ 0) is likewise a near-tie".
6. **F6 v2 (2026-09-04): operating point moved 1.940 → 1.757 and the setpoint
   draw is now on every estimator.** Under a frozen draw PSR's exposure is
   |f″|r, which at 1.940 (f″ = 10) is 53% of |∇C| and inverts 6.1; the new
   point is the C″ = 0 crossing in the M = 5 window. Every A-number below
   changed; the text lines are listed in `../TEXT_CHANGES.md`.
5. **B: FD-best "enters tolerance" is metric-sensitive.** The median dips
   below 0.03 at isolated steps but never holds it for 5 consecutive
   steps → reported as "never" under the hold-5 definition (the figure's
   terminal markers use hold-5).

--------------------------------------------------------------------
## A. F6 (figs/F6.pdf) — v2

Run id: `build_F6.py` @ the commit carrying this file (2026-09-04), cache
`figures/F6_floor_amplification.json`; 100 reps/point, seeds
`default_rng(1000+s)`; N grid 10²–10⁶; T/T₂* = 0.15 (on-figure); 2q TFIM
H = θ·Z₀Z₁ + 1.0·ΣX, **θ₀ = 1.757** (the C″ = 0 crossing in the M = 5
window; was 1.940), T = 5, readout O = Z₀Z₁; estimand ∇C_device =
**−1.4488** (shot-free, δ-free gradient at nominal θ₀). **Setpoint draw
r = 0.02 on every estimator** under the per-change rule (a draw when the
programmed value changes: FD 2, PSR 1 shared, NSR one per execution —
`../DELTA_NOISE.md`). Series: PSR, NSR M=∞, PSR+gate (ε_ins: 2q 10⁻³, 1q
10⁻⁴, coherent-frac 0.5), **NSR M=5 (trunc, plotted)**, FD ε* = **0.208**
(tuned at N=10⁴, frozen; paper's θ ± ε/2 convention), FD ε = 0.05 fixed,
N^(−1/2) reference. Inset: FD V at N=10⁴ with PSR/NSR flats (0.084 / 0.093)
and × at wrong-sign ε (≥ 20% of seeds: ε ≤ 0.024 and ε ≥ 1.00). No B.6.4
curve (owner's call; closed form 0.231 vs 0.180 measured).

Derivatives at θ₀: f′ = −1.449, f″ = +0.006, f‴ = +68.3. Tail slopes
(N ≥ 10³): PSR −0.33 (floors at N ≳ 10⁵), NSR −0.51.

### A1. M at the device headroom  [\owed{M}]
- s_max = **1.757** (= θ₀; **PROVISIONAL**, see contradiction 2)
- Ω̄ = 2πK = **10.00** (K = 1.5915, θ-independent for this generator)
- M = ⌊2K·s_max − ½⌋ = **5** (largest shift used 1.728 ≤ s_max)
- Time-dilation caveat unchanged from v11.

### A2. p_fail at M=5  [\owed{p}]
- analytic excluded mass ψ′(M+3/2)/(π²/2) = **0.0337 (3.4%)** — the
  deployable number; sampler tail (24 modes) 0.0255; bound 0.0368.
  Shot inflation 1/(1−p_fail) = **1.035**.

### A3. R = ‖O_P‖
- **R = 1.0**.

### A4. Floors (RMSE at N=10⁶; % of |∇C|)
- FD ε* = 0.208: **0.180** (12.4%); shot-free MC prediction 0.197; the
  δ/ε term √2·r|f′|/ε* = 0.197 is the whole floor, truncation ε²f‴/24 = 0.123
- FD fixed ε = 0.05: **0.817** (56%); predicted 0.807
- PSR: **0.028** (1.9%) — second-order setpoint displacement
  (√3/2)|f‴|r² = 0.024 (mean r²f‴/2 = 0.014); first-order |f″|r = 0.0001
- PSR+gate: **0.024** (1.7%); exact insertion bias only **0.0037** here
  (was 0.0138 at θ₀ = 1.940) — the gate is not what floors PSR at this point
- NSR M=∞: **0.0095** (0.7%), no floor (slope −0.51)
- NSR M=5 trunc: **0.0102** (0.7%); exact truncation bias 0.0148, cancelled
  by the δ-blur (contradiction 3); rej variant 0.0092
- **PSR+gate floor < FD ε* floor: CONFIRMED, 7.5×** (not the "order of
  magnitude" 6.1 claims; NSR is 19× below FD)
- δ off (diagnostic, not a figure): FD 0.058 (ε* → 0.141), FD fixed 0.027,
  PSR 0.008, NSR 0.016 / 0.018 — δ is ¾ of FD's floor and all of PSR's.
- Handover's per-setpoint NSR rule (diagnostic): NSR floors at 0.078 (5.4%);
  FD and PSR unchanged.

### NSR@cap variants (both measured, trunc plotted)
Unchanged in construction from v11. v2: rej RMSE tail 0.0092 ≈ trunc
0.0102, at 1.035× the executions per useful shot.

--------------------------------------------------------------------
## B. F_loop (figs/F_loop.pdf) — v2 (2026-09-04)

Run id: `build_Floop_trajectory.py` @ the commit carrying this file; cache
`figures/F_loop_curves.npz` (raw iterates) + `F_loop_meta.json` +
`F_loop_trajectory.json`; 20 seeds, 50 steps, B = 6000 executions/gradient
(shift rules 4800 as accounted), T/T₂* = 0.15 on-figure. **Setpoint draw
r = 0.02 on every estimator per step** (per-change rule, `../DELTA_NOISE.md`).
**T = 2.5** (was 0.8; T₂* = 16.7), θ* = (1,1), w = 0.25, start (1.010, 0.680),
μ_soft/μ_stiff = 0.54/21.9. **Optimizer: η_t = η₀/(1 + t/20), η₀ = 0.064;
plotted/scored iterate = tail average θ̄_t** (owner's ruling: the raw fixed-step
iterate sits in a setpoint-kick noise ball of radius 0.025 that no budget
removes, see `../NUMBERS.md`). FD in the paper's θ ± ε/2 convention: best
ε = 0.1 (retrospective, grid {0.1, 0.15, 0.2, 0.3, 0.5}), too large 0.5, too
small 0.05. Y-label "‖θ̄_t − θ*‖ (median and IQR)". Two estimator bugs fixed
(PSR used one of θ₂'s two per-term program sets → ∂/∂θ₂ halved; NSR shifts and
FD probes were clipped into the box) — audit at setup: PSR 0.1%, NSR 1.5%
(14-mode truncation).

Results (median ‖θ̄_t − θ*‖; hold-5 = first step from which the median stays
inside 0.03):
- PSR: hold-5 at **3**, holds to 50; 0.012 at 50 (IQR 0.010–0.014); 90% of seeds inside; bias 0.003
- NSR: hold-5 at **4**, holds to 50; 0.011 (0.007–0.022); 90%; bias 0.007
- FD ε = 0.1 (best): hold-5 at 24; 0.024 (0.022–0.030); 75%; bias 0.004
- FD ε = 0.5 (too large): never; 0.097; 0%; **bias 0.097**
- FD ε = 0.05 (too small): never; 0.039 (0.012–0.133); 45%; **bias 0.051**

Why T moved: at T = 0.8 under the rule with fixed estimators every method
plateaus at ≈ 0.042 (fixed step; cache `figures/F_loop_*_diag_T08.*`). Scan of
T with a per-step gradient-error audit at θ* picks 2.5. Pre-fix cache:
`figures/F_loop_*_v1_prefix.*`; fixed-step T = 2.5 run: commit 8757aca.

NSR in the loop is not better than PSR: both inherit ∇C(θ+δ) through the
single residual measurement. Contradiction 5 (metric sensitivity of FD's
"enters tolerance") still applies to the oracle step.

--------------------------------------------------------------------
## C. F_select (figs/F_select.pdf) — v2 replot (2026-09-05)  [\owed{balanced-plane run}]

Same cache (`figures/F_select_balanced_data.json`, 350 cells × 6 seeds), no sweep
rerun. Colour is now App G.1's cell value, mean over seeds of log₁₀(N_NSR/N_PSR)
(green NSR wins, blue PSR wins, ±0.8), colorbar "log₁₀(N_NSR / N_PSR)"; solid
black = measured crossing, dashed orange = the compiler's selector (Ω̄_AC with
margin γ(q) = min(1, 1.86/√q), from `F_regimes_data.json`, whose general-family
arrays equal this plane's to 2e-16). Fig 14 uses the identical fill and styles.

### C1. Family / ranges — unchanged (7q device alphabet, 35 terms; p 1–10, q 1–35).
### C2. NSR share — **42.3%** (unchanged).
### C3. Selector (G.3) — agreement 88.3%, forfeit median 1.07× / max 1.26× over
the disagreeing cells, ties 41 (L1 exact ties) → 6 (after margin), NSR chosen on
46.0%. Table 4's L1-certificate row (5.76× max, 1.35× median divergent, 36.0%
divergent) unchanged.
### C4. Markers — star (2,1): measured 10^+0.01 (PSR side, near tie). **Circle moved
to (1,3)**: the global-θ rewrite θ·(Z₀Z₁+X₀+X₁) is one coefficient over three
terms; on the plane's own cost model it measures 10^−0.24 (NSR side; the L1
certificate calls it a tie). Contradiction 4 updated accordingly: the instance
is on the boundary, its rewrite is an NSR case. `../SELECTOR_FACTOR2.md` shows
the factor 2 in C_PSR (`build_F_select.py:138`) and the near-tie arithmetic.

--------------------------------------------------------------------
## D. Compile timing (tab:strategies; curve in figs/F_scale_app.pdf)

Run id: `sec6_compile_timing.py` + `build_F_scale.py` cache
`figures/F_scale_data.json` (resumable, medians of 3 compile reps /
10 branch reps) and `figures/sec6_compile_timing.json`; wall-clock
`time.perf_counter`; specialized path, 1D chain, T=1.0, x=0.8, tol=0.1.

Wall-times (specialized path, medians):

| n    | source (s) | +PSR/branch (ms) | +NSR/branch (ms) |
|------|-----------|------------------|------------------|
| 10   | 0.02      | 0.5              | 0.0023           |
| 30   | 0.08      | 1.8              | 0.0058           |
| 100  | 0.63      | 6.5              | 0.017            |
| 300  | 5.65      | 27.0             | 0.053            |
| 1000 | 59.61     | 175.0            | 0.191            |

### D1. Source compile at 10³
**59.6 s** — the draft strip's "~10 s" is WRONG (contradiction 1).

### D2. +PSR per branch at 10³
**175 ms = 0.29% of source** — draft's 0.3% VERIFIED.

### D3. +NSR per branch at 10³
**0.191 ms = 10^−3.7 s** — consistent with the draft's ~10^−3.5 s
(slightly cheaper).

### D4. FD per branch  [\owed{FD}]
**An FD branch re-runs the source solve.** The pipeline has no FD path;
a black-box FD branch is a full specialized compile at the shifted value
x+ε. Measured at n=300: **5.68 s vs 5.64 s source = 101% of source** —
the cell in tab:strategies is ≈100% of source, and that is a real
finding: an FD gradient at 10³ qubits pays ~2×59.6 s of compile per
component per step. The only reuse path is the specializer's closed-form
coefficient table (measured 0.061 ms at n=300) — i.e. FD becomes cheap
only by adopting the differentiation infrastructure's own shift-table
machinery, at which point its compile cost equals NSR's.

### D5. P/k scan at n=300  [\owed{P/k scan}]
Per-branch increment, medians (6 branches PSR / 50 reps NSR table):

PSR per branch (ms):

| P \\ k | 1    | 4    | 14   |
|-------|------|------|------|
| 1     | 27.7 | 27.7 | 27.1 |
| 5     | 27.1 | 26.7 | 27.6 |
| 20    | 27.2 | 27.5 | 27.7 |

NSR per branch, full channel-table emission (ms):

| P \\ k | 1     | 4     | 14    |
|-------|-------|-------|-------|
| 1     | 0.055 | 0.053 | 0.054 |
| 5     | 0.054 | 0.054 | 0.053 |
| 20    | 0.054 | 0.054 | 0.054 |

Trend: **both strategies are flat in P and k at fixed n** — the PSR
branch is dominated by the mapper walk over the n-site schedule (the
k ≤ 14 kick bonds are invisible against it), and the NSR branch is
dominated by the O(n) per-channel table emission. The k-dependence the
spec warned about is real but sits below the emission floor: the
k-scoped arithmetic update alone scales linearly in k (0.55 μs at k=1 →
2.5 μs at k=14) and is ~20× below the full-table cost at n=300. Total
gradient cost still scales with P through the branch COUNT (2mP for PSR,
2NP for NSR); the per-branch increment does not.

### D6. Appendix figure
`figs/F_scale_app.pdf`: (top) log–log source / +PSR / +NSR vs n
(specialized path stated on-figure); (bottom) D5 panel — PSR per-branch
vs P at k ∈ {1,4,14} (orange ramp, flat and overlapping) over the NSR
all-cells line.

--------------------------------------------------------------------
## E. Checks

### E1. Regime audit
- F6: T/T₂* = **0.15**, stamped on-figure. ✓
- F_loop: T/T₂* = **0.15**, stamped on-figure. ✓
- F_select: Hamiltonian level, **no noise** (no time-evolution regime;
  stated on-figure).
- F_scale_app: compile timing only, no emulation ("specialized path, 1D
  chain" stated on-figure).

### E2. PSR pair shares one transport plan?
**No.** Each branch is mapped independently (`TweezerMapper.map_hlist`
produces a per-branch `TransportLog`). The ± branches of a pair have
identical-content plans — same frozen geometry and dressing pairs from
the specializer plan, same kick pair brought together; only the pulse
sign differs — but the pipeline does not deduplicate or share the plan
object between them. (Cost impact is already inside the measured
per-branch numbers.)

### E3. Differentiated θ per program; single-qubit insertion?
- F6: θ = TFIM coupling (generator Z₀Z₁) — two-qubit insertion only.
- F_loop: θ₁ (Z₀Z₁, two-qubit) **and θ₂ (X₀+X₁ — single-qubit
  insertions; this is the one compiled program that exercises them).**
- F_select: Hamiltonian level, no compilation (P random ±1-weighted
  parameters over the 35-term alphabet).
- F_scale / timing: global x over all ZZ bonds (D5 scan: th0 of P
  disjoint k-bond groups).

### E4. NSR^M rejection audit
Confirmed in code (`nsr_rej_est`, build_F6.py): out-of-range draws are
rejected, **never resampled** — they consume budget and contribute 0
with the L1 weight unchanged. The (a) variant renormalises at compile
time instead; the two agree at the shared truncated target (§A above).

### E6. One compile for Fig 7 and Fig 13 (v2, P0-2)
`figures/branch_anatomy_data.json` is now extracted with the 5 µm transit lane
(`transit_dy = 5`, owner's ruling: the lane is the real schedule), the same
mapper configuration as `F_waveform_meta.json`; both give the nine-segment
schedule 119.834 µs (τ 2.085 · lift 2.344 · travel 52.381 · drop 2.344 · CZ
0.696 · lift · travel · drop · 2.915), transport 114.14 µs = 95.2%, NSR lane
5.000 µs. Fig 7 is written to `figures/branch_anatomy.{pdf,png}` only. The
direct-move extraction (110.46 µs) is kept as `branch_anatomy_data_v1_direct.json`.

### E7. Other v2 figure fixes
- fig1 (paper Fig 2): setpoint draws on the secants, analytic cone, C_device
  notation, no "T4", illustration-only (no numeric annotations); numbers in
  `../NUMBERS.md`. Written to `figures/`, `paper_fig_2/`, `figs/`.
- F_cycle (paper Fig 12): "(see fig:schedules)" → "(Figure 13)"; the 1–10 ms
  window label is an open question for the text owner.
- compile_curves (paper Fig 15b): unchanged; 916× is unrounded, Table 6 should
  print 0.191 ms.

### E5. Run ids for tab:strategies rows
- "compile per branch" row (source / +PSR / +NSR / FD): cache
  `figures/F_scale_data.json` (n=10,100,500,1000 from the original
  F-scale run; n=30,300 appended 2026-08-26 by `sec6_compile_timing.py`)
  + `figures/sec6_compile_timing.json` (D4 FD, this machine, 2026-08-26).
- "scaling with P, k" row: `figures/sec6_compile_timing.json` → key
  `D5` (n=300, 2026-08-26).
- Both scripts in `differential_computing/tests/`; commit recorded in
  git history alongside this report.

--------------------------------------------------------------------
## Deliverable checklist
Format (v2, 2026-09-05): F6, F_loop, F_select and fig1 are written to
`paper_fig_2/` and `figs/`; F_select's colour is the log₁₀(N_NSR/N_PSR)
diverging map (the executions-to-target gray fill is gone), thin black
measured crossing, orange dashed selector; F_regimes uses the same styles.
The F6 legend is 5.8 pt (below the ground-rule 7 pt minimum, per owner).

- [x] figs/F6.pdf (+png) — v2: θ₀ = 1.757, δ on every estimator
- [x] figs/F_loop.pdf (+png) — v2: T = 2.5, SGD schedule, tail-averaged iterate
- [x] figs/F_select.pdf (+png), figs/F_regimes.pdf (+png) — v2 log-ratio plane
- [x] figs/fig1_intro_trap.pdf (+png), figs/F_cycle.pdf (+png), figs/F_waveform.pdf (+png)
- [x] figures/branch_anatomy.pdf (+png) — v2 transit-lane compile
- [x] figs/F_lowering.pdf (+png) — Fig 6 redrawn (strip dropped, ≥ 7 pt, no internal names)
- [x] figs/F_scale_app.pdf (+png), figs/compile_curves.pdf (+png) — unchanged
- [x] SEC6_DATA_REPORT.md (this file); ../NUMBERS.md, ../TEXT_CHANGES.md,
      ../DELTA_NOISE.md, ../SELECTOR_FACTOR2.md
- [x] figs/F_epssweep.pdf (+png) — P1-1, shot-free bias floors at T = 1 / 2.5 / 5 µs
