# SEC6_DATA_REPORT — data + figure handover (6-Evaluation.tex v11)

2026-08-26. Emulator = QuTiP under the Appendix C.3 device model, measured
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
   with the paper owner 2026-08-26, marked PROVISIONAL like δ. If App E
   lands a different headroom, F6 and the A1/A2/A4 numbers re-render.
3. **A4(a): the Lemma D.5 bound is ~30× loose here.** Measured NSR^M_trunc
   floor 0.0123 vs bound 0.368 — the alternating (−1)^κ tail cancels; the
   bound is a triangle-inequality worst case. Safe direction (bound ≥
   measured), but the text should not imply the series floors AT the bound.
4. **C4: the TFIM star sits ON the measured crossing of the balanced
   plane** (measured log₁₀ ratio +0.01 ≈ a tie at (P=2, k=1)), not clearly
   inside the PSR region as on the old 15%-NSR plane. If 6.3 prose says
   "the running instance is a PSR case", soften to "on the boundary; its
   global-θ rewrite (also ratio ≈ 0) is likewise a near-tie".
5. **B: FD-best "enters tolerance" is metric-sensitive.** The median dips
   below 0.03 at isolated steps but never holds it for 5 consecutive
   steps → reported as "never" under the hold-5 definition (the figure's
   terminal markers use hold-5).

--------------------------------------------------------------------
## A. F6 (figs/F6.pdf)

Run id: `build_F6.py` @ commit f915c86, cache
`figures/F6_floor_amplification.json`; 100 reps/point, seeds
`default_rng(1000+s)`; N grid 10²–10⁶; T/T₂* = 0.15 (on-figure); 2q TFIM
H = θ·Z₀Z₁ + 1.0·ΣX, θ₀ = 1.940, T = 5, readout O = Z₀Z₁; estimand
∇C_noisy = −0.3850. Series: PSR, NSR M=∞, PSR+gate (ε_ins: 2q 10⁻³, 1q
10⁻⁴, coherent-frac 0.5, T4), **NSR M=5 (trunc, plotted)**, FD ε*=0.17
(retrospective sweep at N=10⁴, frozen), FD ε=0.05 fixed, N^(−1/2)
reference. Inset: FD V at N=10⁴ with PSR/NSR flats and × at wrong-sign ε
(as v9).

### A1. M at the device headroom  [\owed{M}]
- s_max = **1.940** (= θ₀; 2× coupling headroom from √2 Rabi, J∝Ω²/Δ —
  **PROVISIONAL**, no App E number; see contradiction 2)
- Ω̄ = 2πK = **10.00** (compiler certificate for θ on this instance,
  K = 1.5915)
- M = ⌊(2Ω̄s_max/π − 1)/2⌋ = **5** (largest shift used
  (M+½)/(2K) = 1.728 ≤ s_max)
- Time-dilation caveat (recorded per owner discussion): (θ+s)ZZ could
  also be realized by scaling H down and evolving longer, but dephasing
  accrues per wall-time, so that trades headroom for a worse effective
  T/T₂* — at the paper's fixed regime the amplitude cap is the binding
  constraint. One sentence of 6.1 prose can note this.

### A2. p_fail at M=5  [\owed{p}]
- measured (under the sampler): **0.0255**; bound (D.3/D.4 tail):
  **0.0368**. Effective shot inflation 1/(1−p_fail) = **1.026**.

### A3. R = ‖O_P‖
- **R = 1.0** (readout O = Z₀Z₁, unit spectral norm).

### A4. Floors (RMSE at N=10⁶ in parentheses)
- PSR+gate: exact insertion bias **0.0138** (RMSE tail 0.0167)
- NSR^M_trunc: measured **0.0123** (RMSE tail 0.0159); Lemma D.5 bound
  **0.368** — bound ~30× loose here (alternating tail cancels;
  contradiction 3)
- FD ε*: predicted δ/ε floor **0.159** (RMSE tail 0.152)
- FD fixed ε=0.05: predicted **0.179** (RMSE tail 0.175)
- **PSR+gate floor < FD ε* floor: CONFIRMED** (0.0138 < 0.159, 11.5×).

### NSR@cap variants (both measured, trunc plotted)
- (a) NSR^M_trunc: compile-time truncation; sampler renormalised over
  κ ≤ M **and** the L1 weight scaled by the kept mass, so every kept mode
  keeps its exact full-series weight. Floors at the tail bias (A4).
- (b) NSR^M_rej: draws from the full sampler; out-of-range draws are
  **rejected, never resampled** (consume budget, contribute 0, L1
  unchanged) — this leaves the kept-mode weights undistorted, which is
  the content of "unbiased" here: under a hard cap no estimator reaches
  the tail, so both variants share the same truncated target and the
  same floor; naive *resampling* would silently renormalise and distort
  the weights. Measured: rej RMSE tail 0.0162 ≈ trunc 0.0159, at 1.026×
  the executions per useful shot. If the text intends a stronger sense
  of "unbiased" for (b), flag to the theory owner — it is not achievable
  under the cap.

--------------------------------------------------------------------
## B. F_loop (figs/F_loop.pdf)

Unchanged from v9 except the on-figure T/T₂* = 0.15 stamp required by the
ground rules (legend ε values, terminal step markers, tolerance line,
median ± IQR over 20 seeds, inset valley with median paths all kept).
Run id: `build_Floop_trajectory.py` @ commit 8e36d9d, cached trajectories
`figures/F_loop_curves.npz` (REPLOT; env W=0.25, TSTAR=(1,1),
B_BUDGET=6000, ITERS=100), 20 seeds.

For the record (not printed):
- tolerance **0.03** (≈1.5δ, δ = 0.02 programmed resolution jitter)
- three ε: **0.15** (retrospective best), **0.7** (too large), **0.04**
  (too small)
- enter tolerance (median holds 5 consecutive steps): **PSR @ 10**,
  **NSR @ 34**, **FD-best: never** (see contradiction 5)
- FD-best stall distance: median ‖θ−θ*‖ ≈ **0.040** at step 50 (0.036
  median over steps 40–50; 0.044 at step 100)
- fraction of seeds inside tolerance at step 50: PSR 80%, NSR 80%,
  FD-best 35%, FD-small 25%, FD-large 0%.

--------------------------------------------------------------------
## C. F_select (figs/F_select.pdf)  [\owed{balanced-plane run}]

Run id: `build_F_select_balanced.py` @ commit 3665560, cache
`figures/F_select_balanced_data.json`; 6 seeds/cell, seeds
`default_rng(97s+3P+11k)`. Hamiltonian level, no noise (stated on
figure). Shading = measured winner (blue PSR / green NSR), solid =
measured crossing. **The dashed compiler-choice line is NOT drawn**: the
Sec 5.3 certificate (diam → 2Σ|v|, σ → √2) admits NSR at strictly lower
C nowhere on the plane (PSR-or-tie everywhere), so the compiler picks
PSR on the whole plane and there is no certified crossing to draw — the
choice is stated in the caption instead.

### C1. Family / ranges  [\owed{balanced-plane numbers}]
**Balancing option 1 was used** (no family change): same 7-qubit TFIM
device alphabet {X_a, Z_a, Z_aZ_b} (7+7+21 = 35 terms), k extended
upward to the full alphabet, P pulled down: **P ∈ [1,10], k ∈ [1,35]**.

### C2. NSR share
**42.3%** of the sampled plane (measured; near half-half, honest — no
other tuning).

### C3. Forfeiture of the compiler's choice  [\owed{regret}]
Max **5.76×** executions vs the measured optimum; median over the whole
plane **1.00×** (the choice is optimal on 64% of cells); median over the
divergent cells **1.35×**. Divergence is one-sided (a loose certificate
costs shots, never bias).

### C4. Markers
TFIM star (per-bond θ) at **(P=2, k=1)**, measured ratio 10^{+0.01} — on
the crossing (contradiction 4). Global-coefficient rewrite at
**(P=1, k=2)**, ratio 10^{+0.01} — likewise a near-tie.

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
- [x] figs/F6.pdf (+png) — single column, inset V, no title, ≥7 pt
- [x] figs/F_loop.pdf (+png)
- [x] figs/F_select.pdf (+png)
- [x] figs/F_scale_app.pdf (+png)
- [x] SEC6_DATA_REPORT.md (this file)
