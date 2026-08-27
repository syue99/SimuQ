# Selector check on the Fig. 9 (F-select balanced) plane — results

Inputs: `figures/F_select_balanced_data.json` (published plane),
`tests/build_F_select.py` (the located cost model, reused verbatim — spot-check
max relative difference vs `cell_costs` = **0.0**), tangents regenerated from the
sweep's deterministic seeds. Script: `tests/selector_check.py` (unit tests:
`--test`, 7/7 pass). Outputs in `differential_computing/selector_check/`
(the guide's `/mnt/user-data/outputs/` does not exist on this machine).

## Verdict

**One fixed margin does not work; one fixed margin *per k* does — and it is
still a pure compile-time static analysis.** The offset the guide worried
about is absent: at the star cell (P=2, k=1), where the certificate is exact,
γ = **1.027** — the shipped cost model's constants (two branches, σ→√2 vs
measured ⟨σ⟩=1.39) already cancel, so there is no order-2 constant sitting
under the plane. Everything the certificate misses is **shape in k**: γ is flat
in P (medians 0.41–0.47, no trend) and falls monotonically in k as
γ_med(k) ≈ **1.93·k^(−0.52)** (R²=0.85 on the 35 per-k medians) — the √q
diameter concentration the L1 bound cannot see and pairwise anticommutation
grouping only partially recovers. The recommended selector pins the exponent at the
theory value −1/2 and calibrates **one constant**:

    choose NSR  iff  γ(k)·(C_NSR^AC / C_PSR)² < 1,   γ(k) = min(1, c/√k)

With c = 1.86 calibrated on the k ≤ 8 rows only, it scores **90.0% agreement
and max forfeit 1.20× on the held-out k ≥ 9 rows** (full plane: 88.3%,
max 1.26×, median mismatch forfeit 1.07×, NSR chosen on 46.0% vs measured NSR
share 42.3%). Since k is a property of the program text, the selector needs
**no online component** on this plane. Caveat for the write-up: the margin is
a *calibration*, not a bound — its 1.2–1.3× worst case is measured, not
certified — and it was calibrated on this instance/operating point (though
the odd-k→even-k and small-k→large-k held-out splits both hold up). The
uncalibrated AC certificate alone is already a strict improvement worth
shipping regardless: it breaks the constant-PSR degeneracy with zero
soundness cost.

## Selector league table (overlay statistics, §3)

Per-cell measured ratios exist (regenerated exactly; published plane
reproduced bit-identically, max |ΔZ| = 0.0, NSR share 42.29% = published).
Selector = "choose NSR iff predicted log-ratio < 0"; agreement/forfeits vs
the measured winner, same three statistics as the main text:

| selector | agreement | NSR chosen | median forfeit (mismatched) | max forfeit |
|---|---|---|---|---|
| Ω̄_L1 (current, Sec. 5.3) | 57.7% | 0% (PSR everywhere — confirmed) | 1.49× | 5.76× |
| Ω̄_AC | 71.4% | 14.3% | 1.35× | 2.77× |
| Ω̄_AC + flat margin (γ_med=0.452) | 67.1% | 58.6% | 1.18× | 1.84× |
| Ω̄_AC + free fit γ(k)=1.93·k^(−0.52) | 88.6% | 46.3% | 1.07× | 1.27× |
| **Ω̄_AC + γ(k)=min(1, 1.86/√k), c from k≤8** | **88.3%** (90.0% held-out k≥9) | **46.0%** | **1.07×** | **1.26×** (1.20× held-out) |

Held-out validation of the calibration (all from cached cells, no new runs):
free power-law fit on odd k → evaluated on even k: 87.6% / max 1.27×; free
fit on k ≤ 17 → k ≥ 18: 78.9% / max 1.84× (the k ≤ 4 plateau biases the free
exponent under extrapolation — which is why the recommended form pins the
exponent at −1/2 and calibrates only c; that version extrapolates from k ≤ 8
to k ≥ 9 at 90.0% / 1.20×).

Notes: the L1 boundary is absent from the plane (predicted ratio ∈ [1.00,
10.0], never < 1 — PSR-or-tie everywhere, as the paper claims). Plain AC opens
a strict NSR region at P ≤ 4, k ≥ 2: 50 cells chosen NSR of which only 1 is
wrong (forfeit 1.005×, a near-tie), and it halves the worst-case forfeit even
where it stays PSR-side. The flat margin overshoots at small k and undershoots
at large k (that is the k-structure), so it *lowers* agreement vs plain AC
while still capping forfeits at 1.84×. The power-law margin's mismatches are
27 wrong-NSR cells (max forfeit 1.27×) and 13 missed-NSR cells (max 1.10×) —
both directions cost shots only, never bias. Figure:
`fig_selector_overlay.{pdf,png}`.

## γ distribution (§4)

γ = measured_ratio / predicted_ratio_AC, per cell (mean-log over the 6 seeds,
the builder's own aggregation):

- median **0.452**, IQR [0.314, 0.773], min 0.225, max 1.036, max/min **4.61**
- vs P: flat (per-P medians 0.414–0.472, no trend) — the cost model's
  cross-parameter branch-reuse term captures the p-direction correctly.
- vs k: monotone decline 1.03 (k=1..4 plateau) → 0.24 (k=35); fitted
  γ_med(k) = 1.93·k^(−0.52). The exponent ≈ −1/2 is the √q shape gap
  predicted in the task brief.
- **Star cell (P=2, k=1): γ = 1.027.** Ω̄_AC = Ω̄_L1 = Ω_true there (q=1), so
  this is pure offset with zero shape contamination — and it is ≈ 1, not ≈ 2.
  The guide's "predicted 4× at the star" arises in the Ω̄R vs Δτ·sup Σ|v|
  convention; the shipped Fig. 9 cost model already folds the branch count
  and the Hoeffding σ-bound into *both* sides symmetrically, which cancels
  the constant. Under the located cost model there is no offset to calibrate;
  decision rule branch taken: "γ shows residual structure in q" — direction:
  certificate too large (NSR under-chosen), magnitude: k^(−0.52), up to 4.6×
  at k=35.

Figure: `fig_selector_gamma.{pdf,png}`.

## Resolution of the 42% vs 64% inconsistency (§5.2)

Neither number is stale and they are the same grid; they differ in **tie
handling, and the ties are being broken by floating-point noise**. 41/350
cells (11.7%) have an *exactly* tied certificate prediction (N̂_NSR ≡ N̂_PSR;
35 of them are the P=1 column, where cross-parameter branch reuse cannot
occur). The builder computes the choice as `Zpred < 0` on
log10-of-ratio values that are ±1e-16 on tie cells: 34 of the 41 landed
negative (→ "NSR") and 7 non-negative (→ "PSR") by rounding accident, giving
the quoted 64%. Deterministic rules give: ties→PSR **57.7%** (= 1 − 42.3%,
the arithmetic the guide expected), ties→NSR **66.0%**. Since measured NSR
wins 35 of the 41 tie cells, ties→NSR is the better (and defensible: on a
certified tie, prefer the strategy with no insertion) — but the draft must
state a tie rule and quote the matching number; 42.3% is strict NSR wins
either way. Related: the same fp-tie issue makes `render()` (which smooths
Zpred before testing for a crossing) and `main()` (raw sign) disagree about
whether the certificate "crosses" — `cert_crossing_drawn=False` in the cache
sits next to a 36%-divergence statistic computed from fp-broken ties.

## Regression and consistency checks (§5)

1. **Reproduce the plane**: bit-exact (max |ΔZ| = 0.0); NSR share 42.29%;
   builder-convention stats reproduce 64.0% / 1.35× / 5.76×. Verbatim
   `cell_costs` spot-check: 0 relative difference.
2. 42 vs 64: resolved above.
3. **Constant selector confirmed**: coefficients are time-constant (±1 draws;
   checked, not assumed), and the L1-predicted ratio is exactly 1 on every
   cell-seed with no cross-parameter term sharing, ≥ 1 always (up to 10× at
   P=10, k=35 from branch reuse). The guide's "C_NSR/C_PSR ≡ 2" appears here
   as "≡ 1 (tie)" because the model's shared constants absorb the 2; either
   way it is independent of tangent structure — a constant function that no
   fixed threshold can turn into a k-tracking boundary.
4. **Certificate chain Ω̄_L1 ≥ Ω̄_AC ≥ Ω_true**: asserted per parameter draw
   during the sweep (11,550 tangent rows) — zero violations.

## Grouped-certificate diagnostics (§1–§2)

- Unit tests 7/7: single term (2=2=2); q pairwise-anticommuting equal-weight
  terms give exactly 2√q (q=2,3,5) = Ω_true; commuting TFIM-zz degenerates to
  singletons (AC=L1); Heisenberg bond XX+YY+ZZ (all *commuting*): AC=L1=6 vs
  Ω_true=4; frustrated triangle: AC=6 vs Ω_true=4. The last two are the
  known-loose cases: joint-extremizability failure inside commuting families
  is invisible to anticommutation grouping — do not overclaim.
- On the real alphabet {X_a, Z_a, Z_aZ_b} the anticommutation graph has
  **maximum clique size 2** (measured max group size = 2 over all 11,550 rows;
  structurally: nothing anticommutes with both X_i and Z_i inside this
  alphabet). So Ω̄_AC/Ω̄_L1 ∈ [1/√2, 1] per parameter (median 0.887, min
  0.7071) — the AC certificate's ceiling on this device alphabet is a
  constant factor, which is why residual √q shape survives. Richer alphabets
  (Y terms, XX/YY) would group deeper.
- Residual looseness χ_AC = Ω_true/Ω̄_AC: median 0.661, min 0.406; per-k
  medians 1.00 (k≤5) → 0.85 (k=10) → 0.63 (k=20) → 0.475 (k=35): the AC
  certificate is typically *exact* up to k≈5 and degrades as mixed
  non-commuting mass grows.
- Restart sensitivity: random restarts beat the deterministic orderings on
  only 1.65% of rows, max gain 1.21× — with pair-cliques the weight-aware
  greedy is near-optimal, so grouping quality is a minor knob *on this
  alphabet* (it will matter more where cliques are deeper).

## Scalability of the auto-selector (goal note)

Every ingredient of the recommended selector is compile-time and scales past
7 qubits:

- **Ω̄_AC**: symplectic anticommutation is O(q²) bit operations per parameter;
  the clique cover is greedy matching (pairs on this alphabet). No 2ⁿ object
  is ever touched — Ω_true by diagonalization was used here strictly as a
  128×128 reference oracle and is *not* part of the selector.
- **γ(k) = min(1, c/√k)**: the k^(−1/2) shape is n-independent (operator-norm
  concentration of q-term random-sign Pauli sums — the same mechanism at any
  width); the min(1,·) cap keeps the margin from ever penalizing NSR at small
  k, where the certificate is already exact (χ_AC = 1 for k ≤ 5).
- **c**: one constant per device alphabet / operating point. It is cheap to
  calibrate because small-k cells suffice (c from k ≤ 8 extrapolated to
  k ≥ 9 at 90% here) — small-k tangents are small programs, so the
  calibration can even be measured on-device with a handful of executions if
  an analytic value is not wanted. Nothing about the calibration grows with n
  or with program size.

If a future alphabet breaks the √k shape (e.g. deep anticommuting cliques or
strongly aligned coefficients), the same harness detects it: γ flat ⇒ keep
one constant; γ trending in a *static* quantity (k, clique profile) ⇒ fold
that quantity into the margin, still compile-time; γ scattered against all
static predictors ⇒ only then is an online component justified. On this
plane, the third case does not arise.

## Flagged in passing (not fixed; §6)

- The submitted PDF (DiffSimuQ-16) Fig. 9 caption still carries the *old*
  plane's "forfeiting at most 1.9× shots"; the balanced-plane numbers
  (42.3% / 1.35× / 5.76% and whatever agreement rule is chosen) are in
  `F_select_balanced_data.json:balanced_summary`. Version skew to sync.
- `build_F_select_balanced.py` derives its headline stats from fp-sign tie
  breaking (above); if the AC selector is adopted the stats should be
  recomputed with an explicit tie rule.
- The measured plane's own smoothing (σ=0.8 Gaussian on Z before contouring)
  is a rendering choice; all statistics here are computed on unsmoothed cells,
  matching the builder's statistics path.

## Files

- `selector_check_percell.{json,csv}` — §2 table: per (P, k, seed, ℓ) rows
  (Ω̄_L1, Ω̄_AC, Ω_true, χ_AC, n_groups, max_group_size, restart_gain) +
  per-cell N_NSR/N_PSR (measured and certified) in the JSON; summary stats in
  `selector_check_summary.json`.
- `fig_selector_overlay.{pdf,png}`, `fig_selector_gamma.{pdf,png}`.
- Implementation + unit tests: `differential_computing/tests/selector_check.py`
  (`--test` runs §1's table; `--recompute` regenerates; default replots from
  the cached per-cell JSON — no re-simulation).
