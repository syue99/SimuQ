# Owed-figure report — appendix pass

2026-09-02. Answers to `FIGURE_OWED_GUIDE_0902.md`, in its §6 order.
**Numbers first, files second.** Nothing here edits a caption or an appendix
sentence — that is the writing side's call once these numbers land.

Commits: `6bae59d` (App F waveform figure), `bfbc1a8` (§3 fixes),
`19a8a36` (App G.4 fig:regimes + App H compile curves).

---

## §1 — Rerun set (Figs 2, 8, 9): NOT STARTED, and deliberately so

The guide says not to start the rerun set until Fred's δ change lands
("do not start the rerun set until his change lands"). It has not landed in
this tree: every builder still draws `δ ~ N(0, r²)` on the **FD arms only**
(`build_F6.py:234`, `build_Floop_trajectory.py:145-146`, `build_fig1.py:145`,
and six more), and PSR/NSR are still executed at the nominal setpoint. So the
four post-rerun checks are not answerable yet, and the ε-convention relabel
(paper-ε = 2 × code-ε) has not been applied to any figure.

**One thing worth deciding before that rerun**, because it changes what the
change means rather than just its size: the appendix's T4 setpoint cell
(`appendix_2.tex:988-991`) describes r as *"a **deterministic** coefficient
offset θ↦θ+r fixed per operating point, not a per-shot draw"* applied to
*"**all strategies**"*. Neither clause matches any builder today. Extending the
current i.i.d. per-programming draw to every method (as §0 describes) makes the
second clause true but leaves the first false, and the paragraph at
`appendix_2.tex:997-1010` — "it survives averaging because it is not a
fluctuation" — is a statement about a deterministic offset. Under an i.i.d.
draw the δ/ε term survives *shot* averaging but would average down over
independent re-programmings. Fred should say which model he means before the
reruns bake it in; full write-up in `FIGURE_QA_ANSWERS.txt`.

---

## §2 — fig:regimes (App G.4): DONE, 141 s, cached

`tests/build_F_regimes.py` → `figures/F_regimes.{pdf,png}`,
`paper_fig_3/figs/`, cache `figures/F_regimes_data.json`
(`--replot` never recomputes; `--recompute` re-sweeps).

Fig 10's grid (p ∈ [1,10] × q ∈ [1,35], 6 seeds), Fig 10's operating point,
cost model and shot counting — all imported from `build_F_select` /
`selector_check`, not re-implemented. The old p ≤ 20 × q ≤ 14 sweep is
discarded. Axis labels are **q** and **p**.

**Reproduction check: panel (a) reproduces the published Fig 10 plane to
max |ΔZ| = 2.2e-16.** That is what licenses reading the three panels against
each other, and it is asserted in the test suite, not just checked once.

### The three agreement numbers you asked for

Selector = eq:margin on Ω̄_AC, γ(q) = min(1, 1.86/√q), exponent pinned to √q:

| family | NSR share (measured) | agree L1 | agree AC | **agree AC+margin** | max forfeit | NSR chosen |
|---|---|---|---|---|---|---|
| general (device signature) | 42.3% | 57.7% | 71.4% | **88.3%** | 1.26× | 46.0% |
| aligned (ZZ-only) | 27.7% | 72.3% | 72.3% | **92.6%** | 1.39× | 20.9% |
| Heisenberg bonds | 94.3% | 5.7% | 60.6% | **96.0%** | 2.17× | 90.3% |

**The aligned number, which G.4's margin repair hangs on: the margin does not
overcorrect there.** It raises agreement 72.3% → 92.6% and chooses NSR on 20.9%
of cells against a measured 27.7% NSR share — i.e. it stays *conservative* on
that family rather than over-claiming NSR. On the guide's own criterion, the
planned fallback (index the margin on the grouping gain Ω̄_AC/Ω̄_L1 instead of q)
is **not needed**. Not implemented, per the guide's instruction.

### Grouping gain, measured — and a claim of mine that the data corrected

I had written into the builder's docstring that the Heisenberg family would
degenerate to Ω̄_AC = Ω̄_L1, on the grounds that XX, YY and ZZ on a bond commute.
That is true only while a tangent fits inside **one** bond:

| mean Ω̄_AC/Ω̄_L1 | q=1 | q=3 | q=6 | q=12 | q=35 |
|---|---|---|---|---|---|
| general | 1.000 | 0.953 | 0.913 | 0.890 | 0.883 |
| aligned | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Heisenberg | 1.000 | 1.000 | 0.869 | 0.739 | 0.603 |

Past one bond, terms on bonds **sharing a qubit** anticommute (XX₀₁ vs YY₁₂
differ on site 1 alone), cliques form, and the certificate tightens with q. The
docstring is corrected and both facts are unit-tested. The family where grouping
genuinely never helps is **aligned** (ZZ commutes with ZZ: 1.000 at every q).

### Alphabet caveat (stated on the figure)

The device signature has no XX/YY, so panel (c)'s **tangents** are drawn from an
extended 77-term alphabet (35 device + 21 XX + 21 YY). The background, readout
and states stay on the device pool — which is why (a)/(b) still reproduce
Fig 10 exactly. This also scopes G.2's √2-vs-√3 fork: Fig 10's pool has no Y;
Fig 11 panel (c) does.

### Tie rule

Exact certificate ties after the margin are 6 / 5 / 8 cells (general / aligned /
Heisenberg). Agreement with ties→PSR vs ties→NSR: 88.3/86.6, 92.6/91.1,
96.0/95.4. The rule moves nothing by more than 1.7 points here — unlike Fig 10,
where 41 tie cells drove the old 42-vs-64% confusion. Figure uses ties→PSR.

---

## §3 — Small fixes

| item | status |
|---|---|
| `p_fail = 0.0255` → analytic excluded mass | **done** — 0.0337 (3.4%) |
| T4 / `sec6_T4_noise_table.py`: T₂* fixed, T derived — backwards | **done** |
| Fig 7 ledger row `Rz(s·0.25π)` → `Rz(s·0.5π)` | **done** — label only, see below |
| Fig 10 axis relabel k → q, P → p | **done** — replot, plane unchanged |
| `compile_curves.py` for App H, with real generic points | **done** — §"App H" below |

**p_fail.** The cached 0.0255 was the *24-mode sampler's* tail beyond the cap,
which understates the rejection rate because that sampler is itself truncated at
MAXN=24. The deployable number is the analytic excluded mass of the full
1/(n+½)² series, ψ′(M+3/2)/(π²/2) = **0.033696 = 3.4%** at M=5 — matching what
the paper quotes. Shot inflation follows it: 1/(1−p_fail) = **1.0349** (was
1.0261). Both values are now dumped under names that say which is which
(`p_fail` = analytic, `p_fail_sampler` = the old one) and `build_F6.py` computes
both. The D.3/D.4-style bound at M=5 is 0.0368, so the analytic mass sits under
its bound as it must — the old sampler value did too, which is why this slipped.

**Fig 7 ledger row — the compiled phase was never wrong.** Worth being explicit,
because "fix the row" could have been read as "fix the ledger".
`cz_kick_decomposition` stores φ, the coefficient of Z in e^{-iφZ}, and for the
PSR kick φ = π/4. The figure prints an **R_z angle**, and eq:zz-lower fixes
R_z(α) = e^{-iαZ/2}, so the same frame update is α = 2φ = s·0.5π. Printing φ
inside an `Rz(...)` label halved it. Fixed the label, left the ledger alone, and
left a comment at the site so nobody "corrects" the ledger to match the old
label. Re-extraction reproduced the schedule bounds exactly
([0, 2085, 54466, 55162, 107543, 110458] ns).

**T4 dephasing row.** Two corrections. (i) The ruling is the *ratio*: T is each
figure's segment duration and T₂* is derived — F6 T=5.0 µs → T₂*=33.3 µs
(matching App E's "dressed T₂ ≈ 33 µs at the 5.0 µs segment"), F-loop
T=0.8 → 5.33, Fig 2 T=12 → 24 at ratio 0.5. "T₂* = 5.0 µs" appears nowhere in
the emulator and should never be quoted as a device default. (ii) The row now
writes the GKLS normalization out, because "at 1/T₂*" reads either way and the
factor of 2 changes what T/T₂* = 0.15 means: the code builds c = √(γφ/2)·Z, so
the master-equation term is (γφ/2)(ZρZ − ρ) with γφ = 1/T₂*, coherences decay
exp(−t/T₂*), and T/T₂* = 0.15 is a factor exp(−0.15) = 0.861 over the segment
(not exp(−0.30) = 0.741).

**Also worth a line, not on the guide's list:** T4's channel inventory says
"dephasing / decay" (`appendix_2.tex:978`), but there is **no amplitude damping
anywhere in Sec 6** — `T1` is never set by any builder, so the σ⁻ collapse
operator is never constructed, and `leakage_rate` is never set either. The row
should read "dephasing".

---

## §4 — Answers

### G.1, pool construction (unblocks the App G writing session)

- **Sampling per cell.** 6 seeds per (p, q) cell; each seed is one independent
  program of p tangents, each tangent drawing q terms without replacement
  (`build_F_select.draw_params`, rng seeded `97·s + 3·p + 11·q`, so every cell
  is reproducible in isolation). Cell value = mean over the 6 seeds of
  log₁₀(N_NSR/N_PSR).
- **The 11,550 tangents decompose exactly as** Σ_{p=1..10} p × 35 q-values ×
  6 seeds = 55 × 35 × 6 = 11,550. (350 cells; the count is not uniform per cell
  — a cell contributes p tangents per seed, from 6 at p=1 to 60 at p=10.)
- **Alphabet: confirmed {X_a, Z_a, Z_aZ_b} only, no Y_a.** 7 + 7 + 21 = 35 terms
  on 7 qubits, which is also why q tops out at 35. Unit-tested
  (`test_no_Y_in_the_device_signature`). This is the √2 side of G.2's fork —
  with the caveat that Fig 11 panel (c) deliberately steps outside it.
- **The target ε_t behind "executions to target": there isn't a numeric one, by
  construction.** N_S = C_S²·ε_t⁻²·log(1/δ), and the sweep plots C_S² in units
  where ε_t⁻²·log(1/δ) = 1 (also R = 1, dt = 1). Changing the target rescales
  the colorbar and leaves every boundary fixed — that is the target-independence
  claim, and it is why no ε_t needs quoting.
- **Winner and shot counting.**
  N_NSR = Σ_l diam(A_l)², with diam the *exact spectral* diameter of the
  tangent (eigenvalue computation, the oracle);
  N_PSR = 2 Σ_j n_j with n_j = max_l S_l·|c_j|·σ_j and S_l = Σ_{j'} |c_{j'}|σ_{j'}
  — the max over parameters is **cross-parameter branch reuse**: one branch per
  alphabet term serves every coefficient that touches it.
  σ_j is the *measured* mean per-branch shot std of term j at the operating
  point (4 high-entropy states, background drawn on the device alphabet,
  readout Σ_a Z_a). The predicted/certificate variants replace diam by Ω̄ and
  σ by its bound √2.

### H comparison table: BLOCKED

"The executions N at which each method's RMSE first crosses a stated target"
must come from the **reran** Fig 8 cache (§1), which is waiting on the δ change.
Asking the current cache would produce numbers that the rerun invalidates.
Ready to run the moment §1 unblocks; say what target you want (the natural one
is the same usable-ε criterion Fig 1/F6 already share, RMSE/|∇C_device| < 0.5).

---

## App F — the per-channel waveform figure (NOT in the guide; owner-requested)

The guide's §5 cuts it; the owner un-cut it on 09-01 with the observation that
the pipeline already exists, so it need not be a cartoon. It is now real:
`tests/build_F_waveform.py` + `F_waveform_render.py` →
`figures/F_waveform.{pdf,png}`, `paper_fig_2/`, `paper_fig_3/figs/`, cache
`figures/F_waveform_data.npz` + `_meta.json`.

Running example H(x) = sin(2x)(Z₀Z₁ + X₀ + X₁) at x = 0.7, T = 5.0 µs, two
qubits, through the same pipeline as Fig 5. **Both lanes come out of ONE
compile**, which is what makes the comparison mean anything: same geometry, same
solve.

**The claim is structural, and the figure is scoped to it (owner ruling,
09-02).** The PSR branch has to key the transport AODs and the gate AOM, because
the inserted CZ is a digital operation in the gate zone; the NSR branch is the
source program's own schedule with different amplitudes and never touches those
channels. An earlier draft of this figure headlined the per-branch wall clock
(119.83 µs vs 5.00 µs, 95% of the PSR branch being transport + gate) — that
framing is **withdrawn**: on a real machine the branch is not the operative
cost, since measurement and atom loading / rearrangement dominate the shot
budget, and a microsecond-scale difference between branches decides nothing. The
figure now carries no speedup factors, no percentages and no on-figure title;
column widths (wide PSR, narrow NSR) and the labelled axis spans convey the
durations without asserting anything from them. 7.0 × 2.55 in, sized for a
`figure*` rather than its own page.

**The NSR branch is exact, and the builder proves it rather than asserting it.**
The three terms share one coefficient, so H(x+s) = [sin(2(x+s))/sin(2x)]·H(x),
and every machine instruction is linear in its amplitude variable at frozen
geometry. Scaling the amplitudes realizes the shifted target with residual
8.826e-5 = source residual 8.960e-5 × |scale|, exactly — no solve, no re-map, no
added compilation error. Drawn mode n = 0: K = 1.2098, s = 0.20665,
scale = 0.98507.

Two alternatives were measured and rejected; both are recorded in the module
docstring so nobody retries them. A **generic recompile at x+s** moves the atoms
([−10.49, 0.88] → [−10.05, −3.21]) and jumps amplitudes ~10× — that is the FD
lane's full recompile, not a shift. A **frozen-geometry re-solve** lands in a
degenerate dressing/detuning direction (~10⁵ cancelling amplitudes) because the
shifted problem is ill-conditioned without the source as a warm start.

One note for whoever writes the caption: this figure uses the module's realistic
transit lanes (transit_dy = 5 µm: lift, travel, drop) whereas Fig 5 simplifies
to direct single-leg moves on an event-spaced axis, so the two timelines are not
numerically identical by construction.

---

## App H — fig:compile-curves

`tests/compile_curves.py` → `figures/compile_curves.{pdf,png}` +
`paper_fig_3/figs/`. **Plot-only**, from `F_scale_data.json` and
`sec6_compile_timing.json`; nothing re-timed, since re-timing only adds jitter
to numbers the appendix already quotes.

- **Real generic-path points exist and are now drawn** (the guide's optional
  item): n = 2, 3, 4, 5, 6, 8, 10, 12 at 0.032 … **27.063 s**, which *is* the
  guide's (n=12, 27 s) anchor. The n^4.4 line is kept only as a labelled
  extrapolation past the measured ceiling, stopped where it reaches ~1 day of
  compiling (n ≈ 78).
- **The exponent depends on the fit window, and the figure now says so.** Over
  all eight points the slope is **3.79**; over n ≥ 5 it is **4.39**. The
  appendix's 4.4 is the asymptotic-window fit and is right, but a reader
  refitting all eight points gets 3.8, so the window is stated on the figure.
  (Specialized 1D: 1.69 over all points, 1.97 over n ≥ 100. 2D: 1.92 / 1.97.)
- 2D series included per the owner's ruling, with the declared-dropped diagonal
  J/8 tail (~14% relative L1) disclosed on the figure.
- Panel (b) numbers at n = 1000: source compile 59.6 s, PSR branch 175.0 ms,
  NSR branch 0.191 ms (**916×**).
- **FD is drawn twice, and the correction matters.** The earlier draft plotted
  only D4's expensive path, which reads as "FD is intrinsically expensive". It
  is not. D4 measured both: a black-box FD branch that calls the compiler again
  at x+ε pays **5.59 s = 99.4%** of a source compile, but the same branch routed
  through the specializer's closed-form shift table costs **0.059 ms** —
  indistinguishable from NSR's own branch at that n (0.053 ms, a 1.11× ratio).
  So FD's compile cost is a *reuse* question, not an intrinsic one: FD is free
  exactly when it reuses the differentiation infrastructure it is usually
  motivated by not needing. What actually separates FD from the shift rules is
  statistical (Fig 8), not compile time.
- Timings are machine-dependent; the machine string
  (`macOS-26.5-arm64-arm-64bit`) is read from the cache and stamped on the
  figure.

---

## Tests

`test_F_waveform.py` (15) and `test_F_regimes.py` (20) both pass; 35 total, 1.4 s.
They cover the pure functions (amplitude scaling with the phase convention for a
negative scale, the extended alphabet's Pauli bit-vectors, the family draws
delegating verbatim to Fig 10, the margin's cap) and the cached artifacts'
invariants (NSR lane emits no transport/gate, the shifted branch is an exact
rescale, panel (a) reproduces Fig 10, the certificate chain L1 ≥ AC ≥ true holds
in every cell).

## Files

| what | where |
|---|---|
| App F waveform figure | `tests/build_F_waveform.py`, `tests/F_waveform_render.py`, `figures/F_waveform.*`, `figures/F_waveform_caption.txt` |
| fig:regimes | `tests/build_F_regimes.py`, `figures/F_regimes.*`, `paper_fig_3/figs/F_regimes.*` |
| App H curves | `tests/compile_curves.py`, `figures/compile_curves.*` |
| §3 fixes | `tests/build_branch_anatomy.py`, `tests/build_F6.py`, `tests/sec6_T4_noise_table.py`, `T4.csv`, `tests/build_F_select_balanced.py` |
| δ-model write-up | `FIGURE_QA_ANSWERS.txt` |
