# Plot / builder handover — SEC6 figure set

2026-08-27. How to take over the paper figures in `paper_fig_2/` (the
folder the .tex pulls from) and `paper_fig_3/figs/` (the SEC6 handover
deliverable set, kept in sync). Numbers behind the figures:
`paper_fig_3/SEC6_DATA_REPORT.md`.

## Golden rules

1. **Never re-run a simulation to tweak a plot.** Every figure has a
   cache + a replot path (table below). If a style change seems to need
   a re-run, you're on the wrong path — find the cache.
2. Environment: `conda run -n qec_pg python <builder>` from the repo
   root. Local SimuQ fork in `src/` (never pip install).
3. Every figure states its regime on-figure (`T/T₂* = 0.15`, or
   "Hamiltonian level, no noise", or "specialized path, 1D chain").
   No titles inside figures — titles live in captions.
4. Single-column width ~3.0–3.4 in, fonts ≥ 7 pt (owner-approved
   exceptions: F6 legend 5.2 pt).
5. Strategy colors are paper-wide (Okabe–Ito), never re-assign:
   PSR `#0072B2` (blue), NSR `#009E73` (green), FD `#D55E00` (orange;
   the too-small/too-large F_loop arms use `#7b3fa0` / `#d62728`,
   FD-oracle in F_loop uses `#E69F00`). Text over colored fills gets a
   white halo (`patheffects.withStroke`).
6. Commit figure + builder + cache/JSON together, with the "why" in the
   message. Verify by opening the PNG (builders save a PNG twin next to
   every PDF) before committing.

## Figure ↔ builder ↔ cache map

| Figure | Builder (tests/) | Cache (figures/) | Replot cost |
|---|---|---|---|
| F6 (single col) | `build_F6.py` (full sim ~5 min) | `F6_floor_amplification.json` | **`replot_F6.py`, ~2 s** |
| F_loop (single col) | `build_Floop_trajectory.py` | `F_loop_curves.npz` + `F_loop_meta.json` | `REPLOT=1` + env pins, ~1 min |
| F_select (balanced) | `build_F_select_balanced.py` | `F_select_balanced_data.json` | render-only if cache exists, ~10 s |
| F_scale_app | `sec6_compile_timing.py` (+`build_F_scale.py`) | `F_scale_data.json` + `sec6_compile_timing.json` | re-runs D4/D5 (~2 min) then renders |

## F6 — `paper_fig_2/F6.pdf`, `paper_fig_3/figs/F6.pdf`

- **Style changes: edit `replot_F6.py` ONLY, run it, done.** It renders
  the whole single-column figure (RMSE-vs-N main + FD-V inset) from the
  JSON cache. The same layout also exists as the `figS` block inside
  `build_F6.py` — **keep the two in sync** so the next full run doesn't
  revert your styling.
- Current style: figsize (3.0, 3.1), dpi 300; legend 5.2 pt upper right;
  inset at axes-fraction `[0.07, 0.085, 0.48, 0.30]`; y-label
  `RMSE vs ∇C_device` (v11 name — do not reintroduce "noisy");
  `T/T₂* = 0.15` stamped top-left.
- **Trap:** the inset top edge must stay ≤ ~0.40 axes fraction or it
  clips the PSR/NSR curves near N = 10⁴ (they pass y-fraction ≈ 0.44).
- JSON series keys: `psr`, `nsr`, `nsr_trunc` (M=5 cap, plotted),
  `nsr_rej` (reported only), `psr_gate`, `fd`, `fd_fixed`; bands exist
  for all except `psr_gate`/`fd_fixed` (add them to the dump on the
  next full run if you want their IQR ribbons).
- **When a full re-run is required** (new physics, new δ/gate rates, or
  App E lands a real s_max): `build_F6.py` is fully seeded and
  reproduces exactly; ~5 min; it also rewrites the two-panel appendix
  `figures/F6_floor_amplification.*` and the caption txt. NSR@cap
  params live near the `S_MAX = th0` line — s_max = θ0 (2× coupling
  headroom, √2 Rabi, J∝Ω²) is **PROVISIONAL** pending App E.

## F_loop — `paper_fig_2/F_loop.pdf` (+ `F_loop_full` appendix)

- Replot command (env pins are **mandatory** — the builder's defaults
  differ from the cached run, and a bare run would recompute θ*, the
  landscape, and every distance wrongly):

  ```
  REPLOT=1 TSTAR=1.0,1.0 W=0.25 B_BUDGET=6000 FLOOP_ITERS=100 \
    conda run -n qec_pg python differential_computing/tests/build_Floop_trajectory.py
  ```

  Sanity check: the setup printout must say `θ*=[1. 1.] … W=0.25
  μ_soft=0.310 START=[0.802 1.251]` — if it doesn't, the env pins are
  wrong; stop. (Run params are recorded in
  `figures/F_loop_trajectory.json`.)
- REPLOT skips the 20-seed descent sims but still evaluates the
  landscape grids (2q QuTiP, ~1 min).
- Layout facts: valley inset in DATA coords
  `inset_axes([27.5, 0.135, 22.0, 0.325], transform=ax.transData)` —
  the floor was raised from the spec's 0.08 because the purple IQR band
  tops out at 0.128 in that window; re-check band extents before moving
  it. Legend lower-left inside axes; terminal markers = first step the
  median holds tolerance 5 consecutive steps; `T/T₂* = 0.15` stamp.
- Outputs land in `paper_fig_2/` (also `F_loop_full.*` = the unchanged
  two-panel appendix render, plus caption txt + `F_loop_note.md`).
  **Copy to `paper_fig_3/figs/` manually** (`cp`) — this builder does
  not write there.

## F_select — `paper_fig_2/F_select.pdf`, `paper_fig_3/figs/F_select.pdf`

- `build_F_select_balanced.py` imports `build_F_select` and patches
  `PS = 1..10`, `KS = 1..35`, `CACHE` → the balanced plane (NSR 42.3%).
  If the cache JSON exists it renders + recomputes stats only; **delete
  `figures/F_select_balanced_data.json` to re-sweep** (~2 min, 7q).
- Current format (owner-requested): gray half-decade cost fill
  (`#f4f3f0 → #43423f`) + `executions to target (best strategy)`
  colorbar with decade ticks, PSR/NSR winner washes (alpha 0.30) on
  top, solid black measured crossing, star = TFIM (P=2,k=1), open
  circle = global-θ rewrite (P=1,k=2), stamp "Hamiltonian level, no
  noise".
- The dashed compiler-choice contour is drawn **only if** the
  certificate surface `Zpred` takes both signs (it currently never
  admits NSR — PSR-or-tie everywhere — so no dashed line; the caption
  carries the statement). Sign convention: `Z, Zpred =
  log10(N_NSR/N_PSR)`, so **NSR wins where Z < 0** — a sign slip here
  already caused one bug; the forfeiture stats in `main()` depend on it.
- Legacy: `build_F_select.py` renders the OLD 15%-NSR plane to
  `figures/F_select.*` — don't confuse it with the paper figure.

## F_scale_app — `paper_fig_3/figs/F_scale_app.pdf`

- `sec6_compile_timing.py` does D1–D6: extends the **resumable**
  `F_scale_data.json` (only missing n-points are timed; the cache is
  rewritten after every point, so a killed run resumes), measures D4
  (FD full recompile at x+ε) and the D5 3×3 P/k scan at n=300, dumps
  `sec6_compile_timing.json`, renders the two-panel appendix figure
  (top: source/+PSR/+NSR vs n; bottom: D5, PSR k-lines in an orange
  ramp over the NSR dashed line).
- A re-run re-times D4/D5 (~2 min, timing jitter expected); the vs-n
  curve comes from the cache and is stable. There is no pure-replot
  path — if you only need style, add one (read both JSONs, skip
  measurement) rather than accepting jittered numbers.
- Timing numbers are machine-dependent (recorded machine in the cache
  `meta`); if you re-time on different hardware, flag it in the report.

## Where things go

- `paper_fig_2/` — what the paper includes. F6 + F_select builders
  write here directly; F_loop writes here natively; captions
  (`F_loop_caption.txt`) live here too.
- `paper_fig_3/figs/` + `paper_fig_3/SEC6_DATA_REPORT.md` — the SEC6
  handover deliverable set. F6/F_select builders write both places;
  F_loop is copied by hand. Keep the report's numbers in step with any
  re-render (floors, shares, timings are quoted there with run ids).
- `figures/` — caches, JSON dumps, appendix/legacy renders, captions,
  data notes. Caches are committed (they ARE the data).

## Fast verification loop

1. Edit the replot/render script (not the sim).
2. Run it; open the PNG twin with the Read tool; look for: label/curve
   collisions, inset clipping, legend overlap, regime stamp present,
   no in-figure title.
3. If numbers changed (not just style): update
   `paper_fig_3/SEC6_DATA_REPORT.md` and the caption/note files the
   builder rewrites.
4. Commit figure + script + cache/JSON together.
