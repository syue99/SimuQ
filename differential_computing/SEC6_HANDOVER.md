# SEC6 Handover — mapping built work onto the new outline (2026-08-19)

**Purpose.** The outline was reorganized (three RQs + page-budget ruling). This maps
everything already built onto the new spec, flags what the reorg *invalidates*, lists the
genuinely new runs, and hands over the practical machinery.

**Governing spec:** `Downloads/SEC6_FIGURE_SPECS_0819.md` (§0–§8 build rules) **and its REV 1**
(page-budget layout). Conflict rule: **merge rulings (M1–M6) and REV 1 win** over the older
per-figure revision files (FIG1/F6/FLOOP_REVISION) and over anything in this repo's history.

**Env / how to run:** `conda activate qec_pg`; scripts in `differential_computing/tests/`,
outputs in `differential_computing/figures/`. Answers/status doc: `SEC6_DATA_ANSWERS.md`.

---

## 0. THREE reorg changes that invalidate or move built work — read first

1. **M = ∞ headroom cap (G1) — supersedes the NSR-clipping work.** All NSR runs are now
   *uncapped*: no shift rejection, no truncation. This **retires** the D2 NSR-clip
   instrumentation I just added to `build_Floop_real.py` (which measured 9.3% clip near θ*).
   Under M = ∞ the NSR probe shift is executed as-is — **do not box-clip the NSR shift**;
   instead extend the sampled landscape grid to cover θ_i ± s_max, and state "M = ∞, no
   clipping" in every NSR series + note. Wording rule: "NSR shows no floor" is a statement
   *about the uncapped estimator*, not about a bounded device. (The coupling **box** on the
   optimisation variable is separate from the headroom cap on the probe shift — decouple them.)

2. **Instrument relabel (G0) — every compiled-pipeline figure.** Stop saying
   "Hamiltonian-level" for F6 / F-loop / F-phase / F-scale. New label: **"compiled to
   machine-native segments; emulated under the T4 noise model."** Since M1 retired the
   validator, these runs *are* the lowering-correctness evidence. **F3 alone** stays
   "Hamiltonian-level" (it is a deliberate structure study — a scope statement, not a caveat).

3. **Page budget (REV 1) — 3 floats, most figures move.** Only ~0.78pp of floats fit:
   - **Fig A** (full width, 3 panels): **F6-L | F6-R | F-loop RESHAPED to multi-P**.
   - **Fig B** (single col): **F-phase** (NEW).
   - **Fig C** (full width): **companion table + compile-time strip** (F-scale-L compressed).
   - **Appendix:** all three F3 panels, T-workloads, T4 table, T2, F-scale-R, the **single-P
     F-loop descent curve**, and every T/T₂*=0.5 stressor.

---

## 1. Per-artifact status

| Built artifact | Files | New home | Action |
|---|---|---|---|
| **Fig 1 intro trap** | `build_fig1.py` → `fig1_intro_trap.*` | 6.2 back-reference (unchanged) | **Reuse**; 5 small fixes (§1 below) |
| **F6 floor/amplification** | `build_F6.py` → `F6_floor_amplification.*` | **Fig A(a)+(b)** | **Rework** series per M2 + three-floor framing (§2) |
| **F-loop (single-P)** | `build_Floop_real.py` → `F_loop_real.*` | **Appendix** (intuition); **Fig A(c)=NEW multi-P** | **Reshape** to multi-P; retire clip logic (M=∞) |
| **C3 gate-bias** | `build_gate_bias.py` → `gate_bias.*` | feeds F6 "PSR+gate" series + F-phase bias-vs-ε_ins | **Reuse** machinery |
| **F3 who-wins** | `phase_who_wins_3panel.py` → `phase_who_wins_3panel.png` | **Appendix** (all 3 panels) | **Fix** panel (c) label, σ, χ-formula (§4) |
| **T4 table** | `sec6_T4_noise_table.py` → `sec6_T4_noise_table.png` | **Appendix** (sole noise-number home) | **Add M=∞ row**; δ/gate rows present |
| **χ / telescoping** | `sec6_rho_chi.py` → `sec6_rho_chi.json` | feeds F3 + companion table | **Reuse** |
| **Compile-scaling** | `compile_scaling_data.json` | **Fig C right** (narrow) + F-scale-R appendix | **Replot narrow** (§6) |

Retired/legacy (do not use): `build_Floop.py` (surrogate), the old `F_loop.*` figures,
`phase_shots_kick_vs_nyquist.*`, and the `rescale_*` / `floquet_*` artifacts (that is the
separate ML-transfer-map paper, not Sec 6).

---

## 2. New runs the reorg creates (the real work)

Ordered by "genuinely new" vs "replot/relocate":

1. **F-loop multi-P (Fig A-c) — the only fully new descent run.** Sweep **P ∈ {2,4,8}** (+16
   if cheap), same TFIM family. Render **executions-to-threshold vs P**, one line per method
   (NSR, PSR, FD@oracle-tuned ε, FD@fixed ε). Two thresholds (loose/tight). Methods that never
   reach threshold within budget → explicit ✗/open marker at the top ("cannot reach" ≠
   "expensive"). Keep λ=0.3 declared, reference-optimum provenance, cumulative-executions,
   identical per-iteration budget, ≥20 seeds, median±IQR, no PSR-vs-NSR ranking, **M=∞ stated
   (no clipping)**. Required numbers: final error; executions-to-threshold at two thresholds;
   wrong-sign update fraction. — *This is the biggest new compute; see infra §7 (checkpoint it).*

2. **F-phase (Fig B) — NEW.** Contour over **T/T₂\* × ε_ins** (both swept *inside the emulator*
   — ε_ins parameterises the inserted segment's channel; no pulse work). Metric = ratio of
   executions-to-target-RMSE, PSR/NSR, filled contour; **unity contour = crossover**. Overlay
   the **predicted boundary from the C_S constants** (dashed) over the measured (solid) — the
   theory→cost→measurement chain is the point. **M=∞ declared** (crossover driven by NSR shot
   cost ∝ Ω̄² vs PSR insertion bias; the "crossover moves with M" claim is NOT measured here).
   Secondary (number in 6.3 prose, **no panel**): PSR bias vs ε_ins is **linear, slope ≤ C_PSR**
   (Lemma C.9) — fit slope, compare to computed C_PSR, state tight/loose. *`build_gate_bias.py`
   already measures PSR bias vs gate rate — extend it to sweep ε_ins and add the T/T₂\* axis.*

3. **Companion table (Fig C left).** Rows = structural cases (aligned; compressible;
   non-involutive). Columns = winner | mechanism (landscape factor | χ⁻² | NSR-only) |
   executions-to-target | physical time/gradient (from segment durations). Footer rows (not
   columns) for compile overhead: NSR = coefficient arrays, +0 segments; PSR = +k segments.
   Joint-extremizability caption **verbatim**; standing correction holds (`sec6_rho_chi.py` has
   the telescoping χ=1/m evidence).

4. **F-scale compile-time strip (Fig C right).** compile time vs n∈{10,50,100,500,1000},
   3 series (source | NSR deriv | PSR deriv), log-log, **narrow-format legible at ~1/3 width**.
   Expected (measure, don't discover): NSR ≈ coefficient-only overhead; PSR adds bounded
   +k segments; neither combinatorial. Need dispersion (repetitions/point). `compile_scaling_data.json`
   may already hold this — verify it covers n=1000 and has repetitions; else re-time.

---

## 3. Fig 1 remaining fixes (§1 of spec — minor, REV 3 still governs)

1. Separate the `step floor ε≳δ (setpoint resolution)` top-label from the `ε=0.25` label.
2. `ε=0.32` label clipped by right spine — move inboard.
3. Put the R10 ε-window result in the data note; window criterion **identical to F6**
   (RMSE/|∇C_noisy| < 0.5 AND sign-error < 5%) — already the case in code, just state it.
4. **Draw the shift-rule tangent ABOVE the noise cone** (blue currently buried in the purple
   wedge — two opposite meanings overlapping). One-line zorder fix in `build_fig1.py`.
5. Caption keeps its ~45-word REV-3 form. Regime stays T/T₂\*=0.5; one 6.2 clause notes F6 is
   0.15 so they don't read as contradictory.

---

## 4. F6 rework (§2) — what changes from the shipped version

Shipped `F6_floor_amplification.*` is REV-3-clean but the new spec changes the **series** and
the **framing**:

- **M2 overrides F6_REVISION B1.** FD's retrospectively-swept best ε is now a **labelled
  "oracle-tuned FD"** series (the word "oracle" is now *required*, not banned — provided no
  series is shot-free). Required (L) series: **NSR (M=∞) | PSR dressing-only | PSR+gate
  (KEEP — the honest disclosure) | FD@oracle-tuned ε | FD@fixed ε.** One colour/strategy,
  linestyle for variants; `FD@fixed ε` may move to the note if unreadable, but **never drop
  PSR+gate**.
- **[BLOCKER] Three-floor framing** is the panel's new point (put in note + 6.2 prose):
  **FD** floor = δ/ε, *uncertifiable*, no knob removes it;
  **PSR** floor ≤ C_PSR·ε_ins (**Lemma C.9**), *certifiable*, set by gate infidelity;
  **NSR** floor = none at M=∞ (capped would be ≤ 4Ω̄R/(π²(2M+1)), **Lemma D.5**), certifiable,
  set by amplitude headroom. This is what makes F6 *motivate F-phase* rather than just beat FD.
- Keep: G3 estimand (RMSE vs ∇C_noisy, already done), tail-fit exponents+R² (done), predicted
  δ/ε floor line + MC-δ note (done), the V-curve (R), sign-error markers, ε=δ + usable window,
  cross-panel horizontals = (L) at same N (done). **Relabel instrument per G0.**
- The current headline floor is **42%** (sharp θ0, C″≈10). The user has an open call to dial
  T=5→~4 toward the paper's "~20%" — decide before final.

---

## 5. F-loop reshape (§3 + REV 1) — the single-P run is now appendix

- The shipped single-P descent (`F_loop_real.*`, PSR 0.0037 / NSR 0.0027 / FD ε* 0.008 /
  3ε* 0.034 / 0.3ε* 0.080) → **appendix intuition figure**. Keep it; it costs nothing.
- Main-text **Fig A-c = multi-P** (see §2.1). All the FLOOP_REVISION rules still apply
  (declare λ, reference provenance, executions axis, ≥20 seeds, no ranking, real estimators)
  **except**: (a) **M=∞** replaces the clip story — remove the clip instrumentation, state
  "no clipping"; (b) the "device cost IS the objective" is fine but λ MUST be declared
  (already done). FD failure events are at **small ε** (matches F6-R + Fig 1 cone) — write to
  the data, not the outline's large-ε expectation.

---

## 6. F3 fixes (§4) — stays Hamiltonian-level, moves to appendix

- **[BLOCKER] Panel (c) label is wrong:** it says "Heisenberg → foldable χ≪1" but measured
  Heisenberg χ≈0.5–0.67 (not ≪1). Fix (A, strongest): swap pool to **telescoping** tangents,
  label "foldable (telescoping), χ=1/m" — cleanest χ⁻² story; or (B): relabel "non-commuting
  contraction, χ≈0.5". Decide *together* with the device-alphabet foldable panel (X+ZZ,
  χ→1/√2): three candidate pools span χ ∈ {1/m, ≈0.5, ≈0.71}; pick two = one layout decision.
- **[BLOCKER] σ=1.37** annotated 3× (no information) → define once in caption or drop.
- Remove the in-image χ formula (it's the single-Pauli specialisation, wrong for telescoping);
  caption references the Def.
- PSR enclave top-right of (c): refine grid/seeds and decide (real reuse effect vs grid noise) —
  don't ship an unexplained island.
- Honesty rail: if NSR never wins on a Pauli system, say so in prose.
- Optional (Fred): diverging heatmap of log₁₀(N_PSR/N_NSR), zero contour = boundary.

---

## 7. Practical infrastructure (only-I-know notes)

- **~50-min wall-clock limit on background jobs.** Long runs get killed. **Solution used:
  per-(method,seed) checkpointing** — see `build_Floop_real.py` (`F_loop_ckpt/` dir,
  `ck(lab,s)` files, θ* + η-robustness checkpointed separately). Relaunching *resumes* (skips
  completed seeds). **Adopt this pattern for every new multi-seed run** (F-loop multi-P,
  F-phase contour). Run with `python -u`; note `conda run` swallows stdout so **checkpoint
  files are the ground truth of progress**, not the log.
- **Never re-run a sim to tweak a plot** (user rule). Cache curves to disk; F6 recomputes fast
  (2-qubit) so it's cached-free, but F-loop/F-phase must checkpoint.
- **Real-estimator machinery (reuse, don't reinvent):**
  - `observable_program_generator(H, T, n_sample=m, diff_var, value, short_kick=False)` →
    PSR branches. **Use `short_kick=False` (exact α=π/2 shift)** — `short_kick=True` has an
    O(η²) bias floor (found in F6 F5).
  - `nyquist_shift.tangent_hamiltonian` + `bandwidth_K` → NSR tangent/bandwidth.
  - `NoisyQuTiPRunner(nq, noise=NoiseModel(...)).make_probs_fn(psi0)` → basis-state
    probabilities; **diagonal-readout shot model** (sample bitstrings from diag ρ) is the
    correct finite-shot model for summed diagonal costs — a naive [-1,1] binomial on ⟨ΣZZ⟩
    saturates and zeroes the gradient (bug caught in F-loop).
  - `psr_grad / nsr_grad / fd_grad` in `build_Floop_real.py` are the multi-parameter real
    estimators (partial `set_parameterizedHam` per component) — lift these for F-loop multi-P.
- **m (insertion-time segments) — G2 requires ONE m across F6/F-loop/F-phase.** But m is a
  τ-convergence parameter ∝ T: **T=5 (F6) needs m=48; T=1.5 (F-loop) converges at m=16**
  (bias 0.03%). Two options for the handover reader: (i) force m=48 everywhere for surface
  consistency (3× F-loop cost), or (ii) keep per-program m with the convergence table +
  "matches Sec 5.4's per-program lowering" justification (what I did; G2 says "identical",
  so **confirm this with Fred** — it's the one place G2 and physics disagree). Convergence
  data is measured in `build_F6.py` (`m_convergence` in the JSON) and reproducible.
- **Gate bias is operating-point dependent** (0.028 at C3's θ0=1.59/T=1.5; 0.014 at F6's
  θ0=1.94/T=5). 6.3 prose must say "order 10⁻², operating-point dependent, name both points."

---

## 8. Global relabel sweep (G0–G9) — apply to EVERY surviving figure

- G0: "compiled to machine-native segments; emulated under T4" (F6/F-loop/F-phase/F-scale);
  F3 = "Hamiltonian level".
- G1: every NSR series + note says **"M = ∞ (uncapped)"**.
- G2: cost axis = executions; per-method accounting stated; m stated once.
- G3: RMSE vs ∇C_noisy; reference = fine central FD (h=1e-3) of deterministic mesolve, δ/shot-free.
- G4: noise numbers from T4 only; δ=0.02 + gate rates provisional (Q1) → "re-render if changed".
- G5: real estimators, no Gaussian surrogates.
- G6: seeds/reps on figure + dispersion band.
- G7 **banned words** (sweep all captions/notes/images): "rescale", "device-target", "raw",
  "iterations", **"validator"**, **"standard form"** (→ "execution normal form"),
  **"5.5 emulator"** (→ "Sec 5.4"), "kick", "Nyquist" alone. PSR cost phrase = **"motion and
  one gate"**. (New bans vs my earlier work: validator / standard form / 5.5.)
- G8: no section refs inside rendered images (caption only).
- G9: regime stated per figure (0.15 headline, 0.5 stressor).

---

## 9. Owed by Fred (blocks runs, not plotting) — §8

1. T4 rates + δ=0.02 + provenance strings (Q1).
2. F3 panel (c): fix (A) telescoping or (B) relabel, decided with the device-alphabet panel.
3. σ=1.37: define in caption or drop.
4. F-loop optimizer + per-iteration budget (OQ2).
5. Whether any evaluated program exercises a **single-qubit insertion** (T-workloads column).
6. Which figures get T/T₂\*=0.5 appendix stressor variants.
7. **G2 m policy**: identical-m vs per-program-m (the physics tension above).

---

## 10. Suggested execution order for the next agent

1. **Cheap wins first:** Fig 1's 5 fixes (§3); F3 label/σ/formula fixes (§6, pending Fred's
   (A)/(B) call); global G7 word sweep across shipped notes.
2. **F6 rework** (§4): add oracle-tuned + PSR+gate series, three-floor note, G0 relabel. Fast
   (2-qubit, no checkpoint needed).
3. **F-loop multi-P** (§2.1): the big new run — build on `build_Floop_real.py`'s estimators,
   checkpoint per (P, method, seed), M=∞ (drop clipping).
4. **F-phase** (§2.2): extend `build_gate_bias.py` to the (T/T₂\*, ε_ins) grid; checkpoint;
   add the C_S predicted-boundary overlay.
5. **Fig C** (§2.3–2.4): companion table (from `sec6_rho_chi` + F3/F-phase numbers) +
   narrow compile-time strip (from `compile_scaling_data.json`, verify/re-time).
6. Assemble the 3 main floats (Fig A/B/C) at final size; push the rest to appendix.
