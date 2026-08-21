# SimuQ Solver-Scaling Improvement — Handover Guide (2026-08-20/21)

What was built to answer **RQ3** ("does differentiation scale through the
compiler?") by making the SimuQ compilation path itself scale from n≈12 to
n=1000, without adding a new AAIS. Everything here is committed on `main`
(`6649bfb` → `4d1afe4`); 249 tests pass (`conda run -n qec_pg python -m
pytest differential_computing/tests/ -q`).

## 0. The problem

`prov.compile(qs, "quera", "Aquila", "rydberg2d")` on a 1D TFIM chain grew
~n^4.4 and was practically capped at **n=12 (~27 s)**. Four stacked causes:

1. `rydberg2d.generate_qmachine` builds an **O(n²) machine**: n(n−1)/2
   derived ZZ signal lines + a dressing instruction with n(n−1)/2 terms.
2. **Atom positions are free global variables** (even with `inits`), so the
   least-squares solve optimises geometry too, and every dressing
   coefficient is an `Expression` closure-tree over those gvars.
3. `build_eqs` scanned **every instruction per target term**, dressing
   side-effects blew equations to O(n²), and scipy's `least_squares`
   finite-differenced a **dense** Jacobian: (nvar+1) full residual
   evaluations per step with nvar = O(n²).
4. `compile()` ran `generate_as` **twice**, and switch-pruning re-evaluated
   the full residual once per instruction.

## 1. Design decision (user-set): an intermediate layer, NOT a new AAIS

`rydberg2d` stays the single ground-truth device model. A **target-aware
specialization pass** sits between the target QSystem and the solver —
think dead-code elimination + constant folding on the AAIS instance.

### `src/simuq/specializer.py` (new)

- `extract_target(h, n)` — decompose target into 1-body Pauli coefficients
  + ZZ bonds (rejects >2-body / non-{X,Y,Z,ZZ} terms).
- `_embed_interaction_graph(bonds, sites)` — open chain (any labeling) or
  **full row-major m×k NN grid** (bond set must equal the complete grid
  set); anything else raises `NotImplementedError` — it never mis-embeds.
- `make_plan(qs, C_6, shells=1)` → `SpecPlan`:
  - frozen geometry: chain/grid at spacing `R = (C_6/(4|θ|))^(1/6)` so the
    dressing amplitude warm-starts at |o| = 1; qubit 0 translated to the
    origin (rydberg2d hardcodes it); uncoupled sites parked far away;
  - `links` = target bonds only (ZZ lines pruned from O(n²) to O(bonds));
  - `dressing_pairs` = pairs within `shells·R` (chain: exactly the bonds);
  - **declared truncation**: `dropped_zz_l1` = Σ|J_ij|/4 over dropped
    pairs. 1D: ~1.4% relative (next-NN is J/64). 2D: **~14%** (diagonals
    at R√2 carry J/8, geometrically non-cancellable — the compiled model
    stays exact; this field quantifies the physical device's deviation);
  - analytic warm start: `o = sign(θ)`, `d_i = 2c_z(i) + 2Σθ_ij` (degree-
    aware: corners/edges/interior), `Ω_i = 2√(hx²+hy²)`, `φ_i = atan2`.
- `apply_warm_start(mach, plan)` — writes inits into `lvar.init_value`
  (classifies instructions by name; that is the entire plumbing, because
  the solver's `build_obj` already reads `init_value`).
- `nsr_shift_table(plan, s)` — the marginal compile of one NSR
  (waveform-shift) derivative branch: an O(n) closed-form coefficient
  rescale on the SAME schedule structure (dressing o and detunings scale;
  Rabi unchanged). Exactness unit-tested.

### `src/simuq/aais/rydberg2d.py` (parameterised, defaults unchanged)

`generate_qmachine(n, inits, fix_positions=False, links=None,
dressing_pairs=None)`:
- `fix_positions=True`: positions become plain floats → no gvars, dressing
  coefficients become cheap; also the dressing pair Hamiltonian is built
  **without its inert identity part** — cleanHam used to merge all pairs'
  identity coefficients into one O(n)-deep `Expression.__add__` chain,
  which overflowed Python recursion at n≈1000 (identity = global phase;
  the solver never builds an equation for it).
- ZZ line names now `"c{q0}_{q1}_zz"` — the old `"c{q0}{q1}_zz"` was
  unparseable for n ≥ 10. `tweezer_mapper.classify_instruction` parses both.

### `src/simuq/solver.py` (opt-in; legacy behaviour untouched)

- `build_eqs` indexes instruction terms by product tuple → O(1) contributor
  lookup (was: scan all instructions per target term); returns per-equation
  **variable supports**.
- `solver_args["sparse_jac"]` → builds `jac_sparsity` from the supports for
  `least_squares` (grouped finite differences, O(nnz) Jacobians).
- `solver_args["switch_init"]` → warm-started native compiles start the
  {0,1} switch relaxation at 1.0, making the analytic init a residual-zero
  point: **`least_squares` converges in one function evaluation** — the
  solve verifies a witness instead of searching.
- Switch pruning re-evaluates only the equations whose support contains the
  candidate switch (was: full residual per candidate).

### `src/simuq/braket/diffQC_provider.py`

`compile(..., specialize=True, spec_shells=1)`: builds the plan, generates
the pruned machine, applies the warm start, runs a **single** sparse
`generate_as` pass (the old first pass only existed to find an init), and
hands the mapper the frozen geometry via `sol_gvars` and the truncated
`dressing_pairs`. `_run_hardware` forwards `dressing_pairs` to
`TweezerMapper`.

### `differential_computing/tweezer_mapper.py`

- `classify_instruction` handles both ZZ name formats.
- `_build_dressing_H` accepts a truncated pair set and builds its term list
  in one pass (the old `H = H + h` accumulation re-ran cleanHam per pair —
  O(pairs²) symbolic work).

## 2. Headline numbers (all cached — never re-run a sim to tweak a plot)

| quantity | value | data |
|---|---|---|
| generic path | slope ~4.4, ceiling n=12 (27 s) | `figures/F_scale_data.json` |
| specialized 1D chain | n=1000 in ~60 s, slope ~1.9 | same |
| specialized 2D grid | 32×32=1024 in ~104 s, slope ~2.0 | same |
| compiled-H exactness | max\|dH\| ≤ 4e-13 at every size | same |
| one PSR branch (marginal) | 175 ms at n=1000 = **0.29%** of source | same |
| one NSR branch (marginal) | **191 µs** at n=1000 (`nsr_shift_table`) | same |
| full PSR gradient total | 2mM = 95,904 branches at m=48 → **4.7 h** un-amortized | same |
| full NSR gradient total | 2N = 16 tables → **3 ms** | same |

PSR per-branch is two-regime (slope ~1.1 below n≈200, ~1.5 above): the
ledger's per-play position snapshots (n positions × ~4n plays, structurally
O(n²)) are the fastest-growing pass; the dressing-H ledger rebuild is the
largest single pass at n=1000 (~48%). Breakdown measured with instrumented
wrappers; reported as **shares** (the wrapped total exceeds the unwrapped
headline — see the data note).

## 3. Honest scope statements (do not drop these when writing prose)

- **"n=1000" is always the specialized path.** The generic all-pairs path
  dies at n=12. Say so wherever the number appears.
- **Timing scope = machine-native schedule ops + pulse ledger, NOT pulse
  synthesis.** The PulseDSL emission layer has a 16-channel logical cap and
  its physical-channel COMB encoding does not complete at n=100 —
  engineering work, excluded and flagged (G0 wording must match).
- **Topology scope**: open chains + full row-major m×k grids. Stars,
  partial grids, non-uniform couplings, non-row-major labelings are
  rejected with clear errors.
- **2D tail is a disclosure, not an approximation of the compiled model**:
  the compiled NN-grid Hamiltonian is exact; the declared J/8 diagonal tail
  (~14% relative L1) is what the physical atoms add beyond it.
- **NSR vs PSR accounting asymmetry is the RQ3 story**: NSR branches share
  the source segment structure (coefficient table only, +0 segments); PSR
  branches force a structural re-map (branch-specific τ + inserted kick).
  Amortizing PSR's 4.7 h across branches (shared structure, τ enters as two
  durations) is future work; the un-amortized number is the honest one.

## 4. Figure set (FSCALE_REVISION applied)

- `figures/F_scale_strip.{png,pdf}` — Fig C strip (~1/3 page): source /
  +PSR / +NSR series, IQR bands, 0.29% ratio annotated. Caption:
  `F_scale_strip_caption.txt`.
- `figures/F_scale_appendix.{png,pdf}` — (a) generic vs specialized 1D/2D
  with slopes; (b) three series, two-regime slopes with fit windows,
  per-pass inset. Caption: `F_scale_appendix_caption.txt`.
- `figures/F_scale_data_note.md` — accounting (marginal vs full), totals at
  (m=48, M=999), scope, dispersion policy, the suggested 6.4 sentence.
- Builder: `tests/build_F_scale.py` (resumable cache `F_scale_data.json` —
  rewritten after every point, so a killed run continues).
- Superseded intermediates kept for provenance: `compile_scaling_native.*`,
  `compile_scaling_data.json` (the old n≤12 series).

## 5. Tests

`tests/test_specializer.py` — 14 tests: chain/grid plan geometry and
spacing invariants, star-graph rejection, warm-start exactness (edge vs
interior degrees), specialized-vs-target (<1e-9) and vs-vanilla agreement,
two-digit ZZ name parsing, hardware ledger + `verify()` round trip
(gradient reconstruction error < 1e-6), 2D tail bound, grid hardware map,
`nsr_shift_table` exactness.

## 6. Open items

- **Amortized PSR branch compile** (template with τ-parameterized
  durations; kick-relocation only) — would collapse the 4.7 h total.
- **Ledger snapshot O(n²)** — delta-encoding positions would flatten the
  PSR per-branch tail slope.
- **Pulse-emission layer at scale** (PulseDSL channel cap, COMB encoding,
  real shapes) — required before any "device-ready pulses at n=1000" claim.
- **2D in the F-scale narrative** — data exists in `F_scale_data.json`
  (specialized2d series); the strip is 1D-only by design.
- **Non-uniform couplings / other topologies** — specializer raises today.
- G0 instrument-line wording in the spec must be updated by the author to
  "schedule ops + ledger" scope.
