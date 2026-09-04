# DELTA_NOISE.md — the setpoint rule as implemented (P0-0)

2026-09-04. Figure 8 (`tests/build_F6.py`) is the first figure regenerated under
the rule. Figs 2 and 9 follow and will be appended here.

## What a setpoint draw is attached to — the question that decides the figure

`r = 0.02`, zero-mean Gaussian, frozen across the shots of the execution it
belongs to. The rule only fixes *how many independent draws each estimator gets*,
and that is what decides which strategies floor. Two readings:

| model | FD | PSR | NSR |
|---|---|---|---|
| **per_programming** (default, used for the paper figure) | 2 draws (its two probes) | 2m = 96, one per branch execution | N, one per execution |
| per_value (the handover's literal wording) | 2 | **1** — all 2m branches dial the same source coefficient | one per distinct (κ, σ) |

`per_programming` says the setpoint error is noise in the act of dialing, so a
program that is executed again draws again. `per_value` says it is a property of
the value requested, so re-requesting it reproduces it. Selected by
`DELTA_MODEL` in `build_F6.py`; `PSR_DELTA` follows from it.

**The models must not be mixed.** Applying `per_value` to NSR (one draw per
distinct κ,σ) while applying `per_programming` to PSR is not a modelling choice,
it is a bug: it gives NSR a floor no averaging can remove while letting PSR
average, and it is what the first two runs of this figure did.

### Measured floors under each (RMSE at N = 10⁶, 100 seeds)

| series | per_value | mixed (bug) | **per_programming** |
|---|---|---|---|
| PSR | 0.2178 | 0.0274 | **0.0274** |
| PSR + gate | 0.2179 | 0.0274 | **0.0274** |
| NSR M=∞ | 0.1496 | 0.1496 | **0.0102** (no floor, N^−0.50) |
| NSR M=5 | 0.1495 | 0.1495 | **0.0134** (truncation 0.0123) |
| FD @ ε*=0.252 | 0.1390 | 0.1390 | **0.1390** |
| FD @ ε=0.05 | 0.2687 | 0.2687 | **0.2687** |

FD is identical in all three, because FD programs exactly twice per estimate
whatever the model says — which is the point: **the setpoint error is what floors
FD, and only FD.**

### Why per_value inverts the paper

Under it PSR gets one draw at std r and no averaging at all, so it floors at
\|f″\|·r = 0.203 — *above* FD's 0.139. The ordering has nothing to do with FD's
1/ε amplification and everything to do with draw multiplicity: FD's two probes
give its common-mode displacement a std of r/√2 = 0.0141 against PSR's r.

## Where FD's floor actually comes from

At the tuned ε* = 0.252, decomposed:

| term | value | note |
|---|---|---|
| differential √2·r·\|f′\|/ε | 0.0432 | the 1/ε amplification |
| common-mode displacement \|f″\|·r/√2 | 0.1434 | **dominant here**; not in B.6.4 |
| truncation ε²\|f‴\|/24 | 0.0602 | |
| quadrature sum | 0.161 | measured 0.139 |

Both of the first two are δ, so δ sets FD's floor. At ε = 0.05 the differential
term is 0.218 and dominates outright — the amplification is what makes small
steps unusable, and it is visible as the inset's left arm.

**B.6.4 is missing a term.** As written it is truncation ⊕ δ/ε and predicts a
floor of 0.0663 at ε*_analytic = 0.201, against a measured 0.139 at 0.252. Adding
the common-mode displacement in quadrature gives 0.161, which brackets the
measurement. At a point this curved (f″ = 10.1) the displacement is the largest
of the three.

## δ off (diagnostic, not a paper figure)

The pre-δ build of the same script is commit `871e1b5`'s cache: PSR 0.0095,
NSR 0.0107, PSR+gate 0.0167, FD 0.1519, FD fixed 0.1752 at N = 10⁶ (note that
build used code-ε, so its FD numbers are at a different physical step — see
NUMBERS.md on the convention change). All three estimators moved in the
predicted direction: FD's floor is essentially unchanged (it was already
δ-limited), while PSR and NSR acquired floors where they previously had none.
