# DELTA_NOISE.md — the setpoint rule as implemented (P0-0)

2026-09-04. Figure 8 (`tests/build_F6.py`) is the first figure regenerated under
the rule. Figs 2 and 9 follow and will be appended here.

## What a setpoint draw is attached to

`r = 0.02`, zero-mean Gaussian, frozen across the shots of the execution it
belongs to. The rule fixes how many independent draws each estimator gets, and
that is what decides which strategies floor.

**Model used: `per_programming`.** Each program execution dials its own
coefficient and draws its own δ. FD gets 2 per estimate (its two probes), PSR
2m = 96 (one per branch execution), NSR N (one per execution). Selected by
`DELTA_MODEL` in `build_F6.py`.

FD is the only estimator that cannot average the draw away, because it programs
exactly twice per estimate no matter how large N is. That is why **δ floors FD
and only FD**, which is the figure's claim.

### The alternative, and why it is not used

`per_value` attaches δ to the coefficient requested rather than to the act of
requesting it, so programs that dial the same value share a draw. It is a
defensible reading of "setpoint error", but it does not describe this device and
it breaks both shift rules:

- PSR's 96 branches all dial the source coefficient, so they would share ONE
  draw and PSR would floor at |f″|r with no averaging at all.
- NSR's 48 distinct shifted setpoints would each keep their own draw, and
  nothing in the estimate averages them, so NSR would floor at ≈ Ω̄r|f′|/√3.

Measured under it at this operating point: PSR 0.218, NSR 0.150, against FD's
0.139 — i.e. both shift rules worse than FD. That is a property of the model,
not of the strategies, and it is why the figure does not use it.

### Measured floors (RMSE at N = 10⁶, 100 seeds)

| series | floor | % of \|f′\| | set by |
|---|---|---|---|
| PSR | 0.0274 | 7.1% | insertion bias 0.0138 ⊕ residual displacement |f″|r/√96 = 0.021 |
| PSR + gate | 0.0274 | 7.1% | same |
| NSR M=∞ | 0.0102 | 2.7% | **no floor** — still on N^−0.50 at 10⁶ |
| NSR M=5 | 0.0134 | 3.5% | truncation 0.0123 (certifiable, Lemma D.5) |
| FD @ ε*=0.252 | 0.1390 | 36.1% | **δ** — see the decomposition below |
| FD @ ε=0.05 | 0.2687 | 69.8% | δ/ε at an untuned step |

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
