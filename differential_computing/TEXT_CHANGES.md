# TEXT_CHANGES.md — lines that must change as a consequence of the regenerated figures

Fred applies these. Quoted text is from the handover's registry and the current
paper; replacements are what the regenerated data supports. Nothing here is
applied to the paper by me.

## From Figure 8 (P0-0 + P0-5)

Under `DELTA_MODEL=per_programming` (see DELTA_NOISE.md) most of §6.1 survives.
What still has to change:

### C.3 / Fig. 8 caption — ε*
> "FD ε* = 0.17 (tuned at N=10⁴)"

→ **0.252**. Two causes at once: retuned with δ on every estimator, and
re-expressed in the paper's θ ± ε/2 ÷ ε convention (the old 0.17 was in the
builder's θ ± ε ÷ 2ε convention, where the same physical step reads as half).

### Fig. 8 legend — slopes
> N^−0.48 (PSR), N^−0.50 (NSR M=∞)

→ PSR **N^−0.33**, NSR **N^−0.50**. NSR is unchanged; PSR's fit bends because it
now reaches its floor inside the fitted window. Either update or drop the PSR
slope, since a fit through a floored tail is not a rate.

### §6.1 — PSR's floor against FD's
> PSR+gate's floor is "an order of magnitude below FD's best"

→ **5.1×** below (0.0274 vs 0.1390). PSR's floor is no longer the insertion bias
alone: it is that bias (0.0138) combined with the residual setpoint displacement
that survives averaging over the 2m branches (0.021).

### Fig. 8 caption — δ is on
The caption should say δ is applied to every estimator, since that is what now
sets FD's floor and PSR's.

### App. B.6.4 — a missing term
The closed form is truncation ⊕ δ/ε and predicts floor 0.0663 at ε* = 0.201;
measured 0.1390 at 0.252. FD's two probes draw independently, so their midpoint
displaces by (δ⁺+δ⁻)/2 and contributes \|f″\|r/√2 = 0.143 — larger than either
term already in the formula at this operating point. Suggested: add the
displacement term, or present B.6.4 as the scaling rather than the value.

### C.3 — "never divided by anything … same exposure class as the insertion error"
Still defensible under per_programming: PSR's residual displacement (0.021) and
its insertion bias (0.014) are the same order. Worth stating that the
displacement averages over branches, since that is why it is small.

### Remark 4.3 — "FD's setpoint error is amplified"
True and now the load-bearing sentence: FD is the only estimator that cannot
average the setpoint draw away, because it programs exactly twice per estimate.

## Pending, not yet regenerated
Figs 2, 9 (P0-0), Fig 10/14 (P0-1/P0-8), Figs 7/13 (P0-2), Fig 12 (P0-3),
Fig 15b (P0-6), Fig 6 (P0-7).
