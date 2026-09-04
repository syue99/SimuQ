# TEXT_CHANGES.md — lines that must change as a consequence of the regenerated figures

Fred applies these. Quoted text is from the handover's registry and the current
paper; replacements are what the regenerated data supports. Nothing here is
applied to the paper by me.

## From Figure 8 (P0-0 + P0-5)

### §6.1 — the N^{−1/2} claim
> "at the N^{−1/2} rate over four decades"

No longer true of any series once δ is on every estimator. Measured tail fits
are PSR N^−0.06 and NSR N^−0.11: both **floor**. Suggested replacement scope:
the rate claim holds only for the δ-free model, or must be restated as "until
the setpoint floor is reached". **Fred's call — this is a claim, not a number.**

### §6.1 — PSR's floor vs FD's
> PSR+gate's floor is "an order of magnitude below FD's best"

Measured: PSR+gate 0.2179 against FD ε* 0.1390, i.e. **0.6× — above it**, not an
order of magnitude below. The gate bias (0.0138) is no longer what sets PSR's
floor; the setpoint displacement \|f″\|r = 0.203 is.

### B.6 last paragraph — "bounded weight, and nothing divides it"
Still literally true (no division by ε), but at this operating point the
undivided term is 53% of \|f′\|, so the sentence should not be read as "small".

### C.3 — "never divided by anything … same exposure class as the insertion error"
The insertion error here is 0.0138 and the setpoint displacement is 0.2028; they
are not the same exposure class at this point. Needs rewording or a caveat.

### Remark 4.3 — "FD's setpoint error is amplified"
True, but incomplete: NSR's is amplified by Ω̄ and PSR's appears undivided as
\|f″\|r. Candidate: name all three exposures.

### Table 5 — "residual, as realized" row
Candidate new row per the handover: setpoint exposure FD r/ε, NSR Ω̄r, PSR f″r.
Values at the Fig. 8 point: FD 0.064 at ε*, NSR 0.151 (exact-F′; 0.044 leading
order), PSR 0.203.

### Fig. 8 caption
- ε* = 0.17 → **0.252** (retuned with δ on, in the paper's ε convention).
- The legend slopes N^−0.48 / N^−0.50 → N^−0.06 / N^−0.11, or drop the slope
  annotations, since a fit through a floored tail is not a rate.
- The caption should state that δ is on for all six series, since that is what
  now sets three of the six floors.

### C.3 — the ε* value
> "FD ε* = 0.17 (tuned at N=10⁴)"

→ **0.252**. Note this is both a retune (δ on) and a convention change
(θ ± ε/2 ÷ ε); the old 0.17 was in the builder's θ ± ε ÷ 2ε convention, where the
same physical step reads as half the number.

### App. B.6.4 — the analytic floor as a prediction
The closed form gives ε*_analytic = 0.201 and floor 0.0663; measured are 0.252
and 0.139. The form is the right shape (the inset's grey curve tracks the sweep's
left arm) but is a factor ~2 low on the floor at this point, because f‴ is not
constant over the swept range. Suggest presenting it as the scaling, not the
value.

### App. D / §6.1 — the NSR "no floor at M=∞" statement
With δ on there is a floor at M=∞ too: 0.1496, set by the setpoint draw and not
by truncation. The certifiable-vs-uncertifiable contrast survives (NSR's is
Ω̄-bounded, FD's is not), but "no floor" does not.

## Pending, not yet regenerated
Figs 2, 9 (P0-0), Fig 10/14 (P0-1/P0-8), Figs 7/13 (P0-2), Fig 12 (P0-3),
Fig 15b (P0-6), Fig 6 (P0-7).
