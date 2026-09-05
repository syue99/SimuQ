# TEXT_CHANGES.md — lines that must change as a consequence of the regenerated figures

Fred applies these. Quoted text is from the handover's registry and the current paper;
replacements are what the regenerated data supports. Nothing here is applied to the paper.

## From Figure 8 (P0-0 + P0-5) — build 2026-09-04, θ₀ = 1.757, per-change δ rule

### C.3 — operating point
> "θ₀ = 1.940; ∇C_device = −0.385" and "was selected by scanning … for the point maximizing FD's predicted floor"

→ **θ₀ = 1.757, ∇C_device = −1.449**, and the selection rule becomes: *the stationary
point of ∇C_device (C″ = 0) within the coefficient window whose headroom cap gives M = 5*.
Rationale sentence to offer: at C″ = 0 every estimator's setpoint displacement is second
order, so the comparison isolates what each strategy does with the draw (FD divides by ε,
PSR shares one draw, NSR re-dials).

### C.3 / Fig. 8 caption — ε*
> "FD ε* = 0.17 (tuned at N=10⁴)"

→ **0.21** (0.208; paper's θ ± ε/2 convention). B.6.4's closed form gives 0.217.

### Fig. 8 legend — slopes
> N^−0.48 (PSR), N^−0.50 (NSR M=∞)

→ PSR **N^−0.33**, NSR **N^−0.51**. PSR's fit bends because it reaches its floor above
N ≈ 10⁵; either quote it as "floors at N ≳ 10⁵" or drop the PSR slope.

### §6.1 — "at the N^{−1/2} rate over four decades"
Holds for **NSR** (both M=∞ and M=5). PSR leaves the rate at N ≈ 10⁵ (its floor). Reword
to NSR only, or "PSR to N ≈ 10⁵, NSR through 10⁶".

### §6.1 — PSR's floor against FD's
> PSR+gate's floor is "an order of magnitude below FD's best"

→ **7.5×** below (0.024 vs 0.180). If the sentence is kept as "an order of magnitude", it is
true of **NSR** (19×), not PSR.

### §6.1 / Table 5 / B.6 — what floors PSR
The text attributes PSR's floor to the insertion bias ≤ C_PSR·ε_ins. At this operating point
the exact gate bias is 0.0037 and the measured floor is 0.024: PSR floors at its
**second-order setpoint displacement**, RMS (√3/2)|f‴|r² = 0.024 (mean r²f‴/2 = 0.014),
because its one shared draw is never averaged. Candidate Table 5 "setpoint exposure" row:
FD √2·r|f′|/ε; PSR |f″|r (first order) + (√3/2)|f‴|r² (second order); NSR none in the
limit (fresh draw per execution), residual bias r²f‴/2.

### B.6 last paragraph — "bounded weight, and nothing divides it"
Still true for PSR. Add that the bounded weight is a displacement of the evaluation point,
first order in f″, and that the figure sits at f″ = 0 where it is second order.

### B.6.4
As written (truncation ⊕ δ/ε) it is now the right *form* at f″ = 0 (the common-mode term
|f″|r/√2 vanishes) but overestimates the value: 0.231 predicted vs 0.180 measured (0.197
shot-free Monte Carlo). The truncation term ε²f‴/24 is a Taylor estimate and ε*·ω ≈ 1.6
here. Present B.6.4 as the scaling that fixes ε*, not as the floor's value.

### C.3 — "never divided by anything … same exposure class as the insertion error"
Defensible: PSR's displacement (0.024) and insertion bias (0.004) are both bounded, neither
divided by a step. Note that at f″ ≠ 0 the displacement is first order and can dominate.

### Remark 4.3 — "FD's setpoint error is amplified"
True, and it is the whole of FD's floor here: the differential term √2·r|f′|/ε* = 0.197
accounts for the measured 0.180 by itself. NSR's is *not* amplified under the per-change
rule (it averages), so do not add "NSR's is too, by Ω̄".

### Fig. 8 caption — δ is on
State that the setpoint draw is applied to every estimator: two draws for FD (one per
probe), one shared draw for PSR, a fresh draw per execution for NSR (a draw is taken when
the programmed value changes), r = 0.02, frozen across the shots of a programmed value.

### App. D.3 / Table 3 — M = 5, 3.4% rejected
Unchanged (θ₀ = 1.757 keeps M = 5). s_max = θ₀ is still provisional.

### 6.3 / Table 1 gate-bias quote
6.3 quotes C3's 0.028 per-component bias at C3's point. Fig. 8's own PSR+gate bias is now
0.0037 (was 0.0138 at θ₀ = 1.940); if the text quotes Fig. 8's, update it.

## Pending, not yet regenerated
Figs 2, 9 (P0-0), Fig 10/14 (P0-1/P0-8), Figs 7/13 (P0-2), Fig 12 (P0-3), Fig 15b (P0-6),
Fig 6 (P0-7).
