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

## From Figure 9 (P0-0 + P1-2) — build 2026-09-04, T = 2.5, per-change δ rule, SGD schedule

### C.3 — instance
> "θ* = (1,1); start (0.802, 1.251); box [0.2,1.4]²; w = 0.25; T = 0.8 µs; T₂* = 5.33 µs; 32 τ-samples"

→ start **(1.010, 0.680)**; **T = 2.5 µs; T₂* = 16.7 µs**. θ*, box, w, τ-samples unchanged.
Add: the box constrains the iterate, not the programs (NSR's shifts and FD's probes leave it).

### C.3 — optimizer
> "100 steps" → **50 steps**; add: gradient descent with **η_t = η₀/(1 + t/20), η₀ = 1.4/μ_stiff
= 0.064**, and the reported iterate is the **tail average θ̄_t = mean(θ_{⌈t/2⌉..t})**.
> "ε ∈ {0.15, 0.7, 0.04}" → **ε ∈ {0.1, 0.5, 0.05}** (paper's θ ± ε/2 convention; the best
step chosen retrospectively from {0.1, 0.15, 0.2, 0.3, 0.5}).

### Fig. 9 caption — convergence
> "converged at 10 / 34" (PSR / NSR)

→ **PSR 3, NSR 4** (first step from which the median of ‖θ̄_t − θ*‖ stays inside 0.03),
both then hold to step 50 and keep descending (0.012 / 0.011 at 50, centred on θ*).
FD: the retrospective best step holds only from **step 24** and floors at 0.024; the
deployable steps never enter tolerance (0.097 too large, 0.039 too small, the latter with
an IQR of [0.012, 0.133]). The two deployable steps are *biased* (mean offsets 0.097 and
0.051), so averaging cannot rescue them — that is the sentence the figure now supports.

### Fig. 9 caption — "deliberately ill-conditioned"
Goes (Fred's plan). κ = 41 at the new T.

### Fig. 9 caption — δ is on
Say the setpoint draw is applied per step to every estimator (FD two draws per coefficient,
PSR one shared, NSR per execution on the shifted coefficient), r = 0.02.

### Fig. 9 y-label
"(median ± IQR)" → "‖θ̄_t − θ*‖ (median and IQR)" — done in the figure; the bar must be
explained in the caption (tail-averaged iterate).

### §6 prose on NSR in the loop
Any sentence saying NSR converges faster/cleaner than PSR in the loop must go: both hold
tolerance from step 3–4 and their δ exposure is the same (both inherit ∇C(θ+δ) through the
residual measurement). Their difference (0.012 vs 0.011) is noise.

### 6.x / Table 1 if they quote Fig 9's budget
The shift rules actually spend 4800 executions per gradient (residual charged twice in
the accounting), FD 6000. Either quote "≤ 6000" or ask for the parity rerun.

## From Figure 2 (P0-4 + P0-0) — build 2026-09-05

### C.3 — probes
> "ε ∈ {0.18, 0.25, 0.32}" → **ε ∈ {0.36, 0.50, 0.64}** (paper's θ ± ε/2 convention; the
physical probes are unchanged).

### Fig. 2 caption — the stronger statement (B.6.4 at this anchor)
Offer: "At this anchor the best finite-difference step is ε* ≈ 0.11, and even there the
setpoint error leaves an RMSE of 28% of |∇C| (B.6.4 gives 32%); at ε = √2·r = 0.028 the
one-sigma slope error equals |∇C| itself (purple cone)." The drawn secants carry their own
setpoint draws and stay wrong-signed (97–100% of draws).

### Fig. 2 caption / C.3 — notation and labels
"T4" is gone from the figure; C_noisy → C_device everywhere in the figure. Regime 0.5 stays.

## Pending, not yet regenerated
Fig 10/14 (P0-1/P0-8), Figs 7/13 (P0-2), Fig 12 (P0-3), Fig 15b (P0-6), Fig 6 (P0-7).
