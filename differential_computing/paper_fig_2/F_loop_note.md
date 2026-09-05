# F_loop data note (FLOOP_REPLOT §5 — no re-run)

## Fraction of seeds inside tolerance at step 50

"Reliably" is a claim about the band, not the median; this is the number that supports it.

- PSR: 90%  (terminal step, median holds 5 consecutive: 3)
- NSR: 90%  (terminal step, median holds 5 consecutive: 4)
- FD ε=0.1 (best ε — needs θ*): 75%  (terminal step, median holds 5 consecutive: 24)
- FD ε=0.5 (too large: fails): 0%  (terminal step, median holds 5 consecutive: None)
- FD ε=0.05 (too small: unreliable): 45%  (terminal step, median holds 5 consecutive: None)

## Predicted plateau for the best-ε arm vs measured

Best ε = 0.1, |C‴|(soft) = 18.09, δ = 0.02, μ_soft = 0.54.

- b(ε)/μ_soft with b = δ/ε + (ε²/6)|C‴| (central-difference Taylor, as in the builder): 0.757
- same with the spec's (ε²/24) coefficient: 0.385
- truncation-only (ε²/6)|C‴|/μ_soft: 0.056
- measured final median offset: 0.024

The δ/ε term treats the resolution jitter as if it were bias; it is zero-mean per programming, so the full b(ε)/μ_soft overstates the floor (disagreement reported, not scaled to fit — cf. F6's δ/ε floor, where the jitter enters the gradient estimate directly and the floor is real).
