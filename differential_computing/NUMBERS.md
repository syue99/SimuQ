# NUMBERS.md — old → new, per figure

Script/commit for every entry below: `tests/build_F6.py` at the commit carrying
this file. Old values are the pre-δ build (`871e1b5`) and the paper text as
quoted in the handover's §3 registry.

## Figure 8 (P0-0 setpoint noise on all six series, P0-5 analytic inset)

### Unchanged, confirmed

| quantity | value |
|---|---|
| θ₀ | 1.940 |
| ∇C_device(θ₀) | −0.3850 |
| Ω̄ = 2T | 10.0 |
| T, T₂* | 5.0 µs, 33.3 µs |
| M cap | 5 |
| excluded mass p_out | 0.0337 → **3.4%** (analytic; the 24-mode sampler tail is 0.0255) |
| N grid, seeds | 10²…10⁶, 100 reps/point |
| inset grid | 24 points, geometric over [0.02, 1.2], N = 10⁴ |
| ε_ins | 10⁻³ |

### Moved — text edits needed

| quantity | old (text) | new | note |
|---|---|---|---|
| FD ε* | **0.17** | **0.252** | retuned at N=10⁴ with δ on, AND re-expressed in the paper's θ±ε/2 convention |
| PSR tail slope | N^−0.48 | **N^−0.06** | PSR now floors; the fit is over a floored tail and is no longer a rate |
| NSR M=∞ tail slope | N^−0.50 | **N^−0.11** | same |
| PSR floor at N=10⁶ | none (0.0095, still descending) | **0.2178** | = \|f″\|·r displacement |
| NSR M=∞ floor | none (0.0107) | **0.1496** | setpoint floor |
| NSR M=5 floor | 0.0159 (truncation only) | **0.1495** | setpoint floor now dominates the 0.0123 truncation bias |
| PSR+gate floor | 0.0167 | **0.2179** | displacement swamps the 0.0138 gate bias |
| FD @ ε* floor | 0.1519 | **0.1390** | different physical step (convention) and δ unchanged for FD |
| FD fixed ε=0.05 floor | 0.1752 | **0.2687** | ε=0.05 is now half the physical step it was |

### New quantities (P0-5, B.6.4)

| quantity | value |
|---|---|
| f′ = C′(θ₀) | −0.3850 |
| f″ = C″(θ₀) | +10.140 |
| f‴ = C‴(θ₀) | +22.72 (h=0.05; +21.27 at h=0.08, 6% stencil spread) |
| ε*_analytic = (24\|f′\|r/\|f‴\|)^{1/3} | **0.201** vs tuned 0.252 |
| FD floor 0.60\|f‴\|^{1/3}(\|f′\|r)^{2/3} | **0.0663** vs measured 0.1390 |
| PSR displacement \|f″\|·r | **0.2028** — **52.7% of \|f′\|** |
| NSR setpoint floor, leading order Ω̄r\|f′\|/√3 | 0.0445 |
| NSR setpoint floor, exact F′ | **0.1506** vs measured 0.1496 |

## Three findings that contradict the text

**1. PSR no longer rides N^{−1/2}; it floors above FD.** With one shared δ per
estimate (the handover's default rule) PSR estimates the exact gradient *at the
realized setpoint*, so its error floors at \|f″\|·r = 0.203. Measured 0.218 at
N = 10⁶, against FD's tuned floor of 0.139. §6.1's "an order of magnitude below
FD's best" now reads **0.6×, i.e. above it**. Consistency check 2 of the
handover fails: PSR+gate's floor is not ≥10× below FD ε*'s floor.

**2. B.6's "second-order effect" wording fails at this operating point.**
\|f″\|·r = 0.2028 is 52.7% of \|f′\| = 0.3850, not ≪ it. The handover asked for
this to be reported before any caption is written — this is that report.

**3. The leading-order NSR floor is 3.4× low here.** Ω̄r\|f′\|/√3 assumes
F′(h_κ) ≈ f′(θ₀); the exact sum over the sampler's shifts gives 0.1506, matching
the measurement. The formula is fine as a leading-order bound, but the number
0.044 should not appear as a prediction for this figure.

None of these were tuned away. θ₀, seeds, grids, r, and the regime are untouched.

### Why this operating point makes it worse

θ₀ = 1.940 was selected (C.3) as the point *maximizing FD's predicted floor* —
i.e. deliberately sharp. Sharpness is \|f″\|, \|f‴\|, and those are exactly what
set PSR's displacement and NSR's exact-F′ floor. The point chosen to make FD look
bad now penalizes the shift rules more than FD. Moving θ₀ is explicitly out of
scope for me; flagging that the choice interacts with P0-0.

## ε convention

The handover's ground rule (§0) is that figures move to the paper's θ ± ε/2, ÷ε
convention while the text stays. `fd_est` and `fd_floor_pred` now probe θ ± ε/2
and divide by ε, so **every ε number on the figure means half the physical
separation the old ones did**. The inset grid [0.02, 1.2] and the fixed step 0.05
are unchanged as *labels* and therefore halved as *steps*. The tuned ε* lands at
0.252, so the text's 0.17 cannot be preserved — per §0 that is a text edit, not
mine to make.
