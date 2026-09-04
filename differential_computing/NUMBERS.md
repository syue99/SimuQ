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

Values under the model actually used for the figure (`DELTA_MODEL=per_programming`).

| quantity | old (text) | new | note |
|---|---|---|---|
| FD ε* | **0.17** | **0.252** | retuned with δ on AND re-expressed in the paper's θ±ε/2 convention |
| PSR tail slope | N^−0.48 | **N^−0.33** | PSR now reaches its gate-bias floor inside the window, so the tail fit bends |
| NSR M=∞ tail slope | N^−0.50 | **N^−0.50** | unchanged — NSR still rides the rate to 10⁶ |
| PSR floor at N=10⁶ | 0.0095 (δ off) | **0.0274** | gate bias 0.0138 ⊕ residual displacement \|f″\|r/√96 = 0.021 |
| NSR M=∞ at N=10⁶ | 0.0107 | **0.0102** | no setpoint floor under per_programming |
| NSR M=5 | 0.0159 | **0.0134** | truncation floor 0.0123, as before |
| PSR+gate floor | 0.0167 | **0.0274** | |
| FD @ ε* floor | 0.1519 | **0.1390** | |
| FD fixed ε=0.05 floor | 0.1752 | **0.2687** | 0.05 is now half the physical step it was |
| PSR+gate vs FD ε* | "an order of magnitude below" | **5.1×** below | 0.0274 vs 0.1390 |

### New quantities (P0-5, B.6.4)

| quantity | value |
|---|---|
| f′ = C′(θ₀) | −0.3850 |
| f″ = C″(θ₀) | +10.140 |
| f‴ = C‴(θ₀) | +22.72 (h=0.05; +21.27 at h=0.08) |
| ε*_analytic = (24\|f′\|r/\|f‴\|)^{1/3} | **0.201** vs tuned **0.252** |
| B.6.4 floor 0.60\|f‴\|^{1/3}(\|f′\|r)^{2/3} | **0.0663** vs measured **0.1390** |
| FD differential term √2r\|f′\|/ε at ε* | 0.0432 |
| FD common-mode displacement \|f″\|r/√2 | **0.1434** — dominant, and absent from B.6.4 |
| FD truncation ε²\|f‴\|/24 at ε* | 0.0602 |
| PSR displacement, per_programming \|f″\|r/√(2m) | 0.0207 |
| PSR displacement, per_value \|f″\|r | 0.2028 (52.7% of \|f′\|) |

## Findings for the text

**1. The setpoint model had to be pinned, and it decides the figure.** All 2m PSR
branches dial the same source coefficient, so whether δ attaches to the *value*
or to each *programming* changes PSR's floor by √(2m) ≈ 10×. Under per_value PSR
floors at 0.203, above FD; under per_programming at 0.027, below it. The figure
ships per_programming, and DELTA_NOISE.md records both. Fred should confirm in
writing, as the handover requires.

**2. B.6.4 omits the common-mode displacement.** FD's two probes have
independent draws, so their midpoint moves by (δ⁺+δ⁻)/2 and the secant estimates
f′ there, adding \|f″\|r/√2 = 0.143. At θ₀ = 1.940 that is larger than either the
δ/ε term (0.043) or truncation (0.060), which is why the analytic curve sits a
factor ~2 below the sweep near its minimum. Adding it in quadrature gives 0.161
against a measured 0.139.

**3. δ is what floors FD, and only FD.** FD programs twice per estimate whatever
the setpoint model, so it cannot average the draw away; PSR averages over 2m
branches and NSR over N executions. FD's floor is identical (0.1390) under every
model tested.

**4. "An order of magnitude below FD's best" is now 5.1×.** PSR's floor is no
longer the gate bias alone (0.0138 → 10.1×) but gate bias ⊕ residual setpoint
displacement (0.0274 → 5.1×).

## ε convention

The handover's ground rule (§0) is that figures move to the paper's θ ± ε/2, ÷ε
convention while the text stays. `fd_est` and `fd_floor_pred` now probe θ ± ε/2
and divide by ε, so **every ε number on the figure means half the physical
separation the old ones did**. The inset grid [0.02, 1.2] and the fixed step 0.05
are unchanged as *labels* and therefore halved as *steps*. The tuned ε* lands at
0.252, so the text's 0.17 cannot be preserved — per §0 that is a text edit, not
mine to make.
