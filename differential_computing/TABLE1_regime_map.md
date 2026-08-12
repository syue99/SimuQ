# Table 1 — The Regime Map (framework paper spine)

**Symbols.**
`θ` parameter · `∇C` target = gradient of the *noisy* device landscape ·
`T` evolution time · `Γ=1/T₂` dephasing rate ·
`δ` control setpoint-error (std) · `ε` FD step · `Δt` control time-slice ·
`N` shots · `ε_g` target gradient RMSE · `m` generator terms an extensive `θ` touches ·
`v_j=∂u_j/∂θ` · `A=Σ_j v_j H_j` (tangent) · `diam(A)=λ_max−λ_min` ·
`K=(T/2π)·diam(A)` (Nyquist bandwidth).
Cells: **✓** holds · **✗** fails · **✓†** numerically holds, proof open · **?** open ·
superscript `s` = stochastic Nyquist.

| Axis | Kick PSR | Nyquist | FD |
|---|:--:|:--:|:--:|
| generator req. | Pauli + synth. | any Hermitian | none |
| primitive | native + digital synth. | native op. | native op. |
| compilation | zones + transport + gate | waveform only | waveform only |
| executions / grad | `O(m)` | `O(N)` / `O(1)ˢ` | `O(1)` |
| bias | `0` | `O(K/N)` / `0ˢ` | `O(δ^{2/3})` |
| variance | `O(T²)` | `O(K²)` | `O(1/ε²)` |
| extensive `θ` (`m` terms) | `O(m)` exec | `O(1)` dir | `O(1)` |
| fine param, `Δt→0` | ✓ | ✗ `(s∝1/Δt)` | ✓ |
| unbiased on noisy `∇C` | ✓ | ✓† | ✗ |
| slow/coherent-error suppr. | ✓ | ? | ✗ |
| **shots `N→ε_g`** | `O(T²e^{2ΓT}/ε_g²)` | `O(K²e^{2ΓT}/ε_g²)` | `O(e^{2ΓT}/ε_g²)` |
| **δ-robust (no floor)** | ✓ | ✓ | ✗ `(ε_g≳δ^{2/3})` |

**Open cells** (blocking, FRAMEWORK_OUTLINE §Open questions): `unbiased on noisy ∇C`
Nyquist = ✓† (numerics: `noisy_nyquist_vs_fd_kick`; proof pending), and
`slow-error suppr.` Nyquist = ? (check pending). FD `δ^{2/3}` floor verified
(measured `δ^{0.70}`, `δ∈[0.005,0.15]`).
