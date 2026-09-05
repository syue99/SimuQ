# SELECTOR_FACTOR2.md — where C_PSR carries the factor 2 (P0-1)

2026-09-05. Cost model of the strategy plane (Figs 10, 14): `tests/build_F_select.py`,
function `cell_costs` (lines 116–138), shared verbatim by `tests/build_F_regimes.py` and
`tests/selector_check.py`.

```python
# NSR: sum over parameters of diam(A_l)^2 — measured (exact spectrum) and
# predicted (certificate diam <= 2*Sum|v|).
N_nsr      += (e[-1] - e[0]) ** 2                       # line 126
N_nsr_pred += (2.0 * sum(abs(c) for c in d.values())) ** 2   # line 127  (2Σ|v|)² = Ω̄_L1²
# PSR with cross-parameter branch reuse: n_j = max_l S_l |c_j| sigma_j,
# S_l = Sum_j' |c_j'| sigma_j'; predicted replaces sigma by sqrt(2).
return N_nsr, 2.0 * nj.sum(), N_nsr_pred, 2.0 * nj_pred.sum()   # line 138
```

The PSR execution count is **2·Σ_j n_j**: the leading 2 is the (−, +) branch pair per
inserted term, i.e. the 2 of (G.1.1) C_PSR = 2Δτ sup_τ Σ_j |v_j|(τ) R, with Δτ·sup absorbed
into n_j = S_l |c_j| σ_j (σ_j = measured per-branch shot std, √2 in the certificate).

## Consistency check 3 (single Pauli, constant v): C_NSR^exact = C_PSR

At the star (p = 2, q = 1), each parameter is one Pauli with |c| = 1:
- N_NSR = Σ_l diam(A_l)² = 2 · 2² = **8** (measured spectrum; the certificate (2Σ|v|)² = 4 per parameter gives the same 8).
- N_PSR = 2 · Σ_j n_j with n_j = σ_j² : with the certificate σ = √2, N_PSR = 2 · 2 · 2 = **8** — an exact tie;
  with the measured σ_j ≈ 1.39, N_PSR = 7.63 and the measured ratio is 10^+0.02 (PSR side by 2%).

Hence the "2.7% agreement at (2,1)" / near-tie statements only hold because the 2 is in the
code. It is. The plane's cell value is log10(N_NSR/N_PSR) computed from exactly these two
numbers (`run_sweep`, mean over 6 seeds).

## The two markers, on the same cost model (`scratchpad/check_fig10.py`, σ from the Fig 10 operating point)

| program | (p, q) | N_NSR | N_PSR | log10 ratio | certificate |
|---|---|---|---|---|---|
| Fig 9 instance θ₁·Z₀Z₁ + θ₂·(X₀+X₁) | (2; q = 1 and 2) | 20.0 | 19.1 | **+0.02** (near-tie, PSR side) | tie |
| plane cell (2,1) | (2,1) | 8.0 | 7.6 | +0.02 (cell mean +0.012) | tie |
| global-θ rewrite θ·(Z₀Z₁+X₀+X₁) | **(1,3)** | 20.0 | 34.5 | **−0.24** (NSR wins 1.7×) | tie |
| plane cell (1,3) | (1,3) | | | −0.018 (cell mean) | |
| plane cell (1,2) (where the circle used to be) | (1,2) | | | +0.013 | |

The rewrite's N_NSR stays 20 because diam(Z₀Z₁+X₀+X₁) = 2√5 (the ZZ term anticommutes with
the X terms; the L1 certificate 2Σ|v| = 6 overestimates it) while PSR's cost grows as
(Σ|c|σ)² ∝ q². So the rewrite is on the NSR side, not "on the crossing".
