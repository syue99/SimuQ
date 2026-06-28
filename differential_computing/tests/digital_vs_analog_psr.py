"""
digital_vs_analog_psr.py — gradient accuracy: DIGITAL PSR (exact) vs ANALOG PSR
(τ-sampling) vs FD.  Log-log RMSE vs shot budget.

Panel A — DIGITAL (gate) PSR is EXACT.  Task: |ψ(θ)>=e^{-iθX/2}|0>, <Z>=cos θ.
  The shift rule ∂/∂θ = (1/2)[f(θ+π/2) − f(θ−π/2)] is EXACT (no bias, no ε, no
  τ-integral) — 2 evaluations, fixed 1/2 prefactor.  RMSE ∝ N^{-1/2} with no
  floor.  FD at fixed ε has a bias floor.  Digital PSR is the gold standard.

Panel B — ANALOG PSR pays the τ-integral.  Task: H(θ)=θ·Z0+X0 evolved for T
  (Z0, X0 don't commute → the kick time matters), sharp T + FLOORED ε.  Analog
  PSR (Algorithm 1) is unbiased but has a τ-sampling floor ∝ 1/√n_sample; FD's
  bias floor is fixed at floored ε.  PSR (enough τ-samples) converges below.

The contrast: digital PSR is unambiguously better than FD (exact, cheap); the
analog generalization to continuous non-commuting evolution costs τ-sampling
variance, but still beats floored-ε FD on sharp landscapes with enough samples.

Saves figures/digital_vs_analog_psr.png.

Run:  conda run -n qec_pg python differential_computing/tests/digital_vs_analog_psr.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sympy as sp
import qutip as qp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simuq import QSystem, Qubit
from observable_program_generator import observable_program_generator
from noisy_qutip import NoisyQuTiPRunner

BUDGETS = np.array([100, 300, 1000, 3000, 10000, 30000, 100000, 300000])
R = 8000


# ── Panel A: digital gate task |ψ(θ)>=e^{-iθX/2}|0>, <Z> = cos θ ──────────────
THETA_D = 0.7


def cosf(phi):
    return float(np.cos(phi))                     # exact <Z>(phi)


def sample_pm1(fval, n, rng, R):
    p = 0.5 * (1 + np.clip(fval, -1, 1))
    return 2.0 * rng.binomial(int(max(1, n)), p, size=R) / max(1, int(n)) - 1.0


def digital_psr_rmse(N, truth, rng):
    n = N // 2
    fp = sample_pm1(cosf(THETA_D + np.pi / 2), n, rng, R)
    fm = sample_pm1(cosf(THETA_D - np.pi / 2), n, rng, R)
    est = 0.5 * (fp - fm)                          # EXACT shift rule, prefactor 1/2
    return float(np.sqrt(np.mean((est - truth) ** 2)))


def fd_gate_rmse(N, eps, truth, rng):
    n = N // 2
    fp = sample_pm1(cosf(THETA_D + eps), n, rng, R)
    fm = sample_pm1(cosf(THETA_D - eps), n, rng, R)
    est = (fp - fm) / (2 * eps)
    return float(np.sqrt(np.mean((est - truth) ** 2)))


# ── Panel B: analog task H=θ·Z0+X0, <Z0>, sharp T + floored ε ─────────────────
T = 6.0
X_VAL = 0.5
EPS_MIN = 0.2
OBS = qp.tensor(qp.sigmaz(), qp.qeye(2))
PSI0 = qp.tensor(qp.basis(2, 0), qp.basis(2, 0))


def H_param():
    x = sp.Symbol("x")
    qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    return x * q[0].Z + q[0].X, "x"


def H_eval(x):
    Hp, var = H_param()
    return Hp.set_parameterizedHam({var: float(x)})


def analog_exact(runner, eps=1e-3):
    expfn = runner.make_expectation_fn(PSI0, OBS)
    f = lambda x: expfn([[H_eval(x), T]])
    return (f(X_VAL + eps) - f(X_VAL - eps)) / (2 * eps)


def analog_pool(runner, pool_size):
    Hp, var = H_param()
    np.random.seed(123)
    progs = observable_program_generator(Hp, T, n_sample=pool_size,
                                         n_repetition=1, diff_var=var, value=X_VAL)
    expfn = runner.make_expectation_fn(PSI0, OBS)
    H_tot, ug, _ = progs[0]
    b = len(H_tot) // 2
    em = np.array([expfn(H_tot[2 * i]) for i in range(b)])
    ep = np.array([expfn(H_tot[2 * i + 1]) for i in range(b)])
    return em, ep, float(ug)


def analog_psr_rmse(em, ep, ug, n_sample, N, truth, rng):
    P = len(em)
    n_per = int(max(1, round(N / (2 * n_sample))))
    idx = rng.integers(0, P, size=(R, n_sample))
    fm = 2.0 * rng.binomial(n_per, 0.5 * (1 + np.clip(em[idx], -1, 1))) / n_per - 1
    fp = 2.0 * rng.binomial(n_per, 0.5 * (1 + np.clip(ep[idx], -1, 1))) / n_per - 1
    est = (T / n_sample) * ug * np.sum(fm - fp, axis=1)
    return float(np.sqrt(np.mean((est - truth) ** 2)))


def fd_analog_rmse(runner, eps, N, truth, rng):
    expfn = runner.make_expectation_fn(PSI0, OBS)
    fp_ex = expfn([[H_eval(X_VAL + eps), T]])
    fm_ex = expfn([[H_eval(X_VAL - eps), T]])
    n = N // 2
    fp = sample_pm1(fp_ex, n, rng, R); fm = sample_pm1(fm_ex, n, rng, R)
    est = (fp - fm) / (2 * eps)
    return float(np.sqrt(np.mean((est - truth) ** 2)))


def main():
    rng = np.random.default_rng(0)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=150)

    # Panel A
    truthA = -np.sin(THETA_D)
    dp = [digital_psr_rmse(N, truthA, rng) for N in BUDGETS]
    fdA1 = [fd_gate_rmse(N, 0.1, truthA, rng) for N in BUDGETS]
    fdA2 = [fd_gate_rmse(N, 0.8, truthA, rng) for N in BUDGETS]
    axA.loglog(BUDGETS, dp, "o-", color="#2ca02c", lw=2.6, label="digital PSR (exact)")
    axA.loglog(BUDGETS, fdA1, "s--", color="#d62728", lw=1.8, label="FD ε=0.1")
    axA.loglog(BUDGETS, fdA2, "s--", color="#9467bd", lw=1.8, label="FD ε=0.8")
    axA.loglog(BUDGETS, dp[0] * (BUDGETS / BUDGETS[0]) ** -0.5, "k:", lw=1,
               label=r"$\propto N^{-1/2}$")
    axA.set_xlabel("total shots / gradient  N"); axA.set_ylabel("gradient RMSE")
    axA.set_title("(A) DIGITAL (gate) PSR — exact, 2 evals, no floor")
    axA.legend(frameon=False, fontsize=8)

    # Panel B
    runner = NoisyQuTiPRunner(2, noise=None)
    truthB = analog_exact(runner)
    em, ep, ug = analog_pool(runner, 2000)
    ap256 = [analog_psr_rmse(em, ep, ug, 256, N, truthB, rng) for N in BUDGETS]
    ap1024 = [analog_psr_rmse(em, ep, ug, 1024, N, truthB, rng) for N in BUDGETS]
    fdB = [fd_analog_rmse(runner, EPS_MIN, N, truthB, rng) for N in BUDGETS]
    fdB2 = [fd_analog_rmse(runner, 0.4, N, truthB, rng) for N in BUDGETS]
    axB.loglog(BUDGETS, ap256, "o-", color="#17becf", lw=2.2, label="analog PSR n=256")
    axB.loglog(BUDGETS, ap1024, "o-", color="#1f77b4", lw=2.6, label="analog PSR n=1024")
    axB.loglog(BUDGETS, fdB, "s--", color="#d62728", lw=1.8, label=f"FD ε={EPS_MIN} (=floor)")
    axB.loglog(BUDGETS, fdB2, "s--", color="#9467bd", lw=1.8, label="FD ε=0.4")
    axB.set_xlabel("total shots / gradient  N"); axB.set_ylabel("gradient RMSE")
    axB.set_title(f"(B) ANALOG PSR — τ-sampling cost (T={T}, floored ε≥{EPS_MIN})")
    axB.legend(frameon=False, fontsize=8)

    fig.suptitle("Gradient accuracy, log-log: digital PSR is exact (2 evals, no ε); "
                 "analog PSR pays a τ-sampling cost\nfor continuous non-commuting "
                 "evolution but still beats floored-ε FD with enough samples",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "figures", "digital_vs_analog_psr.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"Panel A (digital gate):  exact grad {truthA:+.4f}")
    print(f"  digital PSR : {dp[0]:.4f} → {dp[-1]:.4f}  (N^-1/2, no floor)")
    print(f"  FD ε=0.1    : {fdA1[0]:.4f} → {fdA1[-1]:.4f}")
    print(f"  FD ε=0.8    : {fdA2[0]:.4f} → {fdA2[-1]:.4f}  (bias floor)")
    print(f"Panel B (analog T={T}, floored ε):  exact grad {truthB:+.4f}")
    print(f"  analog PSR n=1024 : {ap1024[0]:.4f} → {ap1024[-1]:.4f}")
    print(f"  FD ε=0.2 (floor)  : {fdB[0]:.4f} → {fdB[-1]:.4f}  (bias floor)")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
