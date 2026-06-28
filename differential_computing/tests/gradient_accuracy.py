"""
gradient_accuracy.py — the core claim: PSR estimates the gradient more ACCURATELY
than finite difference, single-point evaluation (no optimization loop).

Setup: 1-qubit H(x) = x·Z0 + X0 (single chain-rule term, M=1), observable <Z0>,
evaluated at x=0.7.  Ground truth = the exact gradient (fine-ε FD on the exact,
noiseless expectation).  Metric: gradient RMSE = sqrt(bias²+variance) vs exact,
over many shot realizations, at EQUAL total shot budget for PSR and FD.

The honest claim depends on landscape SHARPNESS (set by evolution time T):
  Panel A — SMOOTH landscape (short T): FD's U-shaped ε tradeoff has a tuned
    sweet spot that can MATCH or beat PSR (large ε → low variance, low bias).
  Panel B — SHARP landscape (long T): large ε ALIASES → bias; small ε → variance;
    FD's whole ε curve sits ABOVE PSR.  PSR is more accurate at ANY ε — no
    tunable step rescues FD.  This is the regime where the claim holds.

Both panels: RMSE vs ε (FD) at a fixed shot budget, with PSR (no ε) as a line.
Saves figures/gradient_accuracy.png.

Run:  conda run -n qec_pg python differential_computing/tests/gradient_accuracy.py
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

X_VAL = 0.7
OBS = qp.tensor(qp.sigmaz(), qp.qeye(2))
PSI0 = qp.tensor(qp.basis(2, 0), qp.basis(2, 0))


def H_param():
    x = sp.Symbol("x")
    qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    return x * q[0].Z + q[0].X, "x"


def H_eval(x):
    Hp, var = H_param()
    return Hp.set_parameterizedHam({var: float(x)})


def exact_grad(runner, T, eps=1e-3):
    """Ground-truth gradient: fine-ε FD on the exact expectation."""
    expfn = runner.make_expectation_fn(PSI0, OBS)

    def f(x):
        return expfn([[H_eval(x), T]])
    return (f(X_VAL + eps) - f(X_VAL - eps)) / (2 * eps)


def psr_terms(runner, T):
    """PSR weights + exact per-branch expectations (n_sample=1 → 2 branches)."""
    Hp, var = H_param()
    np.random.seed(0)
    progs = observable_program_generator(Hp, T, n_sample=1, n_repetition=1,
                                         diff_var=var, value=X_VAL)
    expfn = runner.make_expectation_fn(PSI0, OBS)
    w, e = [], []
    for H_tot, ug, _ in progs:
        w.append(+float(ug) * T); e.append(expfn(H_tot[0]))   # minus branch
        w.append(-float(ug) * T); e.append(expfn(H_tot[1]))   # plus branch
    return np.array(w), np.array(e)


def fd_terms(runner, T, eps):
    expfn = runner.make_expectation_fn(PSI0, OBS)
    fp = expfn([[H_eval(X_VAL + eps), T]])
    fm = expfn([[H_eval(X_VAL - eps), T]])
    return np.array([1.0 / (2 * eps), -1.0 / (2 * eps)]), np.array([fp, fm])


def rmse(w, e, n_per_eval, truth, R, rng):
    """RMSE of the shot-sampled estimator Σ w·f over R realizations."""
    n = int(max(1, round(n_per_eval)))
    p = 0.5 * (1 + np.clip(e, -1, 1))
    k = rng.binomial(n, p[None, :], size=(R, len(e)))
    est = (2.0 * k / n - 1.0) @ w
    return float(np.sqrt(np.mean((est - truth) ** 2)))


def panel(ax, runner, T, title, S_fixed, R, rng):
    """RMSE vs ε (FD) at fixed budget, with PSR (no ε) line, for one T."""
    truth = exact_grad(runner, T)
    wp, ep = psr_terms(runner, T)
    psr = rmse(wp, ep, S_fixed / 2, truth, R, rng)
    eps_grid = np.geomspace(0.01, 2.0, 20)
    fd = []
    for eps in eps_grid:
        wf, ef = fd_terms(runner, T, float(eps))
        fd.append(rmse(wf, ef, S_fixed / 2, truth, R, rng))
    ax.loglog(eps_grid, fd, "s-", color="#d62728", lw=2, label="FD (vary ε)")
    ax.axhline(psr, color="#1f77b4", lw=2.6, label="PSR (no ε)")
    ax.set_xlabel(r"FD step size $\varepsilon$")
    ax.set_ylabel(f"gradient RMSE  (N={S_fixed} shots)")
    ax.set_title(title, fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    return truth, psr, min(fd)


def main():
    runner = NoisyQuTiPRunner(2, noise=None)        # noiseless: isolate variance
    R, rng = 6000, np.random.default_rng(0)
    S_fixed = 2000

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.5), dpi=150)
    tA, pA, fA = panel(axA, runner, 0.5, "(A) SMOOTH landscape (T=0.5)",
                       S_fixed, R, rng)
    tB, pB, fB = panel(axB, runner, 5.0, "(B) SHARP landscape (T=5)",
                       S_fixed, R, rng)

    print(f"1q H=x·Z0+X0, <Z0>, x={X_VAL}, N={S_fixed} shots.\n")
    print(f"  SMOOTH T=0.5: exact grad {tA:+.4f}  PSR RMSE {pA:.4f}  "
          f"FD best-ε RMSE {fA:.4f}  →  {'PSR' if pA < fA else 'FD-tuned'} wins")
    print(f"  SHARP  T=5.0: exact grad {tB:+.4f}  PSR RMSE {pB:.4f}  "
          f"FD best-ε RMSE {fB:.4f}  →  {'PSR' if pB < fB else 'FD-tuned'} wins")

    fig.suptitle("Gradient accuracy vs FD step ε (single-point). With ε FREE "
                 "(simulator), a tuned ε beats analog PSR\n(PSR carries a T² + "
                 "τ-sampling variance penalty). FD's U-shape only traps it when ε "
                 "is FLOORED — see fd_bias_floor.", fontsize=8.8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "figures", "gradient_accuracy.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
