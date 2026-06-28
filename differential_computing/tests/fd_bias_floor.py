"""
fd_bias_floor.py — find the regime where FD suffers its U-shape (irreducible bias
floor) and PSR converges below it.

At a FIXED step ε (the realistic case — you cannot retune ε every iteration), FD's
gradient RMSE has a bias floor:
    RMSE_FD² = (ε²·f'''/6)²  +  (σ / (2ε√N))²   →   ε²·f'''/6   as N→∞.
On a SHARP landscape (large f''') this floor is high for EVERY fixed ε: small ε
fights variance, large ε fights bias, no ε escapes (the U-shape).

Analog PSR is unbiased in the shot noise but has its own floor — the τ-sampling
variance at fixed n_sample (∝ 1/√n_sample), independent of N.  PSR beats FD when
its τ-floor sits below FD's bias floor → needs enough n_sample AND/OR a sharp
landscape.

We plot gradient RMSE vs shot budget N (log-log) on a sharp landscape:
  FD at fixed ε → descends then FLOORS at its bias (flat at high N).
  PSR at a few n_sample → descends ~N^{-1/2} then floors at its τ-variance.
PSR (adequate n_sample) ends below every FD fixed-ε floor.

Saves figures/fd_bias_floor.png.

Run:  conda run -n qec_pg python differential_computing/tests/fd_bias_floor.py
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

# SHARP landscape: long evolution so the feature scale in x is SMALL — below the
# hardware control floor ε_min, so FD cannot pick a small-enough ε (every allowed
# ε ≥ ε_min aliases → bias).  This is the genuine "FD U-shape" regime (floored ε
# meets a sharp landscape); on a simulator with arbitrarily small ε FD would win.
T = 6.0
X_VAL = 0.50
EPS_MIN = 0.2          # hardware control-resolution floor on the FD step size
OBS = qp.tensor(qp.sigmaz(), qp.qeye(2))
PSI0 = qp.tensor(qp.basis(2, 0), qp.basis(2, 0))


def H_param():
    x = sp.Symbol("x")
    qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    return x * q[0].Z + q[0].X, "x"


def H_eval(x):
    Hp, var = H_param()
    return Hp.set_parameterizedHam({var: float(x)})


def exact_grad(runner, eps=1e-3):
    expfn = runner.make_expectation_fn(PSI0, OBS)
    f = lambda x: expfn([[H_eval(x), T]])
    return (f(X_VAL + eps) - f(X_VAL - eps)) / (2 * eps)


def psr_pool(runner, pool_size):
    """Exact per-branch expectations for a large pool of τ samples.

    Returns (em, ep, ug): minus/plus branch expectations and the gradient
    coefficient.  Each trial later SUBSAMPLES n_sample τ's from this pool (so the
    τ-sampling variance is properly averaged) and adds shot noise.
    """
    Hp, var = H_param()
    np.random.seed(123)
    progs = observable_program_generator(Hp, T, n_sample=pool_size,
                                         n_repetition=1, diff_var=var, value=X_VAL)
    expfn = runner.make_expectation_fn(PSI0, OBS)
    H_tot, ug, _ = progs[0]               # single-term H → one program
    b = len(H_tot) // 2
    em = np.array([expfn(H_tot[2 * i]) for i in range(b)])
    ep = np.array([expfn(H_tot[2 * i + 1]) for i in range(b)])
    return em, ep, float(ug)


def psr_rmse(em, ep, ug, n_sample, N, truth, R, rng):
    """RMSE of the PSR estimate: subsample n_sample τ's + shot-sample, per trial.

    grad = (T/n_sample)·ug·Σ_k (f-_k − f+_k), with each branch given N/(2·n_sample)
    shots.  Captures BOTH τ-sampling variance and shot variance.
    """
    P = len(em)
    n_per = int(max(1, round(N / (2 * n_sample))))
    idx = rng.integers(0, P, size=(R, n_sample))
    pm = 0.5 * (1 + np.clip(em[idx], -1, 1))      # (R, n_sample)
    pp = 0.5 * (1 + np.clip(ep[idx], -1, 1))
    fm = 2.0 * rng.binomial(n_per, pm) / n_per - 1.0
    fp = 2.0 * rng.binomial(n_per, pp) / n_per - 1.0
    est = (T / n_sample) * ug * np.sum(fm - fp, axis=1)
    return float(np.sqrt(np.mean((est - truth) ** 2)))


def fd_terms(runner, eps):
    expfn = runner.make_expectation_fn(PSI0, OBS)
    fp = expfn([[H_eval(X_VAL + eps), T]])
    fm = expfn([[H_eval(X_VAL - eps), T]])
    return np.array([1.0 / (2 * eps), -1.0 / (2 * eps)]), np.array([fp, fm])


def rmse(w, e, n_per_eval, truth, R, rng):
    n = int(max(1, round(n_per_eval)))
    p = 0.5 * (1 + np.clip(e, -1, 1))
    k = rng.binomial(n, p[None, :], size=(R, len(e)))
    est = (2.0 * k / n - 1.0) @ w
    return float(np.sqrt(np.mean((est - truth) ** 2)))


def main():
    runner = NoisyQuTiPRunner(2, noise=None)
    truth = exact_grad(runner)
    R, rng = 8000, np.random.default_rng(0)
    budgets = np.array([100, 300, 1000, 3000, 10000, 30000, 100000, 300000])
    print(f"SHARP landscape: 1q H=x·Z0+X0, <Z0>, T={T}, x={X_VAL}.  "
          f"exact grad={truth:+.4f}\n")

    psr_specs = [("PSR n=64", 64, "#9edae5"), ("PSR n=256", 256, "#17becf"),
                 ("PSR n=1024", 1024, "#1f77b4")]
    # FD step size is floored at EPS_MIN (hardware) → only ε ≥ EPS_MIN allowed.
    fd_specs = [("FD ε=0.2 (=min)", 0.2, "#d62728"), ("FD ε=0.4", 0.4, "#ff7f0e"),
                ("FD ε=0.8", 0.8, "#9467bd")]

    em, ep, ug = psr_pool(runner, pool_size=2000)
    curves = {}
    for lbl, ns, _ in psr_specs:
        curves[lbl] = [psr_rmse(em, ep, ug, ns, S, truth, R, rng) for S in budgets]
    for lbl, eps, _ in fd_specs:
        w, e = fd_terms(runner, eps)
        curves[lbl] = [rmse(w, e, S / 2, truth, R, rng) for S in budgets]

    print(f"{'N':>8}" + "".join(f"{lbl:>11}" for lbl, *_ in psr_specs + fd_specs))
    for i, S in enumerate(budgets):
        print(f"{S:>8}" + "".join(f"{curves[lbl][i]:>11.4f}"
                                  for lbl, *_ in psr_specs + fd_specs))

    fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=150)
    for lbl, ns, c in psr_specs:
        ax.loglog(budgets, curves[lbl], "o-", color=c, lw=2.4, label=lbl)
    for lbl, eps, c in fd_specs:
        ax.loglog(budgets, curves[lbl], "s--", color=c, lw=1.8, label=lbl)
    ax.set_xlabel("total shots / gradient  N")
    ax.set_ylabel("gradient RMSE")
    ax.set_title(f"Sharp landscape (T={T}) + floored step (ε≥{EPS_MIN}): every "
                 f"allowed FD ε\nhits a high bias floor; PSR (more τ-samples) "
                 f"converges below — no ε rescues FD")
    ax.legend(frameon=False, fontsize=8.5, ncol=2)
    fig.tight_layout()
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "figures", "fd_bias_floor.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
