"""
shots_scaling.py — gradient distance vs TOTAL shots (log-log): FD (best ε) vs PSR
raw vs PSR rescaled, in the noisy regime.

Total shots N = 2·n_sample·n_per (τ-samples × repetitions/branch).  At fixed N the
PSR variance is A/n_sample + 2B/N, so MAXIMIZING τ-samples (few reps/branch) is
optimal — we use that split.  Distance = RMSE to the REAL (ideal) gradient.

Expectation:
  - PSR rescaled: variance ~1/N → N^{-1/2}, converging to a LOW floor (the small
    rescale residual) — the only estimator that reaches the true gradient.
  - PSR raw and FD best-ε: drop, then FLOOR at the uncorrectable attenuation bias
    (they measure/return the attenuated noisy gradient). FD also scales only ~N^{-1/3}.

Run:  conda run -n qec_pg python differential_computing/tests/shots_scaling.py
"""

import json
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
import analytic_rescale as ar
from observable_program_generator import observable_program_generator
from noisy_qutip import NoisyQuTiPRunner
from noise_model import NoiseModel

T = 2.5
T2 = 5.0                  # T/T2* = 0.50 (noisy regime)
OBS = qp.tensor(qp.sigmaz(), qp.qeye(2))
PSI0 = qp.tensor(qp.basis(2, 0), qp.basis(2, 0))
I = qp.qeye(2); X, Z = qp.sigmax(), qp.sigmaz()
R, POOL, NS_CAP = 2500, 1200, 1500


def Hsq(th):
    return th * qp.tensor(Z, I) + qp.tensor(X, I)


def Hsimuq():
    x = sp.Symbol("x"); qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    return x * q[0].Z + q[0].X, "x"


def fmaker(runner):
    H, var = Hsimuq()
    expfn = runner.make_expectation_fn(PSI0, OBS)
    return lambda x: expfn([[H.set_parameterizedHam({"x": float(x)}), T]])


def compute():
    rng = np.random.default_rng(0)
    clean = NoisyQuTiPRunner(2, noise=None)
    noisy = NoisyQuTiPRunner(2, noise=NoiseModel(n_qubits=2, T2=T2))
    fc, fn = fmaker(clean), fmaker(noisy)

    # moderate/large gradient point (stable rescale)
    xs = np.linspace(0.2, 1.6, 90)
    gi = np.array([(fc(x + 1e-3) - fc(x - 1e-3)) / 2e-3 for x in xs])
    x_star = float(xs[np.argmax(np.abs(gi))])
    g_real = (fc(x_star + 1e-3) - fc(x_star - 1e-3)) / 2e-3
    s = ar.lambda_slope(Hsq, OBS, PSI0, T, 2, z_sites=[0], theta=x_star)
    factor = ar.rescale_factor(s, T, T2)

    # PSR branch pool (noisy exact expectations)
    H, var = Hsimuq()
    np.random.seed(123)
    progs = observable_program_generator(H, T, n_sample=POOL, n_repetition=1,
                                         diff_var=var, value=x_star)
    pexp = noisy.make_expectation_fn(PSI0, OBS)
    H_tot, ug, _ = progs[0]; b = len(H_tot)//2
    em = np.array([pexp(H_tot[2*i]) for i in range(b)])
    ep = np.array([pexp(H_tot[2*i+1]) for i in range(b)])
    ug = float(ug)

    # FD exact endpoints, cached per ε
    eps_grid = np.geomspace(0.03, 1.0, 12)
    fd_pm = {e: (fn(x_star + e), fn(x_star - e)) for e in eps_grid}

    def shots(exact_arr, n):
        return 2.0*rng.binomial(int(max(1, n)),
                                0.5*(1+np.clip(exact_arr, -1, 1)))/max(1, n) - 1

    budgets = np.array([200, 600, 2000, 6000, 20000, 60000, 200000])
    fd_best, psr_raw, psr_res = [], [], []
    for N in budgets:
        # FD: best ε (2 evals, N/2 each)
        n_fd = N // 2
        best = np.inf
        for e in eps_grid:
            fp, fm = fd_pm[e]
            fp_est = 2.0*rng.binomial(n_fd, 0.5*(1+np.clip(fp, -1, 1)), size=R)/n_fd - 1
            fm_est = 2.0*rng.binomial(n_fd, 0.5*(1+np.clip(fm, -1, 1)), size=R)/n_fd - 1
            est = (fp_est - fm_est) / (2*e)
            best = min(best, float(np.sqrt(np.mean((est - g_real)**2))))
        fd_best.append(best)
        # PSR: maximize τ-samples (n_per small).  n_sample=min(N/2, cap), n_per rest.
        n_sample = int(min(N//2, NS_CAP))
        n_per = int(max(1, round(N/(2*n_sample))))
        idx = rng.integers(0, len(em), size=(R, n_sample))
        fm = 2.0*rng.binomial(n_per, 0.5*(1+np.clip(em[idx], -1, 1)))/n_per - 1
        fp = 2.0*rng.binomial(n_per, 0.5*(1+np.clip(ep[idx], -1, 1)))/n_per - 1
        raw = (T/n_sample)*ug*np.sum(fm - fp, axis=1)
        psr_raw.append(float(np.sqrt(np.mean((raw - g_real)**2))))
        psr_res.append(float(np.sqrt(np.mean((raw*factor - g_real)**2))))

    return dict(T=T, T2=T2, x_star=float(x_star), g_real=float(g_real),
                factor=float(factor), budgets=list(map(int, budgets)),
                fd_best=fd_best, psr_raw=psr_raw, psr_res=psr_res)


def main():
    figdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    os.makedirs(figdir, exist_ok=True)
    cache = os.path.join(figdir, "shots_scaling_data.json")
    if os.path.exists(cache):
        d = json.load(open(cache))
        print("loaded cache — replotting only")
    else:
        d = compute()
        json.dump(d, open(cache, "w"), default=float)
        print(f"cached: {cache}")

    x_star, g_real, factor = d["x_star"], d["g_real"], d["factor"]
    budgets = np.array(d["budgets"])
    fd_best, psr_raw, psr_res = d["fd_best"], d["psr_raw"], d["psr_res"]

    print(f"T={d['T']}, T/T2*={d['T']/d['T2']:.2f}, x*={x_star:.3f}, "
          f"real grad={g_real:+.4f}, 1/λ={factor:.2f}.")
    print(f"{'N':>9}{'FD best':>10}{'PSR raw':>10}{'PSR resc':>10}")
    for i, N in enumerate(budgets):
        print(f"{N:>9}{fd_best[i]:>10.4f}{psr_raw[i]:>10.4f}{psr_res[i]:>10.4f}")

    fig, ax = plt.subplots(figsize=(7.8, 5.2), dpi=150)
    ax.loglog(budgets, fd_best, "s--", color="#7b1fa2", lw=2, label="FD (best ε)")
    ax.loglog(budgets, psr_raw, "o--", color="#9e9e9e", lw=2, label="PSR raw")
    ax.loglog(budgets, psr_res, "o-", color="#00897b", lw=2.8, label="PSR rescaled")
    # scaling guides
    b0 = budgets[0]
    ax.loglog(budgets, psr_res[0]*(budgets/b0)**-0.5, ":", color="#00897b", lw=1,
              label=r"$N^{-1/2}$")
    ax.loglog(budgets, fd_best[0]*(budgets/b0)**(-1/3), ":", color="#7b1fa2", lw=1,
              label=r"$N^{-1/3}$")
    ax.set_xlabel("total shots / gradient  N   (= 2 · n_sample · n_reps)")
    ax.set_ylabel("distance to real gradient  (RMSE)")
    ax.set_title(f"Gradient accuracy vs total shots (T/T2*={d['T']/d['T2']:.2f})\n"
                 f"PSR rescaled converges (N^-1/2); FD best-ε & PSR raw FLOOR at "
                 f"the attenuation bias")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = os.path.join(figdir, "shots_scaling.png")
    fig.savefig(out)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
