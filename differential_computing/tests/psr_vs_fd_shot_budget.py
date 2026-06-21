"""
psr_vs_fd_shot_budget.py — the core benchmark: PSR vs FD at equal shot budget,
in the neutral-atom moderate-dephasing regime (T1=∞, T/T2* ~ 0.1–0.2).

Bias is only half the story.  FD tracks the true noisy landscape (tiny bias) but
its 1/(2ε) factor amplifies shot noise → high variance.  PSR has a dephasing
bias (~10–40% here) but a macroscopic shift → much lower variance.  What matters
on hardware is total error at a fixed measurement budget:

    RMSE(N) = sqrt( bias² + variance )   vs  ground_truth = d<O>_noisy/dθ.

Method (fast + exact): each evaluation point's EXACT decohered expectation is
computed once with mesolve; shot noise is then drawn by cheap binomial
resampling (±1 observable).  Equal budget: a total of N shots is split evenly
across each estimator's circuit evaluations
    FD  : 2 evaluations (θ±ε)            → N/2 shots each
    PSR : one evaluation per branch       → N/(#branches) shots each
so PSR's need for more distinct circuits is honestly charged against it.

Run:  conda run -n qec_pg python differential_computing/tests/psr_vs_fd_shot_budget.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sympy as sp
import qutip as qp

from simuq import QSystem, Qubit
from observable_program_generator import observable_program_generator
from noisy_qutip import NoisyQuTiPRunner
from noise_model import NoiseModel


def build_2q():
    x = sp.Symbol("x")
    qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    H = sp.sin(2 * x) * q[0].Z * q[1].Z + sp.sin(2 * x) * q[0].X \
        + sp.sin(2 * x) * q[1].X
    return H, 2, qp.tensor(qp.sigmaz(), qp.sigmaz()), "x"


def build_1q():
    x = sp.Symbol("x")
    qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    H = x * q[0].Z + q[0].X
    return H, 2, qp.tensor(qp.sigmaz(), qp.qeye(2)), "x"


def _shot_resample(exact_vals, n_per_eval, n_trials, rng):
    """Vectorized ±1 shot estimates.

    exact_vals : (E,) exact expectations in [-1,1] for E evaluation points
    returns    : (n_trials, E) shot-sampled estimates with n_per_eval shots each
    """
    e = np.clip(np.asarray(exact_vals, float), -1.0, 1.0)
    p = 0.5 * (1.0 + e)                                    # (E,)
    k = rng.binomial(int(n_per_eval), p[None, :], size=(n_trials, len(e)))
    return 2.0 * k / n_per_eval - 1.0


def psr_weights_and_exacts(H, var, theta, T, runner, obs, n_sample, seed=11):
    """Flatten PSR into  grad = Σ_b w_b · f_b  over all branches b.

    Returns (weights[E], exacts[E]); E = number of circuit evaluations (branches).
    """
    expfn = runner.make_expectation_fn(runner.zero_state(), obs)
    np.random.seed(seed)
    programs = observable_program_generator(
        H, T, n_sample=n_sample, n_repetition=1, diff_var=var, value=theta)
    weights, exacts = [], []
    for H_tot_list, ugrad, _ in programs:
        b = len(H_tot_list) // 2
        if b == 0:
            continue
        for i in range(b):
            w = float(ugrad) * (T / b)
            # paper: (f_minus − f_plus); even idx = sgn-1 (minus), odd = plus
            weights.append(+w); exacts.append(expfn(H_tot_list[2 * i]))
            weights.append(-w); exacts.append(expfn(H_tot_list[2 * i + 1]))
    return np.array(weights), np.array(exacts)


def fd_weights_and_exacts(H, var, theta, T, runner, obs, eps):
    expfn = runner.make_expectation_fn(runner.zero_state(), obs)
    f_plus = expfn([[H.set_parameterizedHam({var: theta + eps}), T]])
    f_minus = expfn([[H.set_parameterizedHam({var: theta - eps}), T]])
    w = 1.0 / (2.0 * eps)
    return np.array([+w, -w]), np.array([f_plus, f_minus])


def rmse_curve(weights, exacts, ground_truth, budgets, n_trials, rng):
    """For each total-shot budget, return (rmse, |bias|, std) of Σ w·f_shot."""
    E = len(exacts)
    out = []
    for N in budgets:
        n_per = max(1, int(N // E))
        samp = _shot_resample(exacts, n_per, n_trials, rng)        # (trials, E)
        ests = samp @ weights                                      # (trials,)
        bias = np.mean(ests) - ground_truth
        std = np.std(ests)
        rmse = np.sqrt(bias ** 2 + std ** 2)
        out.append((N, E, n_per, rmse, abs(bias), std))
    return out


def run(name, build, T2, n_sample, fd_eps=0.1, n_trials=4000):
    H, n, obs, var = build()
    T, x_val = 0.5, 0.7
    runner = NoisyQuTiPRunner(n, noise=NoiseModel(n_qubits=n, T2=T2))
    truth_fn = NoisyQuTiPRunner(n, noise=NoiseModel(n_qubits=n, T2=T2))
    # ground truth: fine-ε FD on the exact dephased landscape
    gw, ge = fd_weights_and_exacts(H, var, x_val, T, truth_fn, obs, 1e-2)
    ground_truth = float(gw @ ge)

    pw, pe = psr_weights_and_exacts(H, var, x_val, T, runner, obs, n_sample)
    fw, fe = fd_weights_and_exacts(H, var, x_val, T, runner, obs, fd_eps)

    budgets = [200, 600, 2000, 6000, 20000, 60000, 200000]
    rng = np.random.default_rng(0)
    psr = rmse_curve(pw, pe, ground_truth, budgets, n_trials, rng)
    fd = rmse_curve(fw, fe, ground_truth, budgets, n_trials, rng)

    print(f"\n=== {name}   T1=∞, T2={T2} (T/T2={T/T2:.2f}), gtruth={ground_truth:+.5f} ===")
    print(f"   PSR branches={psr[0][1]} (bias floor={psr[-1][4]:.5f}),  "
          f"FD ε={fd_eps} (bias floor={fd[-1][4]:.5f})")
    print(f"{'N_total':>9} | {'PSR n/ev':>8}{'PSR_rmse':>10}{'PSR_std':>9} | "
          f"{'FD n/ev':>8}{'FD_rmse':>10}{'FD_std':>9} | winner")
    for (N, Ep, npp, pr, pb, ps), (_, Ef, npf, fr, fb, fs) in zip(psr, fd):
        win = "PSR" if pr < fr else "FD"
        print(f"{N:>9} | {npp:>8}{pr:>10.5f}{ps:>9.5f} | "
              f"{npf:>8}{fr:>10.5f}{fs:>9.5f} | {win}")


def main():
    # moderate neutral-atom regime: T/T2* = 0.1 and 0.2
    run("2q  <Z0Z1>", build_2q, T2=5.0, n_sample=1)    # τ-flat → n_sample=1 optimal
    run("2q  <Z0Z1>", build_2q, T2=2.5, n_sample=1)
    run("1q  <Z0>",   build_1q, T2=5.0, n_sample=4)
    print("\nRMSE = sqrt(bias² + var). FD: tiny bias, variance amplified by "
          "1/(2ε). PSR: dephasing-bias floor, low variance. Crossover = the "
          "shot budget below which PSR's lower variance wins.")


if __name__ == "__main__":
    main()
