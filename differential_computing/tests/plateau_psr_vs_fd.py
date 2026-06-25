"""
plateau_psr_vs_fd.py — at a gradient plateau, which is better, PSR or FD?

Near a stationary point the true gradient → 0.  PSR's failure mode there is
differential-attenuation bias (systematic, low variance → it COMMITS to an
answer, maybe wrong).  FD's failure mode is variance (1/(2ε) amplifies shot
noise → it RANDOM-WALKS, but unbiased).  Which actually points toward the true
minimum more reliably at a realistic shot budget?  Test, don't assert.

Setup: 2q sin(2x)·(Z0Z1+X0+X1), <Z0Z1>, which has a stationary point at x=π/4≈
0.785.  Sweep θ across it.  Ground truth = the IDEAL (noiseless) gradient =
the descent direction we want.  Both estimators run under the full realistic
noise (dephasing T2=5 + reference gate error 1e-4/1e-3) at equal total shot
budget N, with FD at a hardware-floored ε.  Per θ we draw R shot realizations
and report, vs the ideal gradient:
  - sign-correct fraction  (does it point the right way?)
  - RMSE                    (how close?)

Run:  conda run -n qec_pg python differential_computing/tests/plateau_psr_vs_fd.py
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


def build_2q_uniform():
    # UNIFORM x-dependence: one common sin(2x) prefactor → no differential
    # attenuation (all gradient contributions scale together).
    x = sp.Symbol("x")
    qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    H = sp.sin(2 * x) * q[0].Z * q[1].Z + sp.sin(2 * x) * q[0].X \
        + sp.sin(2 * x) * q[1].X
    return H, 2, qp.tensor(qp.sigmaz(), qp.sigmaz()), "x"


def build_1q_hetero():
    # HETEROGENEOUS x-dependence: a Z-term (linear, dephasing-insensitive kick)
    # competing with an X-term (sin, dephasing-sensitive kick).  Their gradient
    # contributions can near-cancel and dephase UNEQUALLY → PSR's danger case.
    x = sp.Symbol("x")
    qs = QSystem(); q = [Qubit(qs)]
    H = sp.sin(x) * q[0].X + 0.8 * x * q[0].Z
    return H, 1, qp.sigmaz(), "x"


def g_ideal(H, var, theta, T, n, obs, eps=1e-2):
    clean = NoisyQuTiPRunner(n, noise=None)
    expfn = clean.make_expectation_fn(clean.zero_state(), obs)

    def f(th):
        return expfn([[H.set_parameterizedHam({var: float(th)}), T]])
    return (f(theta + eps) - f(theta - eps)) / (2.0 * eps)


def psr_components(H, var, theta, T, runner, obs, n_sample, seed=11):
    """Flatten PSR into grad = Σ_b w_b·f_b under noisy expectations."""
    expfn = runner.make_expectation_fn(runner.zero_state(), obs)
    np.random.seed(seed)
    programs = observable_program_generator(
        H, T, n_sample=n_sample, n_repetition=1, diff_var=var, value=float(theta))
    w, e = [], []
    for H_tot, ugrad, _ in programs:
        b = len(H_tot) // 2
        for i in range(b):
            ww = float(ugrad) * (T / b)
            w.append(+ww); e.append(expfn(H_tot[2 * i]))
            w.append(-ww); e.append(expfn(H_tot[2 * i + 1]))
    return np.array(w), np.array(e)


def fd_components(H, var, theta, T, runner, obs, eps):
    expfn = runner.make_expectation_fn(runner.zero_state(), obs)
    fp = expfn([[H.set_parameterizedHam({var: theta + eps}), T]])
    fm = expfn([[H.set_parameterizedHam({var: theta - eps}), T]])
    w = 1.0 / (2.0 * eps)
    return np.array([+w, -w]), np.array([fp, fm])


def resample(w, e, N, R, rng):
    """R shot-estimates of Σ w·f with N total shots split over the evaluations."""
    E = len(e)
    n_per = max(1, int(N // E))
    p = 0.5 * (1.0 + np.clip(e, -1, 1))
    k = rng.binomial(n_per, p[None, :], size=(R, E))
    samp = 2.0 * k / n_per - 1.0
    return samp @ w


def run_case(title, H, n, obs, var, xs, n_sample):
    T = 0.5
    noise = NoiseModel(n_qubits=n, T2=5.0, gate_error_1q=1e-4, gate_error_2q=1e-3,
                       gate_coherent_frac=0.5)
    runner = NoisyQuTiPRunner(n, noise=noise)
    N_total, R, FD_EPS = 2000, 600, 0.1

    print(f"\n=== {title} ===")
    print(f"Noise: dephasing T2=5 + gate(1e-4,1e-3).  N={N_total} shots, "
          f"FD ε={FD_EPS}, {R} realizations.  Truth = ideal (noiseless) grad.")
    print(f"{'x':>6}{'g_ideal':>10} | {'PSR sign%':>10}{'PSR rmse':>10} | "
          f"{'FD sign%':>10}{'FD rmse':>10} | RMSE/sign winner")
    rng = np.random.default_rng(0)
    for x in xs:
        gi = g_ideal(H, var, x, T, n, obs)
        pw, pe = psr_components(H, var, x, T, runner, obs, n_sample=n_sample)
        fw, fe = fd_components(H, var, x, T, runner, obs, FD_EPS)
        psr = resample(pw, pe, N_total, R, rng)
        fd = resample(fw, fe, N_total, R, rng)
        ps = float(np.mean(np.sign(psr) == np.sign(gi)))
        fs = float(np.mean(np.sign(fd) == np.sign(gi)))
        pr = float(np.sqrt(np.mean((psr - gi) ** 2)))
        fr = float(np.sqrt(np.mean((fd - gi) ** 2)))
        win_r = "PSR" if pr < fr else "FD"
        win_s = "PSR" if ps > fs else ("FD" if fs > ps else "tie")
        flag = "  <-- PSR sign-flips" if ps < 0.5 else ""
        print(f"{x:>6.3f}{gi:>10.4f} | {ps:>10.1%}{pr:>10.4f} | "
              f"{fs:>10.1%}{fr:>10.4f} | {win_r}/{win_s}{flag}")


def main():
    # Case A: uniform x-dependence — PSR's easy plateau.
    H, n, obs, var = build_2q_uniform()
    run_case("UNIFORM 2q <Z0Z1>, stationary pt x≈0.785", H, n, obs, var,
             [float(v) for v in np.linspace(0.66, 0.95, 9)], n_sample=1)

    # Case B: heterogeneous x-dependence — PSR's danger plateau (differential
    # attenuation can flip the sign near a cancellation).
    H, n, obs, var = build_1q_hetero()
    run_case("HETEROGENEOUS 1q  sin(x)X0 + 0.8x·Z0,  <Z0>", H, n, obs, var,
             [float(v) for v in np.linspace(1.4, 2.6, 13)], n_sample=300)

    print("\nThe heterogeneous case is the real test: where g_ideal→0, does PSR "
          "keep the right\nsign (low RMSE) or commit to the WRONG one (sign-flip)? "
          "Compare to FD's coin-flip.")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
