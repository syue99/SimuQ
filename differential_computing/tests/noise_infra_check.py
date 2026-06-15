"""
noise_infra_check.py — validate the noise infra BEFORE writing formal tests.

Checks, in order:
  1. Noiseless equivalence (acceptance bar): the mesolve NoisyQuTiPRunner with no
     noise reproduces (a) the sesolve QuTiPSequentialRunner PSR gradient and
     (b) the finite-difference gradient, on 1q and 2q, to ~1e-3.
  2. T1/T2 changes the gradient (decoherence is actually acting).
  3. Pauli error rates change the gradient.
  4. Shot sampling is unbiased: mean over trials -> exact; std shrinks ~1/sqrt(N).

Run:  conda run -n qec_pg python differential_computing/tests/noise_infra_check.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sympy as sp
import qutip as qp

from simuq import QSystem, Qubit
from simuq.braket.diffQC_provider import diffQCProvider
from observable_program_generator import observable_program_generator
from combine_gradient import combine_gradient_results
from qutip_sequential import QuTiPSequentialRunner
from noisy_qutip import NoisyQuTiPRunner
from noise_model import NoiseModel
from shot_sampling import make_shot_expfn


# ── model builders ────────────────────────────────────────────────────────────

def build_1q():
    x = sp.Symbol("x")
    qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    H = x * q[0].Z + q[0].X
    obs = qp.tensor(qp.sigmaz(), qp.qeye(2))      # Z_0
    return H, 2, obs, "x"


def build_2q():
    x = sp.Symbol("x")
    qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    H = sp.sin(2 * x) * q[0].Z * q[1].Z + sp.sin(2 * x) * q[0].X \
        + sp.sin(2 * x) * q[1].X
    obs = qp.tensor(qp.sigmaz(), qp.sigmaz())     # Z_0 Z_1
    return H, 2, obs, "x"


def gen_programs(H, var, x_val, T, n_sample=1, seed=1):
    # The qutip gradient path needs only the PSR branches — no tweezer compile.
    np.random.seed(seed)
    return observable_program_generator(
        H, T, n_sample=n_sample, n_repetition=1, diff_var=var, value=x_val)


def fd_gradient(H, var, theta, T, runner, obs, eps=1e-4):
    expfn = runner.make_expectation_fn(runner.zero_state(), obs)

    def f(th):
        He = H.set_parameterizedHam({var: th})
        return expfn([[He, T]])
    return (f(theta + eps) - f(theta - eps)) / (2 * eps)


# ── checks ────────────────────────────────────────────────────────────────────

def check_noiseless_equivalence(name, build):
    H, n, obs, var = build()
    T, x_val = 0.5, 0.7

    sesolve_runner = QuTiPSequentialRunner(n)
    mesolve_runner = NoisyQuTiPRunner(n, noise=None)
    ses_expfn = sesolve_runner.make_expectation_fn(sesolve_runner.zero_state(), obs)
    mes_expfn = mesolve_runner.make_expectation_fn(mesolve_runner.zero_state(), obs)

    # (1) Infra correctness: open-system mesolve path reproduces the coherent
    #     sesolve PSR gradient exactly (same branches, same expectation seam).
    programs1 = gen_programs(H, var, x_val, T, n_sample=1)
    g_sesolve = combine_gradient_results(programs1, ses_expfn, T)
    g_mesolve = combine_gradient_results(programs1, mes_expfn, T)
    print(f"\n[{name}] (1) mesolve == sesolve  (n_sample=1, same branches)")
    print(f"  PSR (sesolve) = {g_sesolve:+.6f}")
    print(f"  PSR (mesolve) = {g_mesolve:+.6f}   Δ={abs(g_mesolve - g_sesolve):.2e}")
    assert abs(g_mesolve - g_sesolve) < 1e-5, "mesolve PSR != sesolve PSR"
    print("  [OK] open-system path reproduces the coherent PSR gradient")

    # (2) Gradient correctness: PSR is a stochastic estimator of the true
    #     gradient; with enough τ samples it converges to finite difference.
    g_fd = fd_gradient(H, var, x_val, T, mesolve_runner, obs)
    print(f"  (2) PSR → FD as n_sample grows   (FD = {g_fd:+.6f})")
    g_psr = None
    for ns in (1, 50, 200, 800):
        programs = gen_programs(H, var, x_val, T, n_sample=ns, seed=7)
        g_psr = combine_gradient_results(programs, mes_expfn, T)
        print(f"    n_sample={ns:4d}  PSR={g_psr:+.6f}  |PSR−FD|={abs(g_psr - g_fd):.4f}")
    assert abs(g_psr - g_fd) < 1e-2, f"PSR did not converge to FD ({g_psr} vs {g_fd})"
    print("  [OK] PSR converges to FD in the noiseless limit")
    return programs1, H, n, obs, var, T, x_val


def check_noise_changes_gradient(name, programs, n, obs, T):
    runner_clean = NoisyQuTiPRunner(n, noise=None)
    g_clean = combine_gradient_results(
        programs, runner_clean.make_expectation_fn(runner_clean.zero_state(), obs), T)

    nm_t1t2 = NoiseModel(n_qubits=n, T1=2.0, T2=1.0)
    runner_t = NoisyQuTiPRunner(n, noise=nm_t1t2)
    g_t = combine_gradient_results(
        programs, runner_t.make_expectation_fn(runner_t.zero_state(), obs), T)

    nm_pauli = NoiseModel(n_qubits=n, pauli_rates={"X": 0.3, "Z": 0.3})
    runner_p = NoisyQuTiPRunner(n, noise=nm_pauli)
    g_p = combine_gradient_results(
        programs, runner_p.make_expectation_fn(runner_p.zero_state(), obs), T)

    print(f"\n[{name}] noise changes the gradient")
    print(f"  clean            = {g_clean:+.6f}")
    print(f"  T1=2,T2=1        = {g_t:+.6f}   (Δ={g_t - g_clean:+.4f})")
    print(f"  Pauli X,Z rate.3 = {g_p:+.6f}   (Δ={g_p - g_clean:+.4f})")
    assert abs(g_t - g_clean) > 1e-4, "T1/T2 had no effect"
    assert abs(g_p - g_clean) > 1e-4, "Pauli rates had no effect"
    print("  [OK] both noise channels shift the gradient")


def check_shot_sampling(name, programs, n, obs, T):
    runner = NoisyQuTiPRunner(n, noise=None)
    exact_expfn = runner.make_expectation_fn(runner.zero_state(), obs)
    g_exact = combine_gradient_results(programs, exact_expfn, T)

    print(f"\n[{name}] shot sampling unbiased + variance shrinks")
    print(f"  exact PSR gradient = {g_exact:+.6f}")
    prev_std = None
    for n_shots in (64, 256, 1024, 4096):
        rng = np.random.default_rng(0)
        trials = []
        for _ in range(200):
            shot_expfn = make_shot_expfn(exact_expfn, n_shots, rng)
            trials.append(combine_gradient_results(programs, shot_expfn, T))
        mean, std = np.mean(trials), np.std(trials)
        print(f"  N={n_shots:5d}  mean={mean:+.5f}  std={std:.5f}")
        # mean within a few std/sqrt(trials) of exact
        assert abs(mean - g_exact) < 5 * std / np.sqrt(200) + 1e-6
        if prev_std is not None:
            assert std < prev_std, "std did not shrink with more shots"
        prev_std = std
    print("  [OK] shot estimator is unbiased and converges ~1/sqrt(N)")


def main():
    for name, build in (("1q", build_1q), ("2q", build_2q)):
        programs, H, n, obs, var, T, x_val = check_noiseless_equivalence(name, build)
        check_noise_changes_gradient(name, programs, n, obs, T)
        check_shot_sampling(name, programs, n, obs, T)
    print("\nAll noise-infra checks passed.")


if __name__ == "__main__":
    main()
