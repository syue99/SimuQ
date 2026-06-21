"""
psr_bias_under_noise.py — who is biased under decoherence, PSR or FD?

Ground truth = d<O>_noisy/dθ, computed as fine-ε central FD on the EXACT
(decohered) mesolve landscape — ε→0 is allowed in simulation. Against that target:

  PSR_noise   : the PSR estimator evaluated with the noisy mesolve expectations
                (large n_sample to remove τ Monte-Carlo variance, no shot noise).
                -> PSR_bias = PSR_noise − ground_truth
                   (is the shift rule still exact under a Lindbladian?)

  FD_coarse   : finite difference at a hardware-realistic step ε (can't be made
                tiny on real control hardware).
                -> FD_step_bias = FD_coarse − ground_truth
                   (the irreducible truncation/step-size bias)

The "clean" row is a sanity check: PSR_bias must be ~0 noiselessly.

Run:  conda run -n qec_pg python differential_computing/tests/psr_bias_under_noise.py
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
from combine_gradient import combine_gradient_results
from noisy_qutip import NoisyQuTiPRunner
from noise_model import NoiseModel


def build_1q():
    x = sp.Symbol("x")
    qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    H = x * q[0].Z + q[0].X
    return H, 2, qp.tensor(qp.sigmaz(), qp.qeye(2)), "x"


def build_2q():
    x = sp.Symbol("x")
    qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    H = sp.sin(2 * x) * q[0].Z * q[1].Z + sp.sin(2 * x) * q[0].X \
        + sp.sin(2 * x) * q[1].X
    return H, 2, qp.tensor(qp.sigmaz(), qp.sigmaz()), "x"


def grad_fd(H, var, theta, T, runner, obs, eps):
    expfn = runner.make_expectation_fn(runner.zero_state(), obs)

    def f(th):
        return expfn([[H.set_parameterizedHam({var: th}), T]])
    return (f(theta + eps) - f(theta - eps)) / (2.0 * eps)


def grad_psr(H, var, theta, T, runner, obs, n_sample, seed=11):
    expfn = runner.make_expectation_fn(runner.zero_state(), obs)
    np.random.seed(seed)
    programs = observable_program_generator(
        H, T, n_sample=n_sample, n_repetition=1, diff_var=var, value=theta)
    return combine_gradient_results(programs, expfn, T)


SETTINGS = [
    ("clean",           dict()),
    ("T2=2.0 deph",     dict(T2=2.0)),
    ("T2=0.5 deph",     dict(T2=0.5)),
    ("T1=2,T2=1",       dict(T1=2.0, T2=1.0)),
    ("Pauli Z r=0.2",   dict(pauli_rates={"Z": 0.2})),
    ("Pauli X r=0.2",   dict(pauli_rates={"X": 0.2})),
]


def run_model(name, build, n_sample):
    H, n, obs, var = build()
    T, x_val = 0.5, 0.7
    EPS_TRUTH = 1e-2     # fine-ε ground truth (ε→0 allowed in sim)
    EPS_HW = 0.1         # hardware-realistic FD step

    print(f"\n=== {name}  (T={T}, x={x_val}, PSR n_sample={n_sample}) ===")
    print(f"{'setting':<16}{'ground_truth':>13}{'PSR_noise':>12}{'PSR_bias':>11}"
          f"{'FD(ε=.1)':>11}{'FD_step_bias':>14}")
    for label, kw in SETTINGS:
        noise = None if not kw else NoiseModel(n_qubits=n, **kw)
        runner = NoisyQuTiPRunner(n, noise=noise)
        truth = grad_fd(H, var, x_val, T, runner, obs, EPS_TRUTH)
        psr = grad_psr(H, var, x_val, T, runner, obs, n_sample)
        fd_hw = grad_fd(H, var, x_val, T, runner, obs, EPS_HW)
        print(f"{label:<16}{truth:>13.5f}{psr:>12.5f}{psr - truth:>+11.5f}"
              f"{fd_hw:>11.5f}{fd_hw - truth:>+14.5f}")


def main():
    run_model("1q  H = x·Z0 + X0   <Z0>", build_1q, n_sample=2000)
    run_model("2q  sin(2x)(Z0Z1+X0+X1) <Z0Z1>", build_2q, n_sample=800)
    print("\nReading: PSR_bias ~ 0 => shift rule stays (near-)exact under that "
          "channel.\nFD_step_bias = the irreducible step-size error at a "
          "hardware-realistic ε.")


if __name__ == "__main__":
    main()
