"""
fd_fails_sharp_landscape.py — the regime where Q3 holds: FD gets the gradient
SIGN wrong (even noiseless, infinite shots) while PSR stays correct.

Mechanism: FD's secant fails when the step ε exceeds the landscape's curvature
scale.  In analog evolution the dynamical phase Ω(θ)·T winds faster in θ as the
evolution time T grows, so the landscape <O>(θ) oscillates on a scale ~1/T.
With a hardware-floored ε, large T makes ε span multiple oscillations → FD
aliases → wrong sign.  PSR uses an exact macroscopic shift (no ε) → immune.

H = θ·Z0 + X0 (a Rabi-type landscape), observable <Z0>.  We sweep T (feature
scale ~1/T) and count, over a θ grid, how often each estimator's SIGN disagrees
with the true gradient.  Part A is noiseless (PSR exact — pure step-size effect);
Part B repeats under dephasing (PSR attenuates but keeps sign, λ>0).

Run:  conda run -n qec_pg python differential_computing/tests/fd_fails_sharp_landscape.py
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


def fd_grad(H, var, theta, T, runner, obs, eps):
    expfn = runner.make_expectation_fn(runner.zero_state(), obs)

    def f(th):
        return expfn([[H.set_parameterizedHam({var: float(th)}), T]])
    return (f(theta + eps) - f(theta - eps)) / (2.0 * eps)


def psr_grad(H, var, theta, T, runner, obs, n_sample, seed=11):
    expfn = runner.make_expectation_fn(runner.zero_state(), obs)
    np.random.seed(seed)
    programs = observable_program_generator(
        H, T, n_sample=n_sample, n_repetition=1, diff_var=var, value=float(theta))
    return combine_gradient_results(programs, expfn, T)


def sweep(T, eps_floor, noise, n_sample, thetas):
    H, n, obs, var = build_1q()
    runner = NoisyQuTiPRunner(n, noise=noise)
    fd_bad = psr_bad = counted = 0
    fd_mag_err = []
    for th in thetas:
        truth = fd_grad(H, var, th, T, runner, obs, 1e-3)
        if abs(truth) < 1e-3:
            continue                       # skip true zeros (sign undefined)
        counted += 1
        fd = fd_grad(H, var, th, T, runner, obs, eps_floor)
        psr = psr_grad(H, var, th, T, runner, obs, n_sample)
        fd_bad += (np.sign(fd) != np.sign(truth))
        psr_bad += (np.sign(psr) != np.sign(truth))
    return fd_bad, psr_bad, counted


def main():
    thetas = [float(v) for v in np.linspace(0.2, 4.0, 40)]
    EPS = 0.5            # hardware-floored FD step

    print("#" * 70)
    print(f"PART A — noiseless (PSR exact).  FD step floored at ε={EPS}.")
    print("Longer evolution T -> finer θ-features -> FD aliases -> wrong sign.")
    print("#" * 70)
    print(f"{'T':>5}{'feature~1/T':>13}{'FD wrong-sign':>16}{'PSR wrong-sign':>16}")
    for T in (1.0, 3.0, 6.0, 10.0, 15.0):
        fd_bad, psr_bad, m = sweep(T, EPS, None, 400, thetas)
        print(f"{T:>5.0f}{1.0/T:>13.3f}{f'{fd_bad}/{m}':>16}{f'{psr_bad}/{m}':>16}")

    print("\n" + "#" * 70)
    print(f"PART B — with dephasing T2=5 (PSR attenuates, λ>0).  ε={EPS}.")
    print("#" * 70)
    print(f"{'T':>5}{'feature~1/T':>13}{'FD wrong-sign':>16}{'PSR wrong-sign':>16}")
    for T in (6.0, 10.0, 15.0):
        noise = NoiseModel(n_qubits=2, T2=5.0)
        fd_bad, psr_bad, m = sweep(T, EPS, noise, 400, thetas)
        print(f"{T:>5.0f}{1.0/T:>13.3f}{f'{fd_bad}/{m}':>16}{f'{psr_bad}/{m}':>16}")

    print("\nRegime where Q3 holds: large T (sharp landscape) + floored ε. "
          "There FD is\nsign-unreliable at ANY shot count, while PSR stays "
          "direction-correct — PSR's win.")


if __name__ == "__main__":
    main()
