"""
psr_dephasing_sweep.py — PSR bias in the neutral-atom regime: T1 → ∞, T2* (Z
dephasing) dominates.

Relaxation (T1) is the least physical channel for neutral atoms and gave the
worst PSR bias.  Here T1 is OFF entirely and we sweep the pure-dephasing
strength from strong (T2 ~ T) to weak (T2 ≫ T), to find whether there is a
realistic operating point (T/T2 small) where the parameter-shift gradient is
near-unbiased — i.e. whether PSR is usable on your platform.

Per row, against ground_truth = d<O>_noisy/dθ (fine-ε FD on the exact dephased
mesolve landscape):
  PSR_bias      = PSR_noise − ground_truth   (absolute)
  rel_bias      = PSR_bias / ground_truth    (what matters for usability)
  FD_step_bias  = coarse-ε (0.1) FD − ground_truth

Run:  conda run -n qec_pg python differential_computing/tests/psr_dephasing_sweep.py
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


# pure-dephasing T2 sweep (T1 off); T fixed below
T2_SWEEP = [0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, None]   # None = clean baseline


def run_model(name, build, n_sample):
    H, n, obs, var = build()
    T, x_val = 0.5, 0.7
    print(f"\n=== {name}   T1=∞ (pure T2 dephasing),  T={T}, x={x_val}, "
          f"PSR n_sample={n_sample} ===")
    print(f"{'T2':>6}{'T/T2':>7}{'gtruth':>10}{'PSR':>10}{'PSR_bias':>11}"
          f"{'rel_bias':>10}{'FD_step':>10}")
    for T2 in T2_SWEEP:
        noise = None if T2 is None else NoiseModel(n_qubits=n, T2=T2)
        runner = NoisyQuTiPRunner(n, noise=noise)
        truth = grad_fd(H, var, x_val, T, runner, obs, 1e-2)
        psr = grad_psr(H, var, x_val, T, runner, obs, n_sample)
        fd_hw = grad_fd(H, var, x_val, T, runner, obs, 0.1)
        bias = psr - truth
        rel = bias / truth if abs(truth) > 1e-9 else float("nan")
        tt = (T / T2) if T2 else 0.0
        lbl = f"{T2:6.1f}" if T2 else "  clean"
        print(f"{lbl}{tt:>7.3f}{truth:>10.5f}{psr:>10.5f}{bias:>+11.5f}"
              f"{rel:>+9.1%}{fd_hw - truth:>+10.5f}")


def main():
    run_model("1q  H = x·Z0 + X0   <Z0>", build_1q, n_sample=1500)
    run_model("2q  sin(2x)(Z0Z1+X0+X1) <Z0Z1>", build_2q, n_sample=400)
    print("\nLook for the T/T2 where rel_bias drops to a usable level (few %). "
          "That is the neutral-atom operating regime for PSR.")


if __name__ == "__main__":
    main()
