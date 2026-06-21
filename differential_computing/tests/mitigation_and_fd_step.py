"""
mitigation_and_fd_step.py — (1) mitigate PSR's dephasing bias via zero-noise
extrapolation, and (2) show how FD's step size ε trades bias vs variance.

PART 1 — PSR + ZNE
  Dephasing attenuates the PSR gradient multiplicatively (PSR ≈ λ·truth, λ<1).
  ZNE amplifies the dephasing rate to c·Γ0 for c = 1,2,3,... measures PSR at
  each (exact, large n_sample → no τ/shot variance), and extrapolates to c→0.
  Targets the IDEAL (noiseless) gradient — which FD cannot provide, since FD
  only ever measures the noisy-landscape gradient.
    - linear (Richardson) from c=1,2:   g0 = 2·g(1) − g(2)
    - quadratic from c=1,2,3:           Lagrange extrapolation to 0

PART 2 — FD step size
  Exact FD bias vs ε (no shots) shows the O(ε²) truncation growth; the analytic
  variance amplification 1/(2ε)² shows the opposing pull.  A hardware resolution
  floor ε_min bounds how small ε can go → an irreducible FD bias on real control.

Run:  conda run -n qec_pg python differential_computing/tests/mitigation_and_fd_step.py
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


def psr_grad(H, var, theta, T, runner, obs, n_sample, seed=11):
    expfn = runner.make_expectation_fn(runner.zero_state(), obs)
    np.random.seed(seed)
    programs = observable_program_generator(
        H, T, n_sample=n_sample, n_repetition=1, diff_var=var, value=theta)
    return combine_gradient_results(programs, expfn, T)


def fd_grad(H, var, theta, T, runner, obs, eps):
    expfn = runner.make_expectation_fn(runner.zero_state(), obs)

    def f(th):
        return expfn([[H.set_parameterizedHam({var: th}), T]])
    return (f(theta + eps) - f(theta - eps)) / (2.0 * eps)


# ── PART 1: ZNE mitigation ────────────────────────────────────────────────────

def zne(name, build, T2_base, n_sample):
    H, n, obs, var = build()
    T, x_val = 0.5, 0.7
    clean_runner = NoisyQuTiPRunner(n, noise=None)
    ideal = psr_grad(H, var, x_val, T, clean_runner, obs, n_sample)   # target (a)

    # PSR at amplified dephasing rates c·Γ0  (T2 = T2_base / c)
    cs = [1, 2, 3]
    gs = []
    for c in cs:
        runner = NoisyQuTiPRunner(n, noise=NoiseModel(n_qubits=n, T2=T2_base / c))
        gs.append(psr_grad(H, var, x_val, T, runner, obs, n_sample))

    g_lin = 2 * gs[0] - gs[1]                       # Richardson, points c=1,2
    # quadratic Lagrange extrapolation to c=0 through (1,2,3)
    g_quad = 3 * gs[0] - 3 * gs[1] + gs[2]

    print(f"\n=== ZNE on {name}   T2_base={T2_base} "
          f"(base T/T2={T/T2_base:.2f}) ===")
    print(f"  ideal (noiseless) gradient      = {ideal:+.5f}   [target]")
    for c, g in zip(cs, gs):
        print(f"  PSR @ c={c} (T2={T2_base/c:.3f}, T/T2={T/(T2_base/c):.2f}) "
              f"= {g:+.5f}   rel_err={(g-ideal)/ideal:+.1%}")
    print(f"  ZNE linear  (2·g1−g2)           = {g_lin:+.5f}   "
          f"rel_err={(g_lin-ideal)/ideal:+.1%}")
    print(f"  ZNE quadratic (3·g1−3·g2+g3)    = {g_quad:+.5f}   "
          f"rel_err={(g_quad-ideal)/ideal:+.1%}")


# ── PART 2: FD step-size tradeoff ─────────────────────────────────────────────

def fd_step(name, build, T2):
    H, n, obs, var = build()
    T, x_val = 0.5, 0.7
    runner = NoisyQuTiPRunner(n, noise=NoiseModel(n_qubits=n, T2=T2))
    # "true" noisy-landscape gradient via very fine ε
    truth = fd_grad(H, var, x_val, T, runner, obs, 1e-3)

    print(f"\n=== FD step size on {name}   T2={T2} (T/T2={T/T2:.2f}), "
          f"noisy-landscape grad={truth:+.5f} ===")
    print(f"{'ε':>7}{'FD_exact':>11}{'|bias|':>10}{'~bias/ε²':>10}"
          f"{'var∝1/(2ε)²':>13}")
    for eps in (0.01, 0.05, 0.1, 0.3, 0.5, 1.0):
        g = fd_grad(H, var, x_val, T, runner, obs, eps)
        bias = abs(g - truth)
        var_amp = 1.0 / (2 * eps) ** 2
        print(f"{eps:>7.2f}{g:>11.5f}{bias:>10.5f}{bias/eps**2:>10.3f}"
              f"{var_amp:>13.1f}")
    print("  bias grows ∝ ε² (truncation); variance amplification grows ∝ 1/ε².")
    print("  On hardware ε ≥ ε_min (DAC/calibration resolution) → bias floored "
          "from below; PSR's fixed π/2 shift needs no such fine resolution.")


def main():
    print("#" * 70)
    print("PART 1 — PSR dephasing bias is mitigable by zero-noise extrapolation")
    print("#" * 70)
    zne("1q  <Z0>", build_1q, T2_base=5.0, n_sample=1500)
    zne("2q  <Z0Z1>", build_2q, T2_base=5.0, n_sample=1)

    print("\n" + "#" * 70)
    print("PART 2 — FD step size: bias (∝ε²) vs variance (∝1/ε²)")
    print("#" * 70)
    fd_step("1q  <Z0>", build_1q, T2=5.0)
    fd_step("2q  <Z0Z1>", build_2q, T2=5.0)


if __name__ == "__main__":
    main()
