"""
scaling_observable_weight.py — does PSR's dephasing bias scale with system size?

The generality sweep flagged a sign flip on a 4-qubit instance with a multi-qubit
observable.  Hypothesis: under per-qubit dephasing, a weight-k observable (Z on k
qubits) decoheres at ~k×, so the EFFECTIVE T/T2* scales with observable weight —
PSR's benign regime shrinks as you measure more qubits, even at fixed per-qubit
T2.  This separates "real scaling law" from "MC artifact".

Controlled family: H_n = sin(2x)·(Σ Z_iZ_{i+1} + Σ X_i) on an n-qubit chain.
For each n, compare a LOCAL observable (Z_0, weight 1) vs a GLOBAL one (full
Z-string, weight n), under the SAME per-qubit dephasing T2.  High n_sample so MC
noise is not the story.  Report λ=PSR/truth and sign.

Run:  conda run -n qec_pg python differential_computing/tests/scaling_observable_weight.py
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


def chain_H(n):
    x = sp.Symbol("x")
    qs = QSystem(); q = [Qubit(qs) for _ in range(n)]
    H = None
    for i in range(n - 1):
        t = sp.sin(2 * x) * q[i].Z * q[i + 1].Z
        H = t if H is None else H + t
    for i in range(n):
        H = H + sp.sin(2 * x) * q[i].X
    return H


def z_obs(n, sites):
    ops = [qp.sigmaz() if k in sites else qp.qeye(2) for k in range(n)]
    return qp.tensor(ops) if n > 1 else ops[0]


def truth_grad(H, theta, T, n, obs, eps=1e-2):
    clean = NoisyQuTiPRunner(n, noise=None)
    expfn = clean.make_expectation_fn(clean.zero_state(), obs)

    def f(th):
        return expfn([[H.set_parameterizedHam({"x": float(th)}), T]])
    return (f(theta + eps) - f(theta - eps)) / (2.0 * eps)


def psr_grad(H, theta, T, runner, obs, n_sample, seed=5):
    expfn = runner.make_expectation_fn(runner.zero_state(), obs)
    np.random.seed(seed)
    programs = observable_program_generator(
        H, T, n_sample=n_sample, n_repetition=1, diff_var="x", value=float(theta))
    return combine_gradient_results(programs, expfn, T)


def main():
    T, theta, T2 = 0.5, 0.7, 5.0       # per-qubit T/T2* = 0.10
    n_sample = 400

    print(f"Chain H_n = sin(2x)(ΣZ_iZ_i+1 + ΣX_i),  T={T}, per-qubit "
          f"T2={T2} (T/T2*={T/T2:.2f}), n_sample={n_sample}\n")
    print(f"{'n':>3}  {'observable':<16}{'weight':>7}{'truth':>10}"
          f"{'PSR':>10}{'λ':>8}{'sign':>6}")
    for n in (2, 3, 4):
        H = chain_H(n)
        for label, sites in (("Z0 (local)", [0]),
                             ("full Z-string", list(range(n)))):
            obs = z_obs(n, sites)
            truth = truth_grad(H, theta, T, n, obs)
            if abs(truth) < 1e-3:
                print(f"{n:>3}  {label:<16}{len(sites):>7}   (truth≈0, skip)")
                continue
            runner = NoisyQuTiPRunner(n, noise=NoiseModel(n_qubits=n, T2=T2))
            g = psr_grad(H, theta, T, runner, obs, n_sample)
            lam = g / truth
            sign = "ok" if np.sign(g) == np.sign(truth) else "FLIP"
            print(f"{n:>3}  {label:<16}{len(sites):>7}{truth:>10.4f}"
                  f"{g:>10.4f}{lam:>8.3f}{sign:>6}")

    print("\nIf λ for the full Z-string DROPS (toward 0 / negative) as n grows "
          "while the\nlocal Z0 stays ~constant → dephasing compounds with "
          "observable weight: a real\nscaling caveat (measure local observables, "
          "or the benign-regime T/T2* tightens).")


if __name__ == "__main__":
    main()
