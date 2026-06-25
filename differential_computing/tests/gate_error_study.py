"""
gate_error_study.py — does the reference-anchored kick gate error bias PSR?

Gate model anchored to Evered et al. 2026 (arXiv:2604.25987): the residual CZ
error after loss post-selection is Z/phase- and scattering-dominated (T2*,
Doppler, |r'> coupling), NOT flip (X); 1q gate errors are negligible.  So the
kick gate error is a Z-type channel on the kicked qubits — coherent Z over-
rotation (Doppler/laser phase) + incoherent Z-dephasing (T2*/scattering) —
calibrated to ε: 1q kicks 1e-4, 2q kicks 1e-3.  Only PSR has kicks; FD has none,
so this is a PSR-specific cost.

Two questions:
  1. At the reference fidelities, is the PSR gradient bias negligible? (predict
     yes — Z-type error has small effect on a Z-basis observable.)
  2. Is the COHERENT piece the one to watch — a systematic, non-averaging bias
     (vs the incoherent piece which should be benign)?  Sweep gate_coherent_frac.

Ground truth = fine-ε FD on the noiseless landscape (FD has no kick, so its
gate-error is zero; the kick error is purely a PSR effect, measured as PSR's
departure from the true gradient).

Run:  conda run -n qec_pg python differential_computing/tests/gate_error_study.py
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
        H, T, n_sample=n_sample, n_repetition=1, diff_var=var, value=float(theta))
    return combine_gradient_results(programs, expfn, T)


def truth_grad(H, var, theta, T, n, obs, eps=1e-2):
    clean = NoisyQuTiPRunner(n, noise=None)
    expfn = clean.make_expectation_fn(clean.zero_state(), obs)

    def f(th):
        return expfn([[H.set_parameterizedHam({var: float(th)}), T]])
    return (f(theta + eps) - f(theta - eps)) / (2.0 * eps)


def run_model(name, build, n_sample):
    H, n, obs, var = build()
    T, x_val = 0.5, 0.7
    truth = truth_grad(H, var, x_val, T, n, obs)

    print(f"\n=== {name}   reference fidelities: 1q ε=1e-4, 2q ε=1e-3   "
          f"(truth={truth:+.5f}) ===")
    print(f"{'coherent_frac':>14}{'PSR':>11}{'abs_bias':>11}{'rel_bias':>10}{'sign':>6}")
    for fc in (0.0, 0.25, 0.5, 0.75, 1.0):
        noise = NoiseModel(n_qubits=n, gate_error_1q=1e-4, gate_error_2q=1e-3,
                           gate_coherent_frac=fc)
        runner = NoisyQuTiPRunner(n, noise=noise)
        g = psr_grad(H, var, x_val, T, runner, obs, n_sample)
        rel = (g - truth) / truth if abs(truth) > 1e-9 else float("nan")
        sign = "ok" if np.sign(g) == np.sign(truth) else "FLIP"
        print(f"{fc:>14.2f}{g:>11.5f}{abs(g - truth):>11.5f}{rel:>+9.2%}{sign:>6}")

    # stress test: exaggerate the gate error 100x to see where coherent bites
    print(f"  -- stress: 2q ε=0.1 (100x), pure coherent vs pure incoherent --")
    for fc, lbl in ((1.0, "coherent"), (0.0, "incoherent")):
        noise = NoiseModel(n_qubits=n, gate_error_1q=1e-2, gate_error_2q=1e-1,
                           gate_coherent_frac=fc)
        runner = NoisyQuTiPRunner(n, noise=noise)
        g = psr_grad(H, var, x_val, T, runner, obs, n_sample)
        rel = (g - truth) / truth if abs(truth) > 1e-9 else float("nan")
        print(f"  {lbl:>12}: PSR={g:+.5f}  rel_bias={rel:+.2%}")


def main():
    run_model("1q  <Z0>", build_1q, n_sample=1500)
    run_model("2q  <Z0Z1>", build_2q, n_sample=1)
    print("\nReference fidelities (1e-4 / 1e-3): expect rel_bias ~ noiseless "
          "residual → gate error is\nbenign. Stress test: watch whether the "
          "COHERENT piece gives a systematic bias the incoherent does not.")


if __name__ == "__main__":
    main()
