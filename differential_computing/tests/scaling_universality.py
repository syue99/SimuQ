"""
scaling_universality.py — is PSR's dephasing attenuation a UNIVERSAL factor?

Under dephasing, the analog PSR gradient is attenuated: PSR_noisy ≈ λ · grad_ideal.
The classical prefactor (T·du/dv) is already applied (we compute it classically),
so λ is the pure QUANTUM attenuation.  Question (user): is λ a universal function
of T/T2* — so we could rescale PSR by 1/λ(T/T2*) and recover the ideal gradient
across ANY case — or is it system-dependent (different per Hamiltonian/observable)?

We measure λ_actual = PSR_noisy / grad_ideal (exact expectations, large n_sample →
no τ/shot noise) for several CASES (different H, observable, evolution time, point),
sweeping T2, and plot λ vs T/T2*.  If the cases COLLAPSE onto one curve, a universal
rescaling works; if they SCATTER, it's system-dependent.  We overlay a candidate
model λ_model = exp(−T/T2*) and report how well rescaling by it recovers the ideal.

Run:  conda run -n qec_pg python differential_computing/tests/scaling_universality.py
"""

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
from observable_program_generator import observable_program_generator
from combine_gradient import combine_gradient_results
from noisy_qutip import NoisyQuTiPRunner
from noise_model import NoiseModel


def build_case(kind):
    x = sp.Symbol("x")
    if kind == "1q-Z":
        qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
        H = x * q[0].Z + q[0].X
        return H, "x", qp.tensor(qp.sigmaz(), qp.qeye(2)), 2
    if kind == "1q-sin":
        qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
        H = sp.sin(x) * q[0].X + 0.7 * x * q[0].Z
        return H, "x", qp.tensor(qp.sigmaz(), qp.qeye(2)), 2
    if kind == "2q-ZZ":
        qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
        H = sp.sin(2 * x) * q[0].Z * q[1].Z + sp.sin(2 * x) * q[0].X \
            + sp.sin(2 * x) * q[1].X
        return H, "x", qp.tensor(qp.sigmaz(), qp.sigmaz()), 2
    if kind == "2q-X":
        qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
        H = x * q[0].Z * q[1].Z + q[0].X + q[1].X
        return H, "x", qp.tensor(qp.sigmaz(), qp.qeye(2)), 2
    raise ValueError(kind)


def psr_grad(H, var, x_val, T, runner, n_qubits, obs, n_sample=200, seed=1):
    psi0 = qp.tensor([qp.basis(2, 0)] * n_qubits)
    expfn = runner.make_expectation_fn(psi0, obs)
    np.random.seed(seed)
    progs = observable_program_generator(H, T, n_sample=n_sample, n_repetition=1,
                                         diff_var=var, value=x_val)
    return combine_gradient_results(progs, expfn, T)


def main():
    cases = [
        ("1q-Z",   0.6, 2.0, "#1f77b4"),
        ("1q-sin", 0.8, 2.0, "#ff7f0e"),
        ("2q-ZZ",  0.7, 1.0, "#2ca02c"),
        ("2q-X",   0.5, 2.0, "#d62728"),
    ]
    T2_list = [None, 8.0, 4.0, 2.0, 1.0, 0.7, 0.5]

    print("λ_actual = PSR_noisy / grad_ideal  (pure quantum attenuation).\n")
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=150)
    for kind, x_val, T, c in cases:
        H, var, obs, nq = build_case(kind)
        clean = NoisyQuTiPRunner(nq, noise=None)
        g_ideal = psr_grad(H, var, x_val, T, clean, nq, obs)
        xs, lam = [], []
        for T2 in T2_list:
            if T2 is None:
                xs.append(0.0); lam.append(1.0); continue
            runner = NoisyQuTiPRunner(nq, noise=NoiseModel(n_qubits=nq, T2=T2))
            g_noisy = psr_grad(H, var, x_val, T, runner, nq, obs)
            xs.append(T / T2); lam.append(g_noisy / g_ideal)
        xs, lam = np.array(xs), np.array(lam)
        axL.plot(xs, lam, "o-", color=c, lw=2, label=f"{kind} (T={T})")
        # rescaled recovery: (PSR_noisy / λ_model) / ideal  with λ_model=exp(-T/T2*)
        recov = lam / np.exp(-xs)
        axR.plot(xs, recov, "o-", color=c, lw=2, label=kind)
        print(f"{kind:>8} (T={T}): λ at T/T2*=" +
              ", ".join(f"{x:.2f}→{l:.3f}" for x, l in zip(xs[1:], lam[1:])))

    xx = np.linspace(0, max(2.0, 1.0), 50)
    axL.plot(np.linspace(0, 1.2, 50), np.exp(-np.linspace(0, 1.2, 50)),
             "k--", lw=1.4, label=r"model $e^{-T/T_2^*}$")
    axL.set_xlabel(r"$T/T_2^*$"); axL.set_ylabel(r"$\lambda = $ PSR$_{noisy}$/ideal")
    axL.set_title("(A) attenuation λ vs T/T2* — do cases collapse?")
    axL.legend(frameon=False, fontsize=8)

    axR.axhline(1.0, color="k", ls="--", lw=1.2, label="perfect recovery")
    axR.set_xlabel(r"$T/T_2^*$")
    axR.set_ylabel(r"rescaled / ideal  $= \lambda / e^{-T/T_2^*}$")
    axR.set_title(r"(B) after rescaling by $1/e^{-T/T_2^*}$ — universal?")
    axR.legend(frameon=False, fontsize=8)

    fig.suptitle("Is PSR's dephasing attenuation a universal factor of T/T2*?  "
                 "If the curves collapse (A) and\nrescaling lands at 1 (B), a known "
                 "1/λ(T/T2*) correction recovers the ideal gradient for any case",
                 fontsize=9.3)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "figures", "scaling_universality.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
