"""
h2_vqe_psr_vs_fd.py — reproduce the H2 VQE of Leng et al. 2022 (arXiv:2210.15812,
Fig 2b) and compare PSR vs FD gradients under shot noise (+ optional decoherence).

Their finding (Fig 2b): under a small observation mini-batch (b_obs), finite
difference and SPSA DO NOT converge because shot noise is amplified by the small
ε, while their parameter-shift method converges to the H2 ground energy.  Their
public code (github.com/YilingQiao/diffquantum) has no explicit noise model — the
paper's hardware runs carried implicit real-device noise.  Here we reproduce the
comparison with an EXPLICIT, controlled noise budget.

Task: minimize <H_H2> over an analog ansatz |ψ(v)>=exp(-iT Σ_k v_k G_k)|00>.
  H_H2 = α0 I + α1 Z0Z1 + α2 X0X1 + α3 Z0 + α4 Z1 (standard 2-qubit H2, E0=-1.8355).
Energy is measured with b_obs shots PER Pauli term (the VQE measurement model).
Per-parameter gradient: PSR (Algorithm 1, n_sample=1 stochastic) vs FD (floored ε).
SGD on v; track the TRUE (noiseless) energy → E0.  Averaged over seeds.

Run:  conda run -n qec_pg python differential_computing/tests/h2_vqe_psr_vs_fd.py
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

I = qp.qeye(2); X, Y, Z = qp.sigmax(), qp.sigmay(), qp.sigmaz()
def t2(a, b): return qp.tensor(a, b)

# H2 Hamiltonian (5-term form, standard 2-qubit H2 coefficients)
A0, A1, A2, A3, A4 = -0.4804, 0.5716, 0.0910, 0.3435, -0.4347
H_H2 = (A0 * t2(I, I) + A1 * t2(Z, Z) + A2 * t2(X, X)
        + A3 * t2(Z, I) + A4 * t2(I, Z))
E0 = float(np.min(H_H2.eigenenergies()))
# Pauli terms to measure (skip identity; add α0 as a constant offset)
PAULI_TERMS = [(A1, t2(Z, Z)), (A2, t2(X, X)), (A3, t2(Z, I)), (A4, t2(I, Z))]

T = 1.0
PSI0 = qp.tensor(qp.basis(2, 0), qp.basis(2, 0))

# SimuQ ansatz generators (each a single Pauli string → single-term PSR kick)
_qs = QSystem(); _q = [Qubit(_qs) for _ in range(2)]
GENS_SIMUQ = [_q[0].X, _q[1].X, _q[0].Z * _q[1].Z, _q[0].X * _q[1].X,
              _q[0].Y, _q[1].Y]
GENS_QP = [t2(X, I), t2(I, X), t2(Z, Z), t2(X, X), t2(Y, I), t2(I, Y)]
NP = len(GENS_SIMUQ)


def H_of_v(v):
    """Full ansatz Hamiltonian Σ v_k G_k as a SimuQ Parametrized/TI Hamiltonian."""
    H = float(v[0]) * GENS_SIMUQ[0]
    for k in range(1, NP):
        H = H + float(v[k]) * GENS_SIMUQ[k]
    return H


def H_param_k(v, k):
    """Single-symbol Hamiltonian: v_k symbolic, others fixed → for PSR on param k."""
    sym = sp.Symbol("vk")
    H = sym * GENS_SIMUQ[k]
    for j in range(NP):
        if j != k:
            H = H + float(v[j]) * GENS_SIMUQ[j]
    return H


def true_energy(v):
    Hgen = sum(vk * Gk for vk, Gk in zip(v, GENS_QP))
    s = (-1j * T * Hgen).expm() * PSI0
    return float(qp.expect(H_H2, s).real)


def energy_expfn(runner, b_obs, rng):
    """expfn(H_list) → shot-noisy <H_H2> on the branch final state."""
    def expfn(H_list):
        rho = runner.run_sequence(H_list, PSI0)
        e = A0
        for coeff, P in PAULI_TERMS:
            ev = float(qp.expect(P, rho).real)
            ev = min(1.0, max(-1.0, ev))
            k = rng.binomial(b_obs, 0.5 * (1 + ev))
            e += coeff * (2.0 * k / b_obs - 1.0)
        return e
    return expfn


def psr_grad(v, runner, b_obs, rng, seed):
    g = np.zeros(NP)
    expfn = energy_expfn(runner, b_obs, rng)
    for k in range(NP):
        np.random.seed(seed + k)        # stochastic τ (one sample) per component
        programs = observable_program_generator(
            H_param_k(v, k), T, n_sample=1, n_repetition=1,
            diff_var="vk", value=float(v[k]))
        g[k] = combine_gradient_results(programs, expfn, T)
    return g


def fd_grad(v, runner, b_obs, eps, rng):
    expfn = energy_expfn(runner, b_obs, rng)
    g = np.zeros(NP)
    for k in range(NP):
        vp = v.copy(); vp[k] += eps
        vm = v.copy(); vm[k] -= eps
        fp = expfn([[H_of_v(vp), T]])
        fm = expfn([[H_of_v(vm), T]])
        g[k] = (fp - fm) / (2 * eps)
    return g


def descend(method, v0, runner, b_obs, eta, eps, n_epochs, seed):
    v = v0.copy()
    energies = [true_energy(v)]
    rng = np.random.default_rng(seed)
    for ep in range(n_epochs):
        if method == "PSR":
            g = psr_grad(v, runner, b_obs, rng, seed=seed + 7 * ep)
        else:
            g = fd_grad(v, runner, b_obs, eps, rng)
        v = v - eta * g
        energies.append(true_energy(v))
    return np.array(energies)


def main():
    b_obs, eta, n_epochs, seeds = 100, 0.10, 60, 4
    fd_eps_list = [0.01, 0.1, 0.5]      # FD's ε dilemma: small→variance, large→bias
    runner = NoisyQuTiPRunner(2, noise=None)        # shot noise only (their setup)

    print(f"H2 VQE — PSR vs FD.  E0={E0:.4f}.  {NP} params, T={T}, "
          f"b_obs={b_obs} shots/term, η={eta}, {n_epochs} epochs, {seeds} seeds.\n"
          f"Noise: shot noise only (reproducing Fig 2b). PSR has NO ε; FD swept "
          f"over ε.\n")

    rng0 = np.random.RandomState(3)
    v0 = rng0.uniform(-1.0, 1.0, NP)

    runs = [("PSR", None)] + [("FD", e) for e in fd_eps_list]
    res = {}
    for method, eps in runs:
        key = "PSR" if method == "PSR" else f"FD ε={eps}"
        E = np.zeros((seeds, n_epochs + 1))
        for s in range(seeds):
            E[s] = descend(method, v0, runner, b_obs, eta, eps or 0.1,
                           n_epochs, seed=10 + s)
        res[key] = E

    keys = list(res)
    hdr = "".join(f"{k:>18}" for k in keys)
    print(f"{'epoch':>6} |{hdr}")
    for ep in (0, 10, 20, 40, 60):
        row = "".join(f"{res[k][:, ep].mean():>11.4f}±{res[k][:, ep].std():>5.3f}"
                      for k in keys)
        print(f"{ep:>6} |{row}")

    print(f"\n{'method':>12}{'final energy':>15}{'gap to E0':>12}")
    for k in keys:
        fe = res[k][:, -1].mean()
        print(f"{k:>12}{fe:>15.4f}{fe - E0:>12.4f}")
    print(f"\nE0 (true ground) = {E0:.4f}.  Expect: PSR converges (no ε); FD at "
          f"small ε=0.01 STALLS\n(shot noise amplified by small ε — the Fig-2b "
          f"finding); FD needs a tuned ε to work,\nand even then matches but "
          f"does not beat PSR — which needs no tuning.")


if __name__ == "__main__":
    main()
