"""
vqe_noisy_comparison.py — PSR vs FD in the optimization loop under DECOHERENCE,
and on a HARDER (sharper) landscape.

Two questions from the H2 VQE follow-up:
  (A) Add our explicit decoherence (dephasing + gate error) on top of shot noise
      — does PSR still converge in the loop?
  (B) H2 at T=1 is an easy single-basin landscape.  Invent a harder one: a longer
      evolution time T sharpens the cost surface, so FD's ε is squeezed from both
      sides — shots force ε LARGE (else variance), sharp features need ε SMALL
      (else aliasing/wrong direction) → no good ε.  PSR has no ε.

Same H2 energy target (E0=-1.8355), analog ansatz exp(-iT Σ v_k G_k)|00>, b_obs
shots/Pauli-term, per-parameter PSR (stochastic n_sample=1) vs FD swept over ε.
Track the TRUE (noiseless) energy → E0.

Run:  conda run -n qec_pg python differential_computing/tests/vqe_noisy_comparison.py
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

A0, A1, A2, A3, A4 = -0.4804, 0.5716, 0.0910, 0.3435, -0.4347
H_H2 = (A0 * t2(I, I) + A1 * t2(Z, Z) + A2 * t2(X, X)
        + A3 * t2(Z, I) + A4 * t2(I, Z))
E0 = float(np.min(H_H2.eigenenergies()))
PAULI_TERMS = [(A1, t2(Z, Z)), (A2, t2(X, X)), (A3, t2(Z, I)), (A4, t2(I, Z))]
PSI0 = qp.tensor(qp.basis(2, 0), qp.basis(2, 0))

_qs = QSystem(); _q = [Qubit(_qs) for _ in range(2)]
GENS_SIMUQ = [_q[0].X, _q[1].X, _q[0].Z * _q[1].Z, _q[0].X * _q[1].X,
              _q[0].Y, _q[1].Y]
GENS_QP = [t2(X, I), t2(I, X), t2(Z, Z), t2(X, X), t2(Y, I), t2(I, Y)]
NP = len(GENS_SIMUQ)


def H_of_v(v):
    H = float(v[0]) * GENS_SIMUQ[0]
    for k in range(1, NP):
        H = H + float(v[k]) * GENS_SIMUQ[k]
    return H


def H_param_k(v, k):
    sym = sp.Symbol("vk")
    H = sym * GENS_SIMUQ[k]
    for j in range(NP):
        if j != k:
            H = H + float(v[j]) * GENS_SIMUQ[j]
    return H


def true_energy(v, T):
    Hgen = sum(vk * Gk for vk, Gk in zip(v, GENS_QP))
    s = (-1j * T * Hgen).expm() * PSI0
    return float(qp.expect(H_H2, s).real)


def energy_expfn(runner, b_obs, rng):
    def expfn(H_list):
        rho = runner.run_sequence(H_list, PSI0)
        e = A0
        for coeff, P in PAULI_TERMS:
            ev = min(1.0, max(-1.0, float(qp.expect(P, rho).real)))
            k = rng.binomial(b_obs, 0.5 * (1 + ev))
            e += coeff * (2.0 * k / b_obs - 1.0)
        return e
    return expfn


def psr_grad(v, T, runner, b_obs, rng, seed):
    g = np.zeros(NP)
    expfn = energy_expfn(runner, b_obs, rng)
    for k in range(NP):
        np.random.seed(seed + k)
        programs = observable_program_generator(
            H_param_k(v, k), T, n_sample=1, n_repetition=1,
            diff_var="vk", value=float(v[k]))
        g[k] = combine_gradient_results(programs, expfn, T)
    return g


def fd_grad(v, T, runner, b_obs, eps, rng):
    expfn = energy_expfn(runner, b_obs, rng)
    g = np.zeros(NP)
    for k in range(NP):
        vp = v.copy(); vp[k] += eps
        vm = v.copy(); vm[k] -= eps
        g[k] = (expfn([[H_of_v(vp), T]]) - expfn([[H_of_v(vm), T]])) / (2 * eps)
    return g


def descend(method, eps, v0, T, runner, b_obs, eta, n_epochs, seed):
    v = v0.copy()
    E = [true_energy(v, T)]
    rng = np.random.default_rng(seed)
    for ep in range(n_epochs):
        if method == "PSR":
            g = psr_grad(v, T, runner, b_obs, rng, seed=seed + 7 * ep)
        else:
            g = fd_grad(v, T, runner, b_obs, eps, rng)
        v = v - eta * g
        E.append(true_energy(v, T))
    return np.array(E)


def run_scenario(title, T, noise, fd_eps_list, eta, b_obs=100, n_epochs=40,
                 seeds=3, v0=None):
    runner = NoisyQuTiPRunner(2, noise=noise)
    if v0 is None:
        v0 = np.random.RandomState(3).uniform(-1.0, 1.0, NP)
    runs = [("PSR", None)] + [("FD", e) for e in fd_eps_list]
    res = {}
    for method, eps in runs:
        key = "PSR (no ε)" if method == "PSR" else f"FD ε={eps}"
        E = np.zeros((seeds, n_epochs + 1))
        for s in range(seeds):
            E[s] = descend(method, eps or 0.1, v0, T, runner, b_obs, eta,
                           n_epochs, seed=10 + s)
        res[key] = E

    print(f"\n=== {title} ===")
    print(f"  T={T}, noise={'none' if noise is None else 'dephasing+gate'}, "
          f"b_obs={b_obs}, η={eta}, {n_epochs} epochs, {seeds} seeds.  E0={E0:.4f}")
    print(f"  {'method':>12}{'final E':>11}{'gap to E0':>11}{'epoch≤E0+0.05':>14}")
    for k, E in res.items():
        fe = E[:, -1].mean()
        reached = np.where(E.mean(0) <= E0 + 0.05)[0]
        ep_reach = int(reached[0]) if len(reached) else -1
        tag = f"{ep_reach}" if ep_reach >= 0 else "never"
        print(f"  {k:>12}{fe:>11.4f}{fe - E0:>11.4f}{tag:>14}")
    return res


def main():
    v0 = np.random.RandomState(3).uniform(-1.0, 1.0, NP)
    full_noise = NoiseModel(n_qubits=2, T2=5.0, gate_error_1q=1e-4,
                            gate_error_2q=1e-3, gate_coherent_frac=0.5)

    # (A) H2 + decoherence (easy landscape, T=1)
    run_scenario("(A) H2 VQE + decoherence (T=1, dephasing T2=5 + gate err)",
                 T=1.0, noise=full_noise, fd_eps_list=[0.01, 0.1], eta=0.10,
                 v0=v0)

    # (B) harder: sharper landscape (longer T) under full noise — FD's ε squeezed
    run_scenario("(B) HARDER: sharp landscape (T=3) + full noise",
                 T=3.0, noise=full_noise, fd_eps_list=[0.01, 0.1, 0.5], eta=0.05,
                 v0=v0)

    print(f"\nReading: 'epoch≤E0+0.05' = first epoch the mean energy gets within "
          f"0.05 of the\nground state (convergence speed).  PSR has no ε to tune. "
          f"On the harder (sharp)\nlandscape, FD's ε is squeezed — small ε → "
          f"variance, large ε → aliasing/bias —\nso no ε converges as well as PSR.")


if __name__ == "__main__":
    main()
