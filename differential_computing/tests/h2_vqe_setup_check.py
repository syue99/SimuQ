"""
h2_vqe_setup_check.py — set up the H2 VQE from Leng et al. 2022 (arXiv:2210.15812,
Fig 2b) and check an analog ansatz can reach the ground-state energy.

H2 Hamiltonian (their form): H = α0 I + α1 Z0Z1 + α2 X0X1 + α3 Z0 + α4 Z1.
Coefficients are the standard 2-qubit H2 (O'Malley 2016, R≈0.75Å), restricted to
this 5-term form.  Ground energy E0 = min eigenvalue.

Ansatz: single analog evolution |ψ(v)> = exp(-i T Σ_k v_k G_k)|00>, minimize the
energy <ψ|H|ψ>.  We scipy-optimize the NOISELESS energy from several starts to
confirm the ansatz reaches ~E0 (so "convergence to the ground state" is meaningful
for the PSR-vs-FD loop).

Run:  conda run -n qec_pg python differential_computing/tests/h2_vqe_setup_check.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import qutip as qp
from scipy.optimize import minimize

I = qp.qeye(2)
X, Y, Z = qp.sigmax(), qp.sigmay(), qp.sigmaz()


def op2(a, b):
    return qp.tensor(a, b)


# H2 Hamiltonian (standard 2-qubit H2, 5-term form of the paper)
A0, A1, A2, A3, A4 = -0.4804, 0.5716, 0.0910, 0.3435, -0.4347
H_H2 = (A0 * op2(I, I) + A1 * op2(Z, Z) + A2 * op2(X, X)
        + A3 * op2(Z, I) + A4 * op2(I, Z))

E0 = float(np.min(H_H2.eigenenergies()))

# analog ansatz generators
GENS = [op2(X, I), op2(I, X), op2(Z, Z), op2(X, X), op2(Y, I), op2(I, Y)]
PSI0 = qp.tensor(qp.basis(2, 0), qp.basis(2, 0))
T = 1.0


def state(v):
    Hgen = sum(vk * Gk for vk, Gk in zip(v, GENS))
    U = (-1j * T * Hgen).expm()
    return U * PSI0


def energy(v):
    s = state(v)
    return float(qp.expect(H_H2, s).real)


def main():
    print(f"H2 (5-term): α=({A0},{A1},{A2},{A3},{A4})")
    print(f"Ground-state energy E0 = {E0:.6f}")
    print(f"Ansatz: single analog evolution, {len(GENS)} generators, T={T}\n")

    best = None
    rng = np.random.RandomState(0)
    for trial in range(12):
        v0 = rng.uniform(-1.5, 1.5, len(GENS))
        res = minimize(energy, v0, method="BFGS",
                       options=dict(maxiter=500))
        if best is None or res.fun < best.fun:
            best = res
    print(f"Best ansatz energy over 12 starts: {best.fun:.6f}")
    print(f"Gap to E0: {best.fun - E0:.6f}")
    if best.fun - E0 < 1e-3:
        print("→ Ansatz reaches the ground state — good for the VQE loop.")
    else:
        print("→ Ansatz minimum sits above E0 — PSR-vs-FD comparison still valid "
              "(both descend to the ansatz minimum); note the residual gap.")
    print(f"\nOptimal v ≈ {np.round(best.x, 3)}")


if __name__ == "__main__":
    main()
