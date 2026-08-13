"""
sec6_rho_chi.py — P0-A.1: confirm ρ = 2χ for Pauli tangents (Hamiltonian-level).

ρ = diam(A)/Σ_j|v_j|   (used in phase_who_wins_3panel)
χ = diam(A)/D1,  D1 = Σ_j|v_j|·diam(H_j);  for Pauli diam(H_j)=2 ⇒ D1=2Σ|v_j| ⇒ ρ=2χ.
Reports both on 3 test tangents + the ratio. Caches figures/sec6_rho_chi.json.
Run: conda run -n qec_pg python differential_computing/tests/sec6_rho_chi.py
"""
import json
import os

import numpy as np
import qutip as qp

X, Y, Z, I = qp.sigmax(), qp.sigmay(), qp.sigmaz(), qp.qeye(2)
FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))


def op2(P, i, Q, j, n):
    o = [I] * n; o[i] = P; o[j] = Q; return qp.tensor(o)


def diam(A):
    e = A.eigenenergies(); return float(e[-1] - e[0])


def measure(terms):
    A = sum(c * op for c, op in terms)
    d = diam(A); L1 = sum(abs(c) for c, _ in terms); D1 = 2 * L1  # Pauli diam=2
    return dict(diam=d, sum_abs_v=L1, D1=D1, rho=d / L1, chi=d / D1, rho_over_chi=(d / L1) / (d / D1))


def main():
    n = 6
    tangents = {
        "uniform_ZZ": [(1.0, op2(Z, i, Z, i + 1, n)) for i in range(n - 1)],
        "sign_alt_ZZ": [((-1.0) ** i, op2(Z, i, Z, i + 1, n)) for i in range(n - 1)],
        "Heisenberg_chain": [(1.0, op2(P, i, P, i + 1, n)) for i in range(n - 1) for P in (X, Y, Z)],
        "Heisenberg_1bond": [(1.0, op2(P, 0, P, 1, n)) for P in (X, Y, Z)],
    }
    out = {"n_qubits": n, "note": "rho=diam/Sum|v|, chi=diam/(2 Sum|v|)=rho/2 for Pauli", "tangents": {}}
    print(f"{'tangent':20s} {'diam':>7} {'S|v|':>6} {'rho':>6} {'chi':>6} {'rho/chi':>8}")
    for name, t in tangents.items():
        m = measure(t); out["tangents"][name] = m
        print(f"{name:20s} {m['diam']:7.2f} {m['sum_abs_v']:6.1f} {m['rho']:6.3f} {m['chi']:6.3f} {m['rho_over_chi']:8.3f}")
    os.makedirs(FIGDIR, exist_ok=True)
    json.dump(out, open(os.path.join(FIGDIR, "sec6_rho_chi.json"), "w"), indent=2, default=float)
    ratios = [m["rho_over_chi"] for m in out["tangents"].values()]
    print(f"\nρ/χ = {min(ratios):.3f}–{max(ratios):.3f} (expect 2.000 exactly for Pauli). "
          f"NOTE: sign-alt ZZ == uniform ZZ (χ=1): sign flips on COMMUTING terms don't compress.")


if __name__ == "__main__":
    main()
