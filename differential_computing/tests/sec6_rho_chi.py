"""
sec6_rho_chi.py — P0-A.1: ρ = 2χ for Pauli tangents + the compression condition.

ρ = diam(A)/Σ_j|v_j|,  χ = diam(A)/D1,  D1 = Σ_j|v_j|·diam(H_j),  A = Σ_j v_j H_j.
For single-Pauli generators diam(H_j)=2 ⇒ D1 = 2Σ|v_j| ⇒ ρ = 2χ.

COMPRESSION (small χ) = failure of JOINT EXTREMIZABILITY of the weighted sum Σ v_j H_j
(the joint value cannot reach Σ|v_j|·diam(H_j)).  It is reachable by EITHER of two
independent mechanisms (SEC6_FOLLOWUP C1 — NOT "requires non-commutativity"):
  (i)  shared-support cancellation within a COMMUTING family — the telescoping tangent
       Σ_j (Z_j − Z_{j+1}) is fully commuting yet χ = O(1/m) (it telescopes to Z_0−Z_m,
       diam(A)=4, but D1 = Σ_j diam(Z_j−Z_{j+1}) = 4m);
  (ii) anticommuting / non-commuting contraction (Heisenberg bonds; X_a with Z_aZ_b).
Sign flips alone do NOT compress: sign-alt ZZ on a chain is still jointly extremizable
(each bond value is independently ±1) ⇒ χ = 1, identical to uniform ZZ.
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


def op1(P, i, n):
    o = [I] * n; o[i] = P; return qp.tensor(o)


def diam(A):
    e = A.eigenenergies(); return float(e[-1] - e[0])


def measure(gens):
    """gens = list of (v_j, H_j) where H_j is the full (possibly multi-term) generator.
    D1 = Σ|v_j|·diam(H_j) uses each generator's OWN diameter (not a fixed Pauli 2), so a
    generator like Z_j−Z_{j+1} contributes diam 4. Reduces to 2Σ|v| for single Paulis."""
    A = sum(v * H for v, H in gens)
    d = diam(A)
    L1 = sum(abs(v) for v, _ in gens)
    D1 = sum(abs(v) * diam(H) for v, H in gens)
    return dict(diam=d, sum_abs_v=L1, D1=D1, rho=d / L1, chi=d / D1, rho_over_chi=(d / L1) / (d / D1))


def main():
    n = 6
    tangents = {
        # single-Pauli generators (diam 2) → ρ = 2χ exactly
        "uniform_ZZ": [(1.0, op2(Z, i, Z, i + 1, n)) for i in range(n - 1)],
        "sign_alt_ZZ": [((-1.0) ** i, op2(Z, i, Z, i + 1, n)) for i in range(n - 1)],
        "Heisenberg_chain": [(1.0, op2(P, i, P, i + 1, n)) for i in range(n - 1) for P in (X, Y, Z)],
        "Heisenberg_1bond": [(1.0, op2(P, 0, P, 1, n)) for P in (X, Y, Z)],
        # COMMUTING but compressed: telescoping generators H_j = Z_j − Z_{j+1} (diam 4 each)
        "telescoping_Z": [(1.0, op1(Z, j, n) - op1(Z, j + 1, n)) for j in range(n - 1)],
    }
    out = {"n_qubits": n,
           "note": "chi=diam(A)/D1, D1=Sum|v_j|diam(H_j); rho=diam/Sum|v|; rho=2chi only for "
                   "single-Pauli generators. Compression=failure of joint extremizability.",
           "tangents": {}}
    print(f"{'tangent':20s} {'diam':>7} {'S|v|':>6} {'D1':>6} {'rho':>6} {'chi':>6} {'rho/chi':>8}")
    for name, t in tangents.items():
        m = measure(t); out["tangents"][name] = m
        print(f"{name:20s} {m['diam']:7.2f} {m['sum_abs_v']:6.1f} {m['D1']:6.1f} "
              f"{m['rho']:6.3f} {m['chi']:6.3f} {m['rho_over_chi']:8.3f}")
    os.makedirs(FIGDIR, exist_ok=True)
    json.dump(out, open(os.path.join(FIGDIR, "sec6_rho_chi.json"), "w"), indent=2, default=float)
    pauli = ["uniform_ZZ", "sign_alt_ZZ", "Heisenberg_chain", "Heisenberg_1bond"]
    ratios = [out["tangents"][k]["rho_over_chi"] for k in pauli]
    tel = out["tangents"]["telescoping_Z"]
    print(f"\nρ/χ = {min(ratios):.3f}–{max(ratios):.3f} on single-Pauli tangents (expect 2.000 exactly).")
    print(f"COMPRESSION CONDITION (C1): sign-alt ZZ == uniform ZZ (χ=1) — sign flips on a "
          f"jointly-extremizable family don't compress. But telescoping Σ(Z_j−Z_{{j+1}}) is "
          f"COMMUTING and DOES compress: χ={tel['chi']:.3f}=O(1/m) (diam {tel['diam']:.0f}, D1 "
          f"{tel['D1']:.0f}). ⇒ condition = joint-extremizability failure, NOT non-commutativity.")


if __name__ == "__main__":
    main()
