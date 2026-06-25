"""
maxcut_setup_check.py — set up the MaxCut QAOA from Leng et al. 2022 (Fig 2c,d)
and check an analog ansatz can reach a good cut.

Graph: 4-vertex cycle (their example), edges (0,1),(1,2),(2,3),(3,0).
Cut operator C = ½ Σ_{(j,k)∈E} (I − Z_j Z_k) = 2 − ½(Z0Z1+Z1Z2+Z2Z3+Z3Z0).
Max cut = 4 (bipartition |0101>/|1010>).  Objective: maximize <C>.

Analog ansatz: |ψ(v)> = exp(−iT Σ_k v_k G_k)|++++>, generators = the 4 ZZ cost
edges + the 4 X mixers (QAOA-style, single analog layer).  Maximize <C> over v
(noiseless, scipy) to see the achievable cut.

Run:  conda run -n qec_pg python differential_computing/tests/maxcut_setup_check.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import qutip as qp
from scipy.optimize import minimize

I = qp.qeye(2); X, Z = qp.sigmax(), qp.sigmaz()
N = 4
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]


def op(p, i, j=None):
    ops = [I] * N
    ops[i] = p
    if j is not None:
        ops[j] = p
    return qp.tensor(ops)


# cut operator and generators
C = 2.0 * qp.tensor([I] * N) - 0.5 * sum(op(Z, i, j) for i, j in EDGES)
MAXCUT = float(np.max(C.eigenenergies()))

GENS = [op(Z, i, j) for i, j in EDGES] + [op(X, i) for i in range(N)]
NP = len(GENS)
PLUS = qp.tensor([(qp.basis(2, 0) + qp.basis(2, 1)).unit()] * N)
T = 3.0


def cut(v):
    Hg = sum(vk * Gk for vk, Gk in zip(v, GENS))
    s = (-1j * T * Hg).expm() * PLUS
    return float(qp.expect(C, s).real)


def main():
    print(f"MaxCut 4-cycle.  edges={EDGES}.  max cut = {MAXCUT:.3f}")
    print(f"Ansatz: single analog layer, {NP} generators (4 ZZ + 4 X), T={T}, "
          f"init |++++> (<C>={cut(np.zeros(NP)):.3f}).\n")

    best = None
    rng = np.random.RandomState(0)
    for _ in range(20):
        v0 = rng.uniform(-1.5, 1.5, NP)
        res = minimize(lambda v: -cut(v), v0, method="BFGS",
                       options=dict(maxiter=400))
        if best is None or res.fun < best.fun:
            best = res
    achieved = -best.fun
    print(f"Best achievable cut over 20 starts: {achieved:.4f}  "
          f"(ratio {achieved / MAXCUT:.3f} of optimal {MAXCUT:.0f})")
    if achieved > 0.95 * MAXCUT:
        print("→ Ansatz nearly reaches max cut — good for the QAOA loop.")
    else:
        print("→ Ansatz reaches a partial cut; PSR-vs-FD comparison still valid "
              "(both maximize the same objective).")


if __name__ == "__main__":
    main()
