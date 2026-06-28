"""
lightcone_slope.py — is the analytic attenuation slope calculable at 100 qubits?

The slope is a sum over qubits of per-qubit terms
    term_i = ∫_0^T [<χ_i(t)|O|χ_i(t)> − <O>(T)] dt,   χ_i(t)=U(T−t)Z_i U(t)|ψ0>.
For a LOCAL observable O and finite evolution T with LOCAL interactions, only
qubits within O's Lieb–Robinson LIGHT CONE contribute — a Z on a distant qubit
cannot affect O within time T.  So the sum is dominated by O(light-cone) qubits,
INDEPENDENT of total system size → computable from a small local subsystem even
in a 100-qubit array (and measurable on-device otherwise).

We demonstrate on a chain H = Σ θ Z_i Z_{i+1} + Σ X_i, observable Z_0, by computing
each qubit's contribution |term_i| and showing it decays sharply with distance
from qubit 0.

Run:  conda run -n qec_pg python differential_computing/tests/lightcone_slope.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import qutip as qp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

I = qp.qeye(2); X, Z = qp.sigmax(), qp.sigmaz()


def emb(op, i, n):
    return qp.tensor([op if k == i else I for k in range(n)])


def chain_H(n, theta):
    H = 0
    for i in range(n - 1):
        H = H + theta * emb(Z, i, n) * emb(Z, i + 1, n)
    for i in range(n):
        H = H + emb(X, i, n)
    return H


def per_qubit_slope_terms(n, theta, T, n_grid=120):
    H = chain_H(n, theta)
    O = emb(Z, 0, n)
    psi0 = qp.tensor([qp.basis(2, 0)] * n)
    ts = np.linspace(0.0, T, n_grid)
    OT = float(qp.expect(O, (-1j * H * T).expm() * psi0).real)
    Us = [(-1j * H * t).expm() for t in ts]
    Zs = [emb(Z, i, n) for i in range(n)]
    terms = np.zeros(n)
    for i in range(n):
        integ = np.zeros(len(ts))
        for k, t in enumerate(ts):
            psit = Us[k] * psi0
            chi = (-1j * H * (T - t)).expm() * (Zs[i] * psit)
            integ[k] = float(qp.expect(O, chi).real) - OT
        terms[i] = np.trapz(integ, ts)
    return terms


def main():
    n = 8
    theta = 0.6
    fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=150)
    print(f"Chain H=Σθ Z_iZ_i+1 + ΣX_i, observable Z_0, θ={theta}, n={n} qubits.")
    print("Per-qubit contribution to the attenuation slope vs distance from Z_0:\n")
    for T, c in [(0.6, "#1f77b4"), (1.2, "#ff7f0e"), (2.0, "#d62728")]:
        terms = per_qubit_slope_terms(n, theta, T)
        tot = np.sum(terms)
        frac = np.abs(terms) / (np.sum(np.abs(terms)) + 1e-12)
        print(f"T={T}:  total slope={tot:+.3f},  |term_i|/Σ by qubit: " +
              " ".join(f"{f:.0%}" for f in frac))
        ax.semilogy(range(n), np.abs(terms) + 1e-6, "o-", color=c, lw=2,
                    label=f"T={T}")
    ax.set_xlabel("qubit index  i  (distance from observable $Z_0$)")
    ax.set_ylabel(r"$|term_i|$  (contribution to slope)")
    ax.set_title(f"Attenuation slope is LOCAL: per-qubit contribution decays with\n"
                 f"distance from the observable (light cone) — chain of {n} qubits")
    ax.legend(frameon=False, fontsize=9, title="evolution time")
    fig.tight_layout()
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "figures", "lightcone_slope.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"\nThe contribution decays with distance → only O(light-cone) qubits "
          f"matter,\nindependent of total system size: computable from a local "
          f"subsystem even at 100 qubits\n(for local O + finite T). saved: {out}")


if __name__ == "__main__":
    main()
