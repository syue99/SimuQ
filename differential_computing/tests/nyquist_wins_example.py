"""
nyquist_wins_example.py — a concrete case where NYQUIST beats kick on shots.

Kick generally wins for aligned/foldable tangents (its co-located ± echo). Nyquist
wins when the tangent is NON-FOLDABLE and has a SUBADDITIVE spectral diameter —
i.e. a native non-commuting analog drive the kick must DECOMPOSE into per-term
Pauli kicks while Nyquist shifts it as one waveform.

Example: XY / hopping tangent  A = Σ_j (X_jX_{j+1} + Y_jY_{j+1})  (also Heisenberg
A = Σ(XX+YY+ZZ)). These have a free-fermion-like spectrum with MANY eigenvalues
(not a single involution ⇒ non-foldable), and diam(A) < 2·Σ|v_j| (subadditive),
so ρ = diam(A)/Σ|v_j| < 2.

Shot-cost ratio (kick/Nyquist) = 4·var/ρ², var∈[0,1] the branch shot variance.
Nyquist wins ⟺ var > ρ²/4. For aligned A=Z (ρ=2) that needs var>1 (never); for the
XY/Heisenberg drives (ρ<2) Nyquist wins at moderate-to-high entropy.

(a) ρ = diam(A)/Σ|v_j| vs system size m — subadditive (<2) for the drives.
(b) shot-cost ratio vs operating-point entropy — Nyquist-wins region for the drives.
Run: conda run -n qec_pg python differential_computing/tests/nyquist_wins_example.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import qutip as qp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
X, Y, Z, I = qp.sigmax(), qp.sigmay(), qp.sigmaz(), qp.qeye(2)


def op_on(m, j, P):
    ops = [I] * m; ops[j] = P
    return qp.tensor(ops)


def two(m, j, P, Q):
    ops = [I] * m; ops[j] = P; ops[j + 1] = Q
    return qp.tensor(ops)


def tangent(m, kind):
    """Return (A, Sigma|v_j|) for a tangent family on m qubits."""
    if kind == "aligned":                       # A = Σ Z_j (foldable, ρ=2)
        A = sum(op_on(m, j, Z) for j in range(m)); nv = m
    elif kind == "XY":                          # A = Σ (XX + YY)  (hopping)
        A = sum(two(m, j, X, X) + two(m, j, Y, Y) for j in range(m - 1)); nv = 2 * (m - 1)
    elif kind == "Heisenberg":                  # A = Σ (XX + YY + ZZ)
        A = sum(two(m, j, X, X) + two(m, j, Y, Y) + two(m, j, Z, Z) for j in range(m - 1))
        nv = 3 * (m - 1)
    return A, nv


def diam(A):
    e = A.eigenenergies()
    return float(e[-1] - e[0])


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    ms = list(range(2, 8))
    fams = {"aligned": ("aligned  $\\Sigma Z_j$", "#009E73", "o"),
            "XY": ("XY / hopping  $\\Sigma(XX{+}YY)$", "#0072B2", "^"),
            "Heisenberg": ("Heisenberg  $\\Sigma(XX{+}YY{+}ZZ)$", "#8856a7", "s")}
    rho = {k: [] for k in fams}
    print(f"{'family':>12} {'m':>3} {'diam(A)':>8} {'Σ|v|':>6} {'ρ=diam/Σ|v|':>12}")
    for k in fams:
        for m in ms:
            A, nv = tangent(m, k)
            d = diam(A); rho[k].append(d / nv)
            print(f"{k:>12} {m:>3} {d:>8.2f} {nv:>6} {d / nv:>12.3f}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    # (a) ρ vs m
    for k, (lab, c, mk) in fams.items():
        axA.plot(ms, rho[k], mk + "-", color=c, ms=6, label=lab)
    axA.axhline(2.0, color="#333", lw=1.0, ls=":")
    axA.text(ms[0], 2.03, r"$\rho=2$ (aligned/foldable — kick's regime)", fontsize=7, color="#333")
    axA.set_xlabel("qubits $m$"); axA.set_ylabel(r"$\rho=\mathrm{diam}(A)/\Sigma_j|v_j|$")
    axA.set_title("(a) subadditive diameter: the drives have $\\rho<2$", fontsize=8.5)
    axA.set_ylim(0, 2.2); axA.legend(fontsize=7); axA.grid(True, alpha=0.15)

    # (b) shot-cost ratio kick/Nyquist = 4·var/ρ²  vs operating-point entropy var
    var = np.linspace(0.0, 1.0, 100)
    for k, (lab, c, mk) in fams.items():
        rr = rho[k][-1]                          # largest m
        ratio = 4 * var / rr ** 2
        axB.plot(var, ratio, "-", color=c, lw=2, label=f"{lab.split('  ')[0]}  (ρ={rr:.2f})")
    axB.axhline(1.0, color="#999", lw=1.0, ls="--")
    axB.fill_between(var, 1.0, 12, color="#0072B2", alpha=0.08)
    axB.text(0.06, 4.0, "Nyquist wins", color="#0072B2", fontsize=10, weight="bold")
    axB.text(0.55, 0.35, "kick wins", color="#00695c", fontsize=10, weight="bold")
    axB.set_xlabel(r"branch shot variance  var $\approx 1-\langle O\rangle^2$")
    axB.set_ylabel(r"kick / Nyquist shot cost  $=4\,\mathrm{var}/\rho^2$")
    axB.set_title(r"(b) at moderate-high entropy the drives flip to Nyquist"
                  "\n(kick wins iff $\\rho>2\\sqrt{\\mathrm{var}}$)", fontsize=8.5)
    axB.set_ylim(0, 8); axB.legend(fontsize=7, loc="upper right"); axB.grid(True, alpha=0.15)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "nyquist_wins_example.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    json.dump(dict(ms=ms, rho=rho), open(os.path.join(FIGDIR, "nyquist_wins_example.json"), "w"),
              indent=2, default=float)
    print(f"\nNyquist wins (ρ<2, non-foldable native drive) at high entropy. figure: {out}")


if __name__ == "__main__":
    main()
