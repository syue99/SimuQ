"""
fig_resource_pillar.py — the resource pillar: analog-native evolution vs digital
Trotter emulation, per gradient BRANCH.

Each PSR branch is a fixed set of continuous evolutions of total logical time T
(constant pulse depth, independent of n — the C7 evidence).  Emulating the same
e^{-iHT} DIGITALLY for a local (TFIM-like) H to Trotter error eps costs, at first
order, ~ n^2 T^2 / eps two-qubit gates per branch (Trotter steps ~ n T^2/eps,
each a ZZ layer of ~n gates).  At NISQ two-qubit fidelities the cumulative gate
error saturates at a "NISQ wall" (~1/(1-F) gates), so the digital route drowns
after a few qubits while the analog route is native and constant.

Analytic (no sim); numbers labeled as assumptions.  Saves figures/fig_resource_pillar.png.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
T = 1.5                              # analog logical evolution time per branch


def gates_digital(n, eps):
    """First-order Trotter 2q-gate count per branch for a TFIM-like chain."""
    return n ** 2 * T ** 2 / (2 * eps)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    n = np.arange(2, 65)
    fig, ax = plt.subplots(figsize=(7.8, 5.4), dpi=150)

    # NISQ / cryo fidelity walls: cumulative 2q-gate error ~ #gates*(1-F) ~ 1
    for F, lab, col, y in [(0.999, "99.9% 2q (NISQ)", "#b0b0b0", 1 / (1 - 0.999)),
                           (0.9999, "99.99% 2q (cryo)", "#8a8a8a", 1 / (1 - 0.9999))]:
        ax.axhline(y, color=col, ls="--", lw=1.1)
        ax.text(2.2, y * 1.25, f"{lab}: gate-error wall", fontsize=8, color="#555")
    ax.axhspan(1 / (1 - 0.999), 1e9, color="#f6f6f6", zorder=0)

    ax.loglog(n, gates_digital(n, 1e-2), "-", color="#d62728", lw=2.4,
              label=r"digital Trotter emulation, $\varepsilon=10^{-2}$  ($\sim n^2 T^2/\varepsilon$)")
    ax.loglog(n, gates_digital(n, 1e-3), "--", color="#d62728", lw=1.8,
              label=r"digital Trotter emulation, $\varepsilon=10^{-3}$")
    ax.loglog(n, np.ones_like(n), "-", color="#00897b", lw=2.8,
              label="analog-native: 1 evolution, depth T  (constant in n)")

    # where digital crosses the NISQ wall
    ncross = np.sqrt(1 / (1 - 0.999) * 2 * 1e-2 / T ** 2)
    ax.axvline(ncross, color="#d62728", ls=":", lw=1, alpha=0.6)
    ax.text(ncross * 1.05, 3, f"digital drowns\nby n≈{ncross:.0f}", fontsize=8.5,
            color="#d62728", va="bottom")

    ax.set_xlabel("qubits  n")
    ax.set_ylabel("two-qubit entangling operations per gradient branch")
    ax.set_ylim(0.4, 1e7); ax.set_xlim(2, 64)
    ax.set_title("Resource pillar: differentiating the analog program natively vs. "
                 "emulating it digitally\n(analog pulse depth is constant in n; digital "
                 "Trotter is not, and drowns at NISQ fidelity)", fontsize=9.3)
    ax.legend(fontsize=8.3, loc="upper left")
    ax.grid(True, which="both", alpha=0.13)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig_resource_pillar.png")
    fig.savefig(out); print(f"saved: {out}")
    print(f"digital (eps=1e-2) crosses the 99.9% NISQ wall at n≈{ncross:.1f}; "
          f"gates at n=2:{gates_digital(2,1e-2):.0f}, n=10:{gates_digital(10,1e-2):.0f}, "
          f"n=50:{gates_digital(50,1e-2):.0f}")


if __name__ == "__main__":
    main()
