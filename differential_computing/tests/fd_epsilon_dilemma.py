"""
fd_epsilon_dilemma.py — a regime where FD cannot converge at ANY ε, under the
physically faithful noise model (kick = gate, no dressing-T2* during the kick).

FD's finite-difference gradient faces a fatal squeeze when noise is strong and
the landscape is sharp:
  - SMALL ε → shot noise amplified by 1/(2ε) → variance explodes (esp. near the
    minimum where the true gradient is small → SNR < 1 → it orbits).
  - LARGE ε → truncation bias ε²·f'''/6 → on a sharp landscape this dominates,
    again worst where the gradient is small → it converges to a wrong point.
So no ε works.  PSR has no ε and converges.

Regime: H2 VQE, longer evolution T (sharper cost surface) + low b_obs (high shot
variance) + dephasing T2 (faithful: evolution-only).  Sweep FD over a wide ε
range; PSR alone.  Track the ideal energy; plot energy vs epoch.

Run:  conda run -n qec_pg python differential_computing/tests/fd_epsilon_dilemma.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import vqe_noisy_comparison as q
from noisy_qutip import NoisyQuTiPRunner
from noise_model import NoiseModel


def main():
    T = 4.0                       # sharp cost surface (feature scale ~1/T)
    b_obs = 20                    # very low shots → high variance
    noise = NoiseModel(n_qubits=2, T2=5.0, gate_error_1q=1e-4,
                       gate_error_2q=1e-3, gate_coherent_frac=0.5)
    runner = NoisyQuTiPRunner(2, noise=noise)     # faithful default (kick=gate)
    v0 = np.random.RandomState(3).uniform(-1.0, 1.0, q.NP)
    eta, n_epochs, seeds = 0.04, 60, 4
    fd_eps = [0.1, 0.2, 0.3, 0.5]

    runs = [("PSR (no ε)", "PSR", None)] + [(f"FD ε={e}", "FD", e) for e in fd_eps]
    res = {}
    for label, method, eps in runs:
        E = np.zeros((seeds, n_epochs + 1))
        for s in range(seeds):
            E[s] = q.descend(method, eps or 0.1, v0, T, runner, b_obs, eta,
                             n_epochs, seed=10 + s)
        res[label] = E

    print(f"FD ε-dilemma — H2 VQE, faithful model (kick=gate).  T={T} (sharp), "
          f"b_obs={b_obs} (low), T2=5.\nE0={q.E0:.4f}, η={eta}, {n_epochs} epochs, "
          f"{seeds} seeds.\n")
    print(f"{'method':>14}{'final E':>11}{'gap to E0':>12}{'best E reached':>16}")
    for label, E in res.items():
        fe = E[:, -1].mean()
        best = E.mean(0).min()
        print(f"{label:>14}{fe:>11.4f}{fe - q.E0:>12.4f}{best:>16.4f}")

    # plot
    steps = np.arange(n_epochs + 1)
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=150)
    colors = {"PSR (no ε)": "#1f77b4"}
    fd_colors = ["#d62728", "#ff7f0e", "#9467bd", "#8c564b"]
    for i, (label, E) in enumerate(res.items()):
        mu, sd = E.mean(0), E.std(0)
        c = colors.get(label, fd_colors[(i - 1) % len(fd_colors)])
        lw = 2.6 if label.startswith("PSR") else 1.8
        ax.plot(steps, mu, label=label, color=c, lw=lw)
        ax.fill_between(steps, mu - sd, mu + sd, color=c, alpha=0.12)
    ax.axhline(q.E0, ls="--", color="k", lw=1, label=f"true ground $E_0$={q.E0:.3f}")
    ax.set_xlabel("epoch"); ax.set_ylabel(r"energy $\langle H_{\mathrm{H_2}}\rangle$")
    ax.set_title("FD's ε-dilemma (sharp landscape + low shots, faithful noise):\n"
                 "PSR converges; every FD ε stalls (small ε → variance, "
                 "large ε → bias)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.abspath(os.path.join(out_dir, "fd_epsilon_dilemma.png"))
    fig.savefig(out)
    print(f"\nsaved: {out}")
    print("If PSR's curve drops well below every FD ε curve, no ε rescues FD in "
          "this regime.")


if __name__ == "__main__":
    main()
