"""
build_fig1.py — SEC6 P2-A: Fig 1 (intro trap image), single-column.

Top panel of fig3_fd_trap only, T/T2*=0.50: the noisy device landscape, FD secants
(wrong sign), and raw PSR = the exact noisy gradient. Minimal (no NSR). Reads the
cached landscape_device_data.json (Hamiltonian-level, T4 noise). PDF+PNG.
Run: conda run -n qec_pg python differential_computing/tests/build_fig1.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
COL = 3.3
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "font.family": "serif",
                     "mathtext.fontset": "stix", "legend.frameon": False,
                     "axes.linewidth": 0.7, "savefig.dpi": 300})
C_INK, C_FD, C_PSR = "#1a1a1a", "#D55E00", "#0072B2"


def main():
    d = json.load(open(os.path.join(FIGDIR, "landscape_device_data.json")))
    regime = d["T"] / d["T2"]; x0 = d["x_star"]; g = d["g_real"]; z0 = d["z0"]
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    gx = np.array(d["gx"])
    ax.plot(gx, d["y_noisy"], color=C_INK, lw=1.6, label="noisy device landscape")
    ex = np.array([x0 - 0.30, x0 + 0.30])
    ax.plot(ex, z0 + g * (ex - x0), color=C_PSR, lw=2.4,
            label=rf"raw PSR $=\nabla C_{{\rm noisy}}$ ({g:+.2f})")
    ramp = plt.cm.Oranges(np.linspace(0.5, 0.9, len(d["secants"])))
    for k, (sec, c) in enumerate(zip(d["secants"], ramp)):
        e = sec["eps"]
        ax.plot([x0 - e, x0 + e], [sec["fm"], sec["fp"]], "o-", color=c, lw=1.2, ms=2.6,
                label="FD secants (wrong sign)" if k == 0 else None)
    ax.plot([x0], [z0], "o", color=C_INK, ms=4)
    ax.text(0.97, 0.05, rf"$T/T_2^*={regime:.2f}$", transform=ax.transAxes, fontsize=7.5,
            color="#555", ha="right")
    ax.set_xlabel(r"parameter $\theta$"); ax.set_ylabel(r"$\langle O\rangle_{\rm noisy}(\theta)$")
    ax.legend(loc="lower center", fontsize=6.4, handlelength=1.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"fig1_intro_trap.{ext}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote fig1_intro_trap.pdf/.png  (T/T2*={regime:.2f}, PSR={g:+.3f})")


if __name__ == "__main__":
    main()
