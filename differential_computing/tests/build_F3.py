"""
build_F3.py — compose Figure 3 (the strategy benchmark) from the two cached
result sets, plus the case-study table. Reads JSON only, no simulation.

  F3(a) kick vs Nyquist — shot-cost ratio vs system size m for three tangent
        structures: the two are the SAME L1 functional (ratio=1 aligned), and
        Nyquist wins only when diam(A) is subextensive relative to ‖v‖₁.
        (case_study_kick_vs_nyquist.json)
  F3(b) FD vs kick vs Nyquist under noise (T/T2*=0.5, control δ, finite shots):
        oracle-FD floors at δ/ε, kick & Nyquist ride N^{-1/2} to ∇C_noisy.
        (noisy_nyquist_vs_fd_kick.json)
  F3(c) the case-study table.

Run: conda run -n qec_pg python differential_computing/tests/build_F3.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
C_FD, C_KICK, C_NYQ = "#D55E00", "#009E73", "#0072B2"
FAM = {"uniform": ("uniform  ΣZ_j", "#1a1a1a", "o"),
       "rotated": ("rotated  Σ(cφX+sφZ)", "#E69F00", "s"),
       "telescope": ("telescope  Σ(Z_j−Z_{j+1})", "#0072B2", "^")}


def load(name):
    return json.load(open(os.path.join(FIGDIR, name)))


def panel_a(ax, cs):
    for fam, (lab, c, mk) in FAM.items():
        rows = cs["data"][fam]
        m = [r["m"] for r in rows]
        ratio = [r["cost_kick"] / r["cost_nyq"] for r in rows]   # kick / Nyquist shots
        ax.loglog(m, ratio, mk + "-", color=c, ms=5, label=lab)
    ax.axhline(1.0, color="#999", lw=0.8, ls="--")
    ax.text(m[0], 1.15, "tie", fontsize=7, color="#666")
    ax.set_xlabel("qubits $m$"); ax.set_ylabel("shot cost  kick / Nyquist")
    ax.set_title("(a) same L1 functional — verdict $=$ diam$(A)$ vs $\\|v\\|_1$", fontsize=8.5)
    ax.legend(fontsize=6.8, loc="upper left"); ax.grid(True, which="both", alpha=0.15)


def panel_b(ax, nz):
    N = np.array(nz["budgets"])
    ax.loglog(N, nz["fd_best"], "s--", color=C_FD, ms=5, label="oracle-FD")
    ax.loglog(N, nz["kick"], "o-", color=C_KICK, ms=5, label="kick-PSR")
    ax.loglog(N, nz["nyq_none"], "^-", color=C_NYQ, ms=5, label="Nyquist")
    ax.loglog(N, np.array(nz["kick"])[0] * (N / N[0]) ** -0.5, ":", color="#999",
              lw=1, label="$N^{-1/2}$")
    ax.axhline(min(nz["fd_best"]), color=C_FD, lw=0.8, ls="-.")
    ax.text(N[-1] * 0.35, min(nz["fd_best"]) * 1.2, r"FD $\delta/\varepsilon$ floor",
            fontsize=7, color="#a0451a", ha="right")
    ax.set_xlabel("total shots $N$"); ax.set_ylabel(r"RMSE vs $\nabla C_{\rm noisy}$")
    ax.set_title(f"(b) noisy finite-shot  ($T/T_2^*$={nz['T_over_T2']:.2f}, $r$={nz['r']})",
                 fontsize=8.5)
    ax.legend(fontsize=6.8); ax.grid(True, which="both", alpha=0.15)


def panel_table(ax):
    ax.axis("off")
    cols = ["tangent $A$", "diam$(A)$", "$\\|v\\|_1$", "kick/Nyq", "verdict"]
    rows = [
        ["uniform  $\\Sigma Z_j$", "$2m$", "$m$", "$1$", "tie (kick O(1) edge)"],
        ["rotated  $\\Sigma(c_\\varphi X_j{+}s_\\varphi Z_j)$", "$2m$", "$\\sim\\!1.3m$", "$2$", "Nyquist O(1), non-foldable"],
        ["telescope  $\\Sigma(Z_j{-}Z_{j+1})$", "$4$", "$2m$", "$m^2$", "Nyquist $\\propto m^2$ (unfolded)"],
    ]
    t = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center",
                 colWidths=[0.30, 0.11, 0.11, 0.12, 0.36])
    t.auto_set_font_size(False); t.set_fontsize(7.2); t.scale(1, 1.5)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#f0f0f0"); cell.set_text_props(weight="bold")
    ax.set_title("(c) case study — kick vs Nyquist shots by tangent structure "
                 "(no $\\pi^2$ advantage)", fontsize=8.5, pad=2)


def main():
    cs = load("case_study_kick_vs_nyquist.json")
    nz = load("noisy_nyquist_vs_fd_kick.json")
    fig = plt.figure(figsize=(9.2, 6.4))
    gs = GridSpec(2, 2, height_ratios=[1.25, 0.8], hspace=0.42, wspace=0.28)
    panel_a(fig.add_subplot(gs[0, 0]), cs)
    panel_b(fig.add_subplot(gs[0, 1]), nz)
    panel_table(fig.add_subplot(gs[1, :]))
    fig.suptitle("F3 — Differentiation-strategy benchmark", fontsize=10, y=0.98)
    out = os.path.join(FIGDIR, "F3_strategy_benchmark.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
