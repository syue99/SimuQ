"""
build_F3.py — compose Figure 3 (the strategy benchmark) from cached results.
Reads JSON only, no simulation.

  F3(a) kick vs Nyquist REGIME map — same L1 functional; who wins is set by the
        structural alignment ρ=diam(A)/Σ|v_j| and the branch correlation f₊f₋:
        kick wins ⟺ ρ>2√(1−f₊f₋). Neither universally wins.
        (regime_kick_vs_nyquist.json + analytic boundary)
  F3(b) FD vs kick vs Nyquist under noise (T/T2*=0.5, δ, finite shots): oracle-FD
        floors at δ/ε, kick & Nyquist ride N^{-1/2} to ∇C_noisy.
        (noisy_nyquist_vs_fd_kick.json)
  F3(c) the regime characteristic.

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


def load(name):
    return json.load(open(os.path.join(FIGDIR, name)))


def panel_a(ax, rg):
    c = np.linspace(0.0, 1.0, 200); rho_b = 2 * np.sqrt(c)
    ax.fill_betweenx(np.linspace(0, 2, 200), 0, 2, color=C_NYQ, alpha=0.12)
    ax.fill_betweenx(c, rho_b, 2.0, color=C_KICK, alpha=0.25)
    ax.plot(rho_b, c, "k-", lw=1.5)
    ax.text(1.68, 0.12, "KICK wins\n(aligned +\ncorrelated)", color="#00695c",
            fontsize=8, ha="center", weight="bold")
    ax.text(0.6, 1.3, "Nyquist wins\n(non-foldable\nsubextensive,\nor $f_+f_-{<}0$)",
            color=C_NYQ, fontsize=8, ha="center", weight="bold")
    ax.axvline(2.0, color="#333", lw=1.0, ls=":")
    ax.text(1.98, 1.86, "foldable", fontsize=6.5, ha="right", color="#333")
    r = np.array(rg["ratio"])
    ax.scatter(np.full_like(r, 2.0) - 0.02, r,
               c=np.where(r < 1, C_KICK, C_FD), s=12, zorder=5)
    ax.plot([0.5], [0.6], "^", color=C_NYQ, ms=9, zorder=5)
    ax.set_xlabel(r"$\rho=\mathrm{diam}(A)/\Sigma_j|v_j|$")
    ax.set_ylabel(r"$1-f_+f_-$")
    ax.set_title(r"(a) kick vs Nyquist regime: wins iff $\rho>2\sqrt{1-f_+f_-}$"
                 "\n(dots: real $A{=}Z$ pts, $\\rho{=}2$; same L1 — no $\\pi^2$)", fontsize=8.3)
    ax.set_xlim(0, 2.08); ax.set_ylim(0, 2.0)


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
                 fontsize=8.3)
    ax.legend(fontsize=7); ax.grid(True, which="both", alpha=0.15)


def panel_table(ax):
    ax.axis("off")
    cols = ["regime", "condition", "winner", "why"]
    rows = [
        ["aligned/foldable, correlated", "$\\rho{=}2,\\ f_+f_->0$", "kick",
         "co-located $\\pm$ branches: 2nd moment $2{-}2f_+f_-$ small"],
        ["aligned/foldable, anti-corr.", "$\\rho{=}2,\\ f_+f_-<0$", "Nyquist",
         "kick's both-branches becomes a penalty"],
        ["non-foldable subextensive", "$\\rho<2$", "Nyquist",
         "diam$(A)<\\Sigma|v_j|$; single combined shift (exotic tangents)"],
    ]
    t = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center",
                 colWidths=[0.27, 0.16, 0.10, 0.47])
    t.auto_set_font_size(False); t.set_fontsize(7.4); t.scale(1, 1.5)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#f0f0f0"); cell.set_text_props(weight="bold")
    ax.set_title("(c) neither universally wins — same L1 functional; regime set by "
                 "structure $\\rho$ and landscape $f_+f_-$", fontsize=8.5, pad=2)


def main():
    rg = load("regime_kick_vs_nyquist.json")
    nz = load("noisy_nyquist_vs_fd_kick.json")
    fig = plt.figure(figsize=(10.2, 7.4))
    gs = GridSpec(2, 2, height_ratios=[1.2, 0.72], hspace=0.5, wspace=0.28)
    panel_a(fig.add_subplot(gs[0, 0]), rg)
    panel_b(fig.add_subplot(gs[0, 1]), nz)
    panel_table(fig.add_subplot(gs[1, :]))
    fig.suptitle("F3 — Differentiation-strategy benchmark", fontsize=11, y=0.98)
    out = os.path.join(FIGDIR, "F3_strategy_benchmark.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
