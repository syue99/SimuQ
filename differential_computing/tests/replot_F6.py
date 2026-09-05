"""
replot_F6.py — re-render the single-column F6 (main RMSE-vs-N + FD-V inset)
from the cached figures/F6_floor_amplification.json. NO simulation.

Layout mirrors the figS block in build_F6.py; keep the two in sync when
changing style. Bands for PSR+gate and FD-fixed are not in the cache and
are omitted (faint series, alpha-0.10 bands).

Run: conda run -n qec_pg python differential_computing/tests/replot_F6.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
OUT3 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "paper_fig_3", "figs"))
OUT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "paper_fig_2"))
C_FD, C_PSR, C_NSR = "#D55E00", "#0072B2", "#009E73"


def main():
    d = json.load(open(os.path.join(FIGDIR, "F6_floor_amplification.json")))
    N = np.array(d["N"])
    series = [
        (d["psr"], d["psr_band"], C_PSR, "-", "o", 1.0,
         rf"PSR ($N^{{{d['exp_psr']:.2f}}}$)"),
        (d["nsr"], d["nsr_band"], C_NSR, "-", "s", 1.0,
         rf"NSR $M{{=}}\infty$ ($N^{{{d['exp_nsr']:.2f}}}$)"),
        (d["nsr_trunc"], d["nsr_trunc_band"], C_NSR, "-.", "v", 0.85,
         rf"NSR $M{{=}}{d['M_cap']}$ (headroom cap)"),
        (d["psr_gate"], None, C_PSR, "--", "s", 0.5,
         r"PSR + gate ($\varepsilon_{\rm ins}$)"),
        (d["fd"], d["fd_band"], C_FD, "-", "o", 1.0,
         rf"FD $\varepsilon^*$={d['eps_star']:.2f}"),
        (d["fd_fixed"], None, C_FD, ":", "^", 0.75,
         rf"FD $\varepsilon$={d['fd_fixed_eps']:g} fixed"),
    ]

    plt.rcParams.update({"font.size": 7})
    figS, axS = plt.subplots(figsize=(3.0, 3.1), dpi=300)
    for mm, band, c, st, mk, al, lab in series:
        axS.loglog(N, mm, st, marker=mk, color=c, ms=3.2, lw=1.2, alpha=al,
                   label=lab, mec="white", mew=0.25)
        if band is not None:
            axS.fill_between(N, band[0], band[1], color=c, alpha=0.10)
    axS.loglog(N, d["psr"][0] * (N / N[0]) ** -0.5, ":", color="#999", lw=1.0,
               label=r"$N^{-1/2}$")
    axS.set_xlabel(r"total executions $N$ (one gradient)", fontsize=7.5)
    axS.set_ylabel(r"RMSE vs $\nabla C_{\rm device}$", fontsize=7.5)
    axS.tick_params(labelsize=7)
    axS.grid(True, which="both", alpha=0.12)
    axS.legend(fontsize=5.2, loc="upper right", framealpha=0.85, handlelength=1.3,
               borderpad=0.25, labelspacing=0.22, handletextpad=0.4)
    axS.text(0.02, 0.98, r"$T/T_2^*=0.15$", transform=axS.transAxes, fontsize=7,
             color="#52514e", va="top")

    epsR = np.array(d["epsR"])
    fd_r = np.array(d["fd_r"])
    wr = np.array(d["fd_wrong"]) >= 0.2
    itgt = d["N"].index(10000)
    axV = axS.inset_axes([0.07, 0.085, 0.48, 0.30])
    # B.6.4's analytic curve is NOT drawn: as written it is truncation + delta/eps
    # only, and at this operating point it misses FD's common-mode displacement
    # term, so it sits a factor ~2 under the sweep near the minimum.  The
    # discrepancy is reported in NUMBERS.md instead of drawn as if it agreed.
    axV.loglog(epsR, fd_r, "-", color=C_FD, lw=1.2)
    axV.loglog(epsR[~wr], fd_r[~wr], "o", color=C_FD, ms=2.2)
    axV.loglog(epsR[wr], fd_r[wr], "X", color="#1a1a1a", ms=4.5)
    axV.axhline(d["psr"][itgt], color=C_PSR, lw=1.2)
    axV.axhline(d["nsr"][itgt], color=C_NSR, lw=1.2, ls="--")
    axV.set_xlabel(r"FD step $\varepsilon$  ($N$=$10^4$)", fontsize=7, labelpad=1,
                   bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    axV.tick_params(labelsize=7, pad=1)
    axV.tick_params(which="minor", left=False, bottom=False)
    for sp in axV.spines.values():
        sp.set_linewidth(0.7)

    figS.tight_layout(pad=0.4)
    for out in (OUT3, OUT2):
        os.makedirs(out, exist_ok=True)
        figS.savefig(os.path.join(out, "F6.pdf"), bbox_inches="tight",
                     pad_inches=0.02)
        figS.savefig(os.path.join(out, "F6.png"), bbox_inches="tight",
                     pad_inches=0.02)
    plt.close(figS)
    print("replotted F6 from cache -> paper_fig_3/figs + paper_fig_2")


if __name__ == "__main__":
    main()
