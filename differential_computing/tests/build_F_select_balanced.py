"""
build_F_select_balanced.py — F-select on a BALANCED plane (SEC6 handover C,
\\owed{balanced-plane run}, 2026-08-25).

Ruling option 1: same 7q TFIM device family {X_a, Z_a, Z_aZ_b} (35 terms), extend
k upward to the full alphabet (k = 1..35) and P downward (P = 1..10) so the
measured PSR/NSR regions are closer to half-half. Nothing else tuned.

Figure (paper_fig_3/figs/F_select.pdf): shading = measured winner (blue PSR,
green NSR), solid = measured crossing, dashed = the compiler's certificate
choice (Sec 5.3 policy: PSR unless PSR inadmissible or certified headroom admits
NSR at strictly lower C) — drawn only if the certified crossing exists on the
plane. Star = TFIM instance (p=2, q=1); open circle = its global-coefficient
rewrite (p=1, q=2).  Axis names follow the appendix: p = # differentiated
coefficients, q = # alphabet terms each touches (the code still calls the
sweep variables PS/KS). Hamiltonian level, no noise (stated on figure).

Numbers (C1-C4) printed and dumped to F_select_balanced_data.json.

Run: conda run -n qec_pg python differential_computing/tests/build_F_select_balanced.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import build_F_select as bs

FIGDIR = bs.FIGDIR
OUT3 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "paper_fig_3", "figs"))
OUT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "paper_fig_2"))
CACHE = os.path.join(FIGDIR, "F_select_balanced_data.json")

# balanced plane (ruling option 1): P down, k up to the full 35-term alphabet
bs.PS = list(range(1, 11))
bs.KS = list(range(1, 36))
bs.CACHE = CACHE

C_PSR, C_NSR = "#0072B2", "#009E73"
INK, SEC, GRID, SURFACE = bs.INK, bs.SEC, bs.GRID, bs.SURFACE
REG_CACHE = os.path.join(FIGDIR, "F_regimes_data.json")   # Fig 14's arrays (== Fig 10's) + AC certificate
GAMMA0 = 1.86                                              # eq:margin, App G.3.1
C_SEL = "#e07b00"                                          # selector line: highlight orange (both planes)
LW_MEAS, LW_SEL = 1.0, 1.4

# P0-1 (2026-09-05): the plane is coloured by the TARGET-FREE quantity App G.1 defines,
# mean over seeds of log10(N_NSR / N_PSR): hue = sign (green NSR wins, blue PSR wins),
# saturation = |log ratio|.  Shared with Fig 14 so panel (a) there is this figure.
RATIO_LIM = 0.8                                            # data range on the plane: -0.76 .. +0.28
RATIO_LEVELS = np.round(np.arange(-RATIO_LIM, RATIO_LIM + 1e-9, 0.1), 2)


def ratio_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("nsr_psr", [C_NSR, "#ffffff", C_PSR])


def selector_field(family="general"):
    """log10 N-ratio the compiler's selector believes: Omega_AC certificate + margin
    gamma(q) = min(1, GAMMA0/sqrt(q)); chooses NSR where it is < 0.  From Fig 14's cache."""
    if not os.path.exists(REG_CACHE):
        return None
    reg = json.load(open(REG_CACHE))
    ks = np.array(reg["meta"]["KS"], float)
    marg = np.log10(np.minimum(1.0, GAMMA0 / np.sqrt(ks)))[:, None]
    return np.array(reg[family]["Zpred_AC"]) + marg


def render(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    from matplotlib.colors import LinearSegmentedColormap

    g = data["general"]
    Ps, ks = data["meta"]["Ps"], data["meta"]["ks"]
    Pg, Kg = np.meshgrid(Ps, ks)
    Z = gaussian_filter(np.array(g["Z"]), sigma=0.8)
    sel = selector_field("general")
    Zsel = gaussian_filter(sel, sigma=0.8) if sel is not None else None

    plt.rcParams.update({"font.size": 7})
    fig, ax = plt.subplots(figsize=(3.4, 3.0), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.tick_params(labelsize=7, colors=SEC)
    for s in ax.spines.values():
        s.set_color(GRID)

    # colour = log10(N_NSR/N_PSR), diverging about 0 (App G.1's cell value)
    pc = ax.contourf(Pg, Kg, np.clip(Z, -RATIO_LIM + 1e-6, RATIO_LIM - 1e-6),
                     levels=RATIO_LEVELS, cmap=ratio_cmap(), antialiased=True)
    cb = fig.colorbar(pc, ax=ax, fraction=0.045, pad=0.02,
                      ticks=[-0.8, -0.4, 0.0, 0.4, 0.8])
    cb.set_label(r"$\log_{10}\,(N_{\rm NSR}\,/\,N_{\rm PSR})$", fontsize=7, color=SEC)
    cb.ax.tick_params(labelsize=7, colors=SEC)
    # solid = measured crossing (ratio 1); dashed = the compiler's selector (G.3.1),
    # same styles as Fig 14
    ax.contour(Pg, Kg, Z, levels=[0.0], colors="k", linewidths=LW_MEAS, zorder=4)
    cert_crosses = bool(Zsel is not None and Zsel.min() < 0 < Zsel.max())
    if cert_crosses:
        ax.contour(Pg, Kg, Zsel, levels=[0.0], colors=C_SEL, linewidths=LW_SEL,
                   linestyles="dashed", zorder=5)

    halo = [pe.withStroke(linewidth=2.0, foreground="#ffffff")]
    ax.text(0.10, 0.90, "NSR\nwins", transform=ax.transAxes, fontsize=8.5,
            weight="bold", color="#00654a", ha="center", va="center",
            path_effects=halo, zorder=5)
    ax.text(0.85, 0.12, "PSR\nwins", transform=ax.transAxes, fontsize=8.5,
            weight="bold", color="#00517e", ha="center", va="center",
            path_effects=halo, zorder=5)

    ax.plot([2], [1], marker="*", ms=11, color="#eb6834",
            markeredgecolor=INK, markeredgewidth=0.6, zorder=6, clip_on=False)
    ax.annotate("TFIM instance", xy=(2, 1), xytext=(10, -3),
                textcoords="offset points", fontsize=7, color="#a0451a",
                path_effects=halo)
    # the global-coefficient rewrite θ·(Z0Z1 + X0 + X1) has p = 1, q = 3 (was drawn at (1, 2))
    ax.plot([1], [3], marker="o", ms=5, markerfacecolor="none",
            markeredgecolor="#a0451a", markeredgewidth=1.0, zorder=6,
            clip_on=False)
    ax.annotate("global-θ rewrite", xy=(1, 3), xytext=(5, 9),
                textcoords="offset points", fontsize=7, color="#a0451a",
                path_effects=halo)

    ax.set_xticks(range(1, 11))
    ax.set_yticks([1, 5, 10, 15, 20, 25, 30, 35])
    ax.set_xlabel("# differentiated coefficients  $p$", fontsize=7.5, color=SEC)
    ax.set_ylabel("# alphabet terms per coefficient  $q$", fontsize=7.5, color=SEC)
    ax.text(0.985, 0.985, "Hamiltonian level, no noise",
            transform=ax.transAxes, fontsize=7, color=SEC, ha="right",
            va="top", path_effects=halo)
    fig.tight_layout(pad=0.4)
    os.makedirs(OUT3, exist_ok=True)
    os.makedirs(OUT2, exist_ok=True)
    for out in (OUT3, OUT2):
        fig.savefig(os.path.join(out, "F_select.pdf"), bbox_inches="tight",
                    pad_inches=0.02)
        fig.savefig(os.path.join(out, "F_select.png"), bbox_inches="tight",
                    pad_inches=0.02)
    plt.close(fig)
    return cert_crosses


def main():
    if not os.path.exists(CACHE):
        bs.run_sweep()
    data = json.load(open(CACHE))
    cert_crosses = render(data)

    g = data["general"]
    Z = np.array(g["Z"])
    Zp = np.array(g["Zpred"])
    # C2: NSR share of the sampled plane (measured)
    nsr_share = float((Z < 0).mean())
    # C3: forfeiture of the compiler's certificate choice vs measured optimum.
    # choice = NSR iff certificate says NSR at STRICTLY lower C, else PSR.
    # Z/Zpred = log10(N_NSR/N_PSR), so NSR-at-lower-C means Zpred < 0.
    choice_nsr = Zp < 0
    # forfeit factor per cell: 1 where choice matches measured winner, else 10^|Z|
    mism = (choice_nsr != (Z < 0))
    forf = np.where(mism, 10.0 ** np.abs(Z), 1.0)
    print(f"C1: family=TFIM device alphabet (7q; 7 X + 7 Z + 21 ZZ = 35 terms), "
          f"p∈[1,10], q∈[1,35], seeds={data['meta']['seeds']}")
    print(f"C2: NSR share of plane = {nsr_share*100:.1f}%")
    print(f"C3: compiler-choice forfeiture max = {forf.max():.2f}x  "
          f"median over plane = {np.median(forf):.2f}x  "
          f"median over divergent cells = "
          f"{np.median(forf[mism]) if mism.any() else 1.0:.2f}x  "
          f"(divergent on {mism.mean()*100:.1f}% of cells)")
    print(f"C3 aux: certificate crossing drawn = {cert_crosses}")
    print(f"C4: TFIM star at (p=2, q=1) [PSR side, measured "
          f"ratio 10^{Z[0,1]:+.2f}]; global-θ rewrite at (p=1, q=3) "
          f"[plane cell 10^{Z[2,0]:+.2f}; the instance itself measures 10^-0.24, NSR side]")
    sel = selector_field("general")
    if sel is not None:                      # the paper's selector numbers (G.3), AC + margin
        ch = sel < -1e-9; mm = ch != (Z < 0); ff = 10.0 ** np.abs(Z[mm])
        print(f"C5: selector (Omega_AC + margin, gamma0={GAMMA0}): agreement {100*(1-mm.mean()):.1f}%, "
              f"forfeit median {np.median(ff) if mm.any() else 1:.2f}x max {ff.max() if mm.any() else 1:.2f}x, "
              f"ties before margin (L1 certificate) {int((np.abs(Zp) < 1e-9).sum())}, after {int((np.abs(sel) <= 1e-9).sum())}")
    summary = dict(nsr_share=nsr_share, forfeit_max=float(forf.max()),
                   forfeit_median=float(np.median(forf)),
                   forfeit_median_divergent=float(np.median(forf[mism])) if mism.any() else 1.0,
                   divergent_frac=float(mism.mean()),
                   cert_crossing_drawn=cert_crosses,
                   star_ratio_log10=float(Z[0, 1]), circle_ratio_log10=float(Z[2, 0]),
                   circle_pq=(1, 3))
    d = json.load(open(CACHE))
    d["balanced_summary"] = summary
    json.dump(d, open(CACHE, "w"), indent=1)
    print("wrote paper_fig_3/figs/F_select.pdf/.png + balanced summary in cache")


if __name__ == "__main__":
    main()
