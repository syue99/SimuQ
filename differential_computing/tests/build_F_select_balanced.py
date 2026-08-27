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
plane. Star = TFIM instance (P=2, k=1); open circle = its global-coefficient
rewrite (P=1, k=2). Hamiltonian level, no noise (stated on figure).

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
    Zp = gaussian_filter(np.array(g["Zpred"]), sigma=0.8)
    F = gaussian_filter(np.array(g["logminN"]), sigma=0.5)

    plt.rcParams.update({"font.size": 7})
    fig, ax = plt.subplots(figsize=(3.4, 3.0), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.tick_params(labelsize=7, colors=SEC)
    for s in ax.spines.values():
        s.set_color(GRID)

    # original F_select format: neutral gray cost fill in discrete
    # half-decade bands (readable off the colorbar), hue reserved for the
    # winner washes
    grays = LinearSegmentedColormap.from_list(
        "costgray", ["#f4f3f0", "#43423f"])
    lo = np.floor(F.min() * 2) / 2
    hi = np.ceil(F.max() * 2) / 2
    levels = np.arange(lo, hi + 0.25, 0.5)
    pc = ax.contourf(Pg, Kg, F, levels=levels, cmap=grays)
    cb = fig.colorbar(pc, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("executions to target (best strategy)", fontsize=7,
                 color=SEC)
    cb.ax.tick_params(labelsize=7, colors=SEC)
    ticks = np.arange(np.ceil(lo), np.floor(hi) + 1)
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"$10^{{{int(t)}}}$" for t in ticks])

    # shading = measured winner (transparent washes over the gray fill)
    ax.contourf(Pg, Kg, Z, levels=[-99.0, 0.0, 99.0], colors=[C_NSR, C_PSR],
                alpha=0.30, antialiased=True, zorder=3)
    ax.contour(Pg, Kg, Z, levels=[0.0], colors="k", linewidths=1.5, zorder=4)
    # compiler's certificate choice (dashed) — only if it crosses on this plane
    cert_crosses = bool((Zp > 0).any() and (Zp < 0).any())
    if cert_crosses:
        ax.contour(Pg, Kg, Zp, levels=[0.0], colors=INK, linewidths=1.2,
                   linestyles="dashed", zorder=4)

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
    ax.plot([1], [2], marker="o", ms=5, markerfacecolor="none",
            markeredgecolor="#a0451a", markeredgewidth=1.0, zorder=6,
            clip_on=False)
    ax.annotate("global-θ rewrite", xy=(1, 2), xytext=(5, 9),
                textcoords="offset points", fontsize=7, color="#a0451a",
                path_effects=halo)

    ax.set_xticks(range(1, 11))
    ax.set_yticks([1, 5, 10, 15, 20, 25, 30, 35])
    ax.set_xlabel("# differentiated coefficients  P", fontsize=7.5, color=SEC)
    ax.set_ylabel("# alphabet terms per coefficient  k", fontsize=7.5, color=SEC)
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
          f"P∈[1,10], k∈[1,35], seeds={data['meta']['seeds']}")
    print(f"C2: NSR share of plane = {nsr_share*100:.1f}%")
    print(f"C3: compiler-choice forfeiture max = {forf.max():.2f}x  "
          f"median over plane = {np.median(forf):.2f}x  "
          f"median over divergent cells = "
          f"{np.median(forf[mism]) if mism.any() else 1.0:.2f}x  "
          f"(divergent on {mism.mean()*100:.1f}% of cells)")
    print(f"C3 aux: certificate crossing drawn = {cert_crosses}")
    print(f"C4: TFIM star at (P=2, k=1) [PSR side, measured "
          f"ratio 10^{Z[0,1]:+.2f}]; global-θ rewrite at (P=1, k=2) "
          f"[measured ratio 10^{Z[1,0]:+.2f}]")
    summary = dict(nsr_share=nsr_share, forfeit_max=float(forf.max()),
                   forfeit_median=float(np.median(forf)),
                   forfeit_median_divergent=float(np.median(forf[mism])) if mism.any() else 1.0,
                   divergent_frac=float(mism.mean()),
                   cert_crossing_drawn=cert_crosses,
                   star_ratio_log10=float(Z[0, 1]), circle_ratio_log10=float(Z[1, 0]))
    d = json.load(open(CACHE))
    d["balanced_summary"] = summary
    json.dump(d, open(CACHE, "w"), indent=1)
    print("wrote paper_fig_3/figs/F_select.pdf/.png + balanced summary in cache")


if __name__ == "__main__":
    main()
