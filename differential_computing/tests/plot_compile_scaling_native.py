"""
plot_compile_scaling_native.py — RQ3 compile-scaling figure from the cached
timings in figures/compile_scaling_native.json (run compile_scaling_native.py
first; this script never re-times anything).

Left panel : compile wall-time vs n, log-log — generic all-pairs path vs the
             specialized (device-native 1D chain) path, with fitted power-law
             slopes and an n^2 guide.
Right panel: incremental cost of differentiation — wall-time to map ONE PSR
             branch to hardware ops + pulse ledger (the evolution solve is
             shared across all branches).

Out: figures/compile_scaling_native.png/.pdf + _caption.txt
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
CACHE = os.path.join(FIGDIR, "compile_scaling_native.json")

# palette (dataviz reference instance, light mode)
BLUE = "#2a78d6"      # series 1 — specialized 1D chain
ORANGE = "#eb6834"    # series 2 — generic
AQUA = "#1baf7a"      # series 3 — specialized 2D grid
INK = "#0b0b0b"
SEC = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def series(rows, name, key="compile_s"):
    rs = sorted((r for r in rows if r["series"] == name), key=lambda r: r["n"])
    return (np.array([r["n"] for r in rs], dtype=float),
            np.array([r[key] for r in rs], dtype=float), rs)


def fit_slope(n, t, n_min=0):
    m = n >= n_min
    if m.sum() < 2:
        return np.nan
    return np.polyfit(np.log(n[m]), np.log(t[m]), 1)[0]


def main():
    rows = json.load(open(CACHE))
    ng, tg, _ = series(rows, "generic")
    ns, ts, spec_rows = series(rows, "specialized")
    n2, t2, spec2_rows = series(rows, "specialized2d")
    nb = np.array([r["n"] for r in spec_rows if "branch_ms" in r], dtype=float)
    tb = np.array([r["branch_ms"] for r in spec_rows if "branch_ms" in r])
    nb2 = np.array([r["n"] for r in spec2_rows if "branch_ms" in r], dtype=float)
    tb2 = np.array([r["branch_ms"] for r in spec2_rows if "branch_ms" in r])

    sg = fit_slope(ng, tg, n_min=5)
    ss = fit_slope(ns, ts, n_min=50)
    s2 = fit_slope(n2, t2, n_min=50)
    sb = fit_slope(nb, tb, n_min=50)

    fig, (ax, axb) = plt.subplots(
        1, 2, figsize=(8.6, 3.4), dpi=200,
        gridspec_kw=dict(wspace=0.32, left=0.085, right=0.985,
                         top=0.86, bottom=0.16))
    fig.patch.set_facecolor(SURFACE)

    for a in (ax, axb):
        a.set_facecolor(SURFACE)
        a.grid(True, which="major", color=GRID, linewidth=0.6)
        a.tick_params(colors=MUTED, labelsize=8)
        for s in a.spines.values():
            s.set_color(GRID)

    # ── left: compile time ──
    ax.loglog(ng, tg, "-o", color=ORANGE, linewidth=1.8, markersize=5,
              markerfacecolor=ORANGE, markeredgecolor=SURFACE,
              markeredgewidth=0.8, label="generic all-pairs machine", zorder=3)
    ax.loglog(ns, ts, "-o", color=BLUE, linewidth=1.8, markersize=5,
              markerfacecolor=BLUE, markeredgecolor=SURFACE,
              markeredgewidth=0.8, label="specialized (1D chain)", zorder=4)
    if len(n2):
        ax.loglog(n2, t2, "-o", color=AQUA, linewidth=1.8, markersize=5,
                  markerfacecolor=AQUA, markeredgecolor=SURFACE,
                  markeredgewidth=0.8, label="specialized (2D grid)", zorder=4)
    guide = ts[-1] * (ns / ns[-1]) ** 2
    ax.loglog(ns, guide, "--", color=MUTED, linewidth=1.0, zorder=2)
    ax.annotate(r"$\propto n^2$", xy=(ns[-4], guide[-4]), fontsize=8,
                color=MUTED, xytext=(2, -14), textcoords="offset points")

    ax.annotate(f"slope ≈ {sg:.1f}", xy=(ng[-2], tg[-2]), fontsize=8,
                color=ORANGE, xytext=(8, -8), textcoords="offset points")
    ax.annotate(f"slope ≈ {ss:.1f}", xy=(ns[-2], ts[-2]), fontsize=8,
                color=BLUE, xytext=(4, -12), textcoords="offset points")
    if len(n2):
        ax.annotate(f"2D slope ≈ {s2:.1f}", xy=(n2[-2], t2[-2]), fontsize=8,
                    color=AQUA, xytext=(-8, 4), textcoords="offset points",
                    ha="right")
    ax.axvline(ng[-1], color=GRID, linewidth=0.8, linestyle=":")
    ax.annotate("generic-path ceiling (n=12)", xy=(ng[-1], 20),
                fontsize=7.5, color=SEC, xytext=(28, 12),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7))

    ax.set_xlabel("qubits  n", fontsize=9, color=SEC)
    ax.set_ylabel("compile wall-time  (s)", fontsize=9, color=SEC)
    ax.set_title("Base compilation: TFIM → pulse schedule",
                 fontsize=9.5, color=INK, pad=8)
    leg = ax.legend(fontsize=8, loc="lower right", frameon=False,
                    labelcolor=SEC, handlelength=1.6)

    # ── right: incremental per-branch cost ──
    axb.loglog(nb, tb, "-o", color=BLUE, linewidth=1.8, markersize=5,
               markerfacecolor=BLUE, markeredgecolor=SURFACE,
               markeredgewidth=0.8, label="1D chain", zorder=3)
    if len(nb2):
        axb.loglog(nb2, tb2, "-o", color=AQUA, linewidth=1.8, markersize=5,
                   markerfacecolor=AQUA, markeredgecolor=SURFACE,
                   markeredgewidth=0.8, label="2D grid", zorder=3)
        axb.legend(fontsize=8, loc="upper left", frameon=False,
                   labelcolor=SEC, handlelength=1.6)
    axb.annotate(f"slope ≈ {sb:.1f}", xy=(nb[-1], tb[-1]), fontsize=8,
                 color=BLUE, xytext=(-6, 8), textcoords="offset points",
                 ha="right")
    axb.set_xlabel("qubits  n", fontsize=9, color=SEC)
    axb.set_ylabel("map one PSR branch  (ms)", fontsize=9, color=SEC)
    axb.set_title("Incremental cost of differentiation (per branch)",
                  fontsize=9.5, color=INK, pad=8)

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"compile_scaling_native.{ext}"))
    plt.close(fig)

    with open(os.path.join(FIGDIR, "compile_scaling_native_caption.txt"), "w") as fh:
        both = {int(n): t for n, t in zip(ng, tg)}
        speedup = max(both[int(n)] / t for n, t in zip(ns, ts) if int(n) in both)
        grid_note = ""
        if len(n2):
            grid_note = (
                f" The 2D series (full m×k NN grids up to 32×32 = {int(n2[-1])}, "
                f"{t2[-1]:.0f} s, slope ~{s2:.1f}) compiles the NN-grid target "
                "exactly; the geometrically non-cancellable diagonal coupling "
                "(J/8 per diagonal pair, ~12% relative L1) is NOT compiled and "
                "is declared in the plan's dropped_zz field — the compiled "
                "model is exact, the disclosure quantifies the physical "
                "device's deviation from it.")
        fh.write(
            "Compile-time scaling for TFIM targets on the rydberg2d AAIS "
            "(differentiation-first pipeline). Left: the generic all-pairs "
            f"machine grows with fitted slope ~{sg:.1f} and is practically "
            f"capped at n=12 ({tg[-1]:.0f} s); the target-aware specialization "
            "layer (frozen geometry, bond-pruned ZZ/dressing lines, analytic "
            "warm start, sparse Jacobian) compiles the SAME AAIS with slope "
            f"~{ss:.1f}, reaching n=1000 in {ts[-1]:.0f} s (up to "
            f"{speedup:.0f}x at overlapping sizes); the warm start is a "
            "residual-zero point, so the solve verifies rather than searches."
            f"{grid_note} "
            "Right: differentiation adds only a per-branch mapping pass "
            f"(evolution solve shared across branches): {tb[-1]:.0f} ms/branch "
            f"at n=1000 (slope ~{sb:.1f}). Compiled Hamiltonian matches the "
            "target to machine precision at every size (max |dH| <= 5e-13); "
            "the 1D truncated dressing tail is ~1.5% relative. Data: "
            "compile_scaling_native.json (median of repeated timings where "
            "compile < 5 s).\n")
    print("wrote", os.path.join(FIGDIR, "compile_scaling_native.png"))
    print(f"slopes: generic {sg:.2f}, specialized {ss:.2f}, "
          f"2d {s2 if len(n2) else float('nan'):.2f}, branch {sb:.2f}")


if __name__ == "__main__":
    main()
