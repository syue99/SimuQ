"""
compile_curves.py — App H (fig:compile-curves): compile time vs system size,
and what differentiating the program adds on top.

PLOT-ONLY.  Every number is read from a committed cache; nothing is re-timed,
because re-timing only adds jitter to numbers the appendix already quotes:

  figures/F_scale_data.json        generic path (n = 2..12), specialized 1D
                                   (n = 4..1000), specialized 2D (n = 16..1024)
  figures/sec6_compile_timing.json Table 5's rows: source compile and the PSR /
                                   NSR per-branch increments at
                                   n = 10, 30, 100, 300, 1000; plus D4 (an FD
                                   branch's full recompile at the shifted value)

Two panels:
  left   compile time vs n.  The generic path is drawn from its MEASURED points
         (it ceilings at n = 12, 27.1 s) with the fitted n^4.4 continued as a
         dashed extrapolation past the ceiling — the extrapolation is labelled
         as such, never as data.  Specialized 1D and 2D are measured throughout.
  right  the differentiation increment per branch against the source compile:
         PSR re-maps the schedule (a kick segment splits the evolution and
         inserts transport + CZ), NSR emits an O(n) coefficient table on the
         shared schedule.  FD is drawn TWICE, because D4 measured it twice and
         the honest answer needs both: a black-box FD branch that calls the
         compiler again at x+eps pays a full recompile (99.4% of the source
         compile, no reuse), while the SAME branch routed through the
         specializer's closed-form shift table costs 0.059 ms — indistinguishable
         from NSR's own 0.053 ms.  So FD's compile cost is not intrinsic: FD is
         free exactly when it reuses the differentiation infrastructure it is
         usually motivated by not needing.  What separates FD from the shift
         rules is statistical (Fig 8), not compile time.

The 2D series carries a disclosure: the compiled NN-grid model is exact
(max|dH| <= 1.5e-14 at 32x32), but the diagonal J/8 tail (~14% relative L1) is
NOT compiled — it is declared in the plan's dropped_zz field and is the physical
device's deviation from the compiled model.

Timing scope (inherited from build_F_scale): schedule ops + pulse ledger, NOT
pulse-shape synthesis (the PulseDSL emission layer has a 16-channel logical cap
and its COMB encoding does not complete at n = 100).  Timings are
machine-dependent; the machine string is read from the cache and stamped on the
figure.

Run:  conda run -n qec_pg python differential_computing/tests/compile_curves.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
OUT3 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "paper_fig_3", "figs"))
SCALE_JSON = os.path.join(FIGDIR, "F_scale_data.json")
TIMING_JSON = os.path.join(FIGDIR, "sec6_compile_timing.json")

INK, SEC, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
C_GEN, C_1D, C_2D = "#8a4b00", "#0b0b0b", "#6a51a3"
C_PSR, C_NSR, C_FD = "#0072B2", "#009E73", "#D55E00"
HALO = [pe.withStroke(linewidth=1.9, foreground="white")]
# The generic path's exponent depends on the fit window: over ALL measured
# points the slope is 3.8, because n <= 4 is overhead-dominated; over the
# asymptotic window n >= FIT_FROM it is 4.4, which is the number App H quotes.
# The figure fits the window and prints both, so nobody re-fits all eight points
# and gets 3.8 while the text says 4.4.
FIT_FROM = 5


def load():
    scale = json.load(open(SCALE_JSON))
    timing = json.load(open(TIMING_JSON))
    series = {}
    for r in scale["rows"]:
        series.setdefault(r["series"], []).append((r["n"], r["compile_s"]))
    for k in series:
        series[k] = np.array(sorted(series[k]), dtype=float)
    return scale, timing, series


def generic_slope(gen, fit_from=None):
    """Asymptotic exponent of the generic path (small n is overhead-dominated)."""
    m = gen[:, 0] >= (fit_from or FIT_FROM)
    return float(np.polyfit(np.log(gen[m, 0]), np.log(gen[m, 1]), 1)[0])


def render(scale, timing, series):
    rows = timing["D1_D3_rows"]
    ns = np.array(sorted(int(k) for k in rows), dtype=float)
    src = np.array([rows[str(int(n))]["compile_s"] for n in ns])
    psr_ms = np.array([rows[str(int(n))]["psr_branch_ms"] for n in ns])
    nsr_ms = np.array([rows[str(int(n))]["nsr_branch_ms"] for n in ns])

    plt.rcParams.update({"font.size": 7})
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.1, 2.9), dpi=300)
    fig.patch.set_facecolor(SURFACE)

    # ── left: compile time vs n ──
    gen = series["generic"]
    slope = generic_slope(gen)
    axL.loglog(gen[:, 0], gen[:, 1], "o-", color=C_GEN, lw=1.3, ms=3.4,
               label="generic path (measured)")
    anchor = gen[-1, 1] / gen[-1, 0] ** slope
    n_stop = float((1e5 / anchor) ** (1.0 / slope))     # ~1 day of compiling
    n_ext = np.geomspace(gen[-1, 0], n_stop, 40)
    axL.loglog(n_ext, anchor * n_ext ** slope, ls=(0, (3, 2)),
               color=C_GEN, lw=1.0, alpha=0.85,
               label=rf"$\propto n^{{{slope:.1f}}}$ extrapolation "
                     rf"(fit $n\geq{FIT_FROM}$)")
    axL.annotate(f"a day of compiling by $n\\approx${n_stop:.0f}",
                 xy=(n_stop, 1e5), xytext=(115, 1.1e4), fontsize=6.2,
                 color=C_GEN, path_effects=HALO,
                 arrowprops=dict(arrowstyle="->", color=C_GEN, lw=0.7))
    axL.annotate(f"ceiling: {gen[-1, 1]:.1f} s at $n$={int(gen[-1, 0])}",
                 xy=(gen[-1, 0], gen[-1, 1]), xytext=(2.2, 900),
                 fontsize=6.2, color=C_GEN, path_effects=HALO,
                 arrowprops=dict(arrowstyle="->", color=C_GEN, lw=0.7))

    sp = series["specialized"]
    axL.loglog(sp[:, 0], sp[:, 1], "s-", color=C_1D, lw=1.3, ms=3.0,
               label="specialized, 1D chain")
    tw = series["specialized2d"]
    axL.loglog(tw[:, 0], tw[:, 1], "^-", color=C_2D, lw=1.3,
               ms=3.2, label=r"specialized, 2D grid")

    axL.annotate(f"1D: {sp[-1, 1]:.0f} s at $n$={int(sp[-1, 0])}",
                 xy=(sp[-1, 0], sp[-1, 1]), xytext=(sp[-1, 0] * 0.055,
                                                    sp[-1, 1] * 0.13),
                 fontsize=6.2, color=SEC, path_effects=HALO)
    axL.annotate(f"2D: {tw[-1, 1]:.0f} s at $n$={int(tw[-1, 0])}",
                 xy=(tw[-1, 0], tw[-1, 1]), xytext=(tw[-1, 0] * 0.055,
                                                    tw[-1, 1] * 3.2),
                 fontsize=6.2, color=C_2D, path_effects=HALO)
    axL.set_xlabel("qubits  $n$", fontsize=7.4, color=INK)
    axL.set_ylabel("source compile (s)", fontsize=7.4, color=INK)
    axL.legend(fontsize=5.9, frameon=False, loc="upper left",
               handlelength=1.9, borderpad=0.2, labelspacing=0.28)
    axL.set_title("(a) compiling the source program", fontsize=7.4, color=INK)

    # ── right: what differentiation adds, per branch ──
    axR.loglog(ns, psr_ms, "o-", color=C_PSR, lw=1.4, ms=3.4,
               label="PSR branch (re-map: kick splits the evolution)")
    axR.loglog(ns, nsr_ms, "s--", color=C_NSR, lw=1.4, ms=3.2,
               label="NSR branch (O($n$) coefficient table)")
    axR.loglog(ns, src * 1e3, ":", color=SEC, lw=1.1,
               label="source compile, for scale")

    d4 = timing["D4"]
    axR.loglog([d4["n"]], [d4["fd_full_recompile_s"] * 1e3], "X", color=C_FD,
               ms=7, label=f"FD branch, black box: recompiles "
                           f"({d4['fd_pct_of_source']:.0f}% of source)")
    axR.loglog([d4["n"]], [d4["fd_table_reuse_ms"]], "o", mfc="none",
               mec=C_FD, mew=1.4, ms=7,
               label="FD branch, same shift table: free")
    axR.annotate("", xy=(d4["n"], d4["fd_table_reuse_ms"] * 1.9),
                 xytext=(d4["n"], d4["fd_full_recompile_s"] * 1e3 * 0.55),
                 arrowprops=dict(arrowstyle="->", color=C_FD, lw=0.9,
                                 ls=(0, (2, 1.6))))
    axR.text(d4["n"] * 1.25, 3.0,
             "FD is free — through the\nshift table it is\nmotivated by not needing",
             fontsize=6.0, color=C_FD, path_effects=HALO, va="center")
    axR.text(11, 3.4e-3,
             rf"at $n$=1000 a shift-table branch is "
             rf"{psr_ms[-1] / nsr_ms[-1]:.0f}$\times$ cheaper than a PSR branch",
             fontsize=6.2, color="#0f6b52", path_effects=HALO, va="bottom")
    axR.set_xlabel("qubits  $n$", fontsize=7.4, color=INK)
    axR.set_ylabel("per-branch increment (ms)", fontsize=7.4, color=INK)
    axR.legend(fontsize=5.9, frameon=False, loc="upper left",
               handlelength=1.9, borderpad=0.2, labelspacing=0.28)
    axR.set_title("(b) the price of differentiating it", fontsize=7.4,
                  color=INK)

    for ax in (axL, axR):
        ax.set_facecolor(SURFACE)
        ax.grid(True, which="both", alpha=0.13)
        ax.tick_params(labelsize=6.5, colors=SEC)
        for s in ax.spines.values():
            s.set_color(GRID)

    mach = scale["meta"].get("machine", "unrecorded machine")
    fig.text(0.5, -0.02,
             f"wall clock on {mach}; schedule ops + pulse ledger only (no "
             f"pulse-shape synthesis).  2D grid: compiled NN model exact, "
             f"diagonal $J/8$ tail (~14% L1) declared-dropped.",
             fontsize=5.8, color=SEC, ha="center")
    fig.tight_layout(pad=0.4)
    for out in (FIGDIR, OUT3):
        os.makedirs(out, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out, f"compile_curves.{ext}"),
                        bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"wrote compile_curves.pdf/.png -> {FIGDIR}, {OUT3}")


def report(scale, timing, series):
    gen, sp, tw = series["generic"], series["specialized"], series["specialized2d"]
    rows = timing["D1_D3_rows"]
    print(f"generic path      : n={int(gen[0,0])}..{int(gen[-1,0])}, "
          f"{gen[-1,1]:.1f} s at the n={int(gen[-1,0])} ceiling "
          f"({len(gen)} measured points)")
    print(f"specialized 1D    : n={int(sp[0,0])}..{int(sp[-1,0])}, "
          f"{sp[-1,1]:.1f} s at n={int(sp[-1,0])}")
    print(f"specialized 2D    : n={int(tw[0,0])}..{int(tw[-1,0])}, "
          f"{tw[-1,1]:.1f} s at n={int(tw[-1,0])}")
    for tag, arr, wins in (("generic", gen, (2, FIT_FROM)),
                           ("specialized 1D", sp, (4, 100)),
                           ("specialized 2D", tw, (16, 100))):
        parts = []
        for lo in wins:
            m = arr[:, 0] >= lo
            parts.append(f"n>={lo}: {np.polyfit(np.log(arr[m, 0]), np.log(arr[m, 1]), 1)[0]:.2f}")
        print(f"  fitted slope, {tag:15s} " + ",  ".join(parts))
    n1000 = rows["1000"]
    print(f"at n=1000: source {n1000['compile_s']:.1f} s, "
          f"PSR branch {n1000['psr_branch_ms']:.1f} ms, "
          f"NSR branch {n1000['nsr_branch_ms']:.3f} ms "
          f"({n1000['psr_branch_ms'] / n1000['nsr_branch_ms']:.0f}x)")
    d4 = timing["D4"]
    nsr_at_n = rows[str(d4["n"])]["nsr_branch_ms"]
    print(f"FD branch (D4, n={d4['n']}): black box = "
          f"{d4['fd_full_recompile_s']:.2f} s = {d4['fd_pct_of_source']:.1f}% of "
          f"the source compile; through the shift table = "
          f"{d4['fd_table_reuse_ms']:.3f} ms, vs NSR's own branch "
          f"{nsr_at_n:.3f} ms at the same n "
          f"({d4['fd_table_reuse_ms'] / nsr_at_n:.2f}x) "
          f"-> FD's compile cost is a reuse question, not an intrinsic one")


def main():
    scale, timing, series = load()
    render(scale, timing, series)
    report(scale, timing, series)


if __name__ == "__main__":
    main()
