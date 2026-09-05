"""
build_F_lowering.py — Fig 6 (the lowering stack), redrawn per the plot-redo handover P0-7.

Replaces figures/compiler.png.  Rules applied: the instance strip is DROPPED (handover
default: Figs 7 and 13 carry the running instance); every label >= 7 pt; no internal names
(no builder / config / function names, no section codes).  Colour language: orange = this
paper, blue = reused from SimuQ, grey = artifact / input.  Two-column float (7.0 in).

Run: conda run -n qec_pg python differential_computing/tests/build_F_lowering.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
OUT3 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "paper_fig_3", "figs"))
OUT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "paper_fig_2"))

C_NEW, C_NEW_FILL = "#D55E00", "#fdf1e8"      # this paper
C_OLD, C_OLD_FILL = "#0072B2", "#e8f1f8"      # reused (SimuQ)
C_ART, C_ART_FILL = "#6b6b6b", "#eeeeee"      # artifact / input
INK, SEC = "#1a1a1a", "#52514e"


def box(ax, x, y, w, h, edge, fill, lw=1.2, ls="-", r=0.08, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fill, ec=edge, lw=lw, ls=ls, zorder=z))


def arrow(ax, x0, y0, x1, y1, z=4):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
                                 lw=1.1, color=INK, zorder=z, shrinkA=0, shrinkB=0))


def badge(ax, x, y, letter, color):
    ax.add_patch(plt.Circle((x, y), 0.13, fc="white", ec=color, lw=1.1, zorder=5))
    ax.text(x, y, letter, ha="center", va="center", fontsize=7.5, weight="bold", color=color, zorder=6)


def main():
    plt.rcParams.update({"font.size": 7, "font.family": "sans-serif"})
    fig, ax = plt.subplots(figsize=(7.0, 2.5), dpi=300)
    ax.set_xlim(0, 14.0); ax.set_ylim(0, 4.9); ax.axis("off")

    # ── legend row ──
    for x, edge, fill, lab in [(0.3, C_NEW, C_NEW_FILL, "this paper"),
                               (2.3, C_OLD, C_OLD_FILL, "reused (SimuQ solver; scheduler design)"),
                               (7.4, C_ART, C_ART_FILL, "artifact / input")]:
        box(ax, x, 4.4, 0.42, 0.28, edge, fill, r=0.05)
        ax.text(x + 0.55, 4.54, lab, va="center", fontsize=7, color=INK)

    # ── five stages ──
    Y, H = 1.55, 2.25
    stages = [
        (1.15, 2.1, "A", C_NEW, C_NEW_FILL, "Normal form",
         "parametrized\nprogram; the\ndifferentiated\ncoefficients marked", "this paper"),
        (3.6, 2.3, "B", C_NEW, C_NEW_FILL, "Runtime",
         "shift schedule,\nweights, shots;\nline-binned\nsegments", "this paper"),
        (6.25, 2.5, "C", C_OLD, C_OLD_FILL, "SimuQ solver", None, "reused"),
        (9.1, 2.3, "D", C_ART, C_ART_FILL, "Machine-native\nsegments", None, "artifact"),
        (11.75, 2.05, "E", C_NEW, C_NEW_FILL, "Scheduler +\npulse ledger",
         "multiplexing,\ntransport plans,\ninsertions", "this paper"),
    ]
    for x, w, L, edge, fill, title, body, tag in stages:
        box(ax, x, Y, w, H, edge, fill)
        badge(ax, x + 0.24, Y + H - 0.26, L, edge)
        ax.text(x + w / 2 + (0.2 if L == "D" else 0.14), Y + H - 0.30, title, ha="center", va="center",
                fontsize=7.5, weight="bold", color=INK, zorder=6, linespacing=1.05)
        if body:
            ax.text(x + w / 2, Y + 0.95, body, ha="center", va="center", fontsize=7, color=INK, zorder=6,
                    linespacing=1.15)
        ax.text(x + w / 2, Y + 0.17, tag, ha="center", va="center", fontsize=7, color=edge, zorder=6,
                style="italic")
    # C: the two sub-boxes this paper adds inside the reused solver
    cx, cw = 6.25, 2.5
    for yy, txt in [(Y + 1.08, "NSR: shifted\ncoefficients"), (Y + 0.40, "PSR: binned\nsegment + kick")]:
        box(ax, cx + 0.16, yy - 0.02, cw - 0.32, 0.56, C_NEW, "#ffffff", lw=1.0, r=0.05, z=5)
        ax.text(cx + cw / 2, yy + 0.26, txt, ha="center", va="center", fontsize=7, color=INK, zorder=6,
                linespacing=1.05)
    # D: the emulator below it (this paper)
    ex_, ew = 9.1, 2.3
    box(ax, ex_, 0.1, ew, 1.05, C_NEW, C_NEW_FILL)
    ax.text(ex_ + ew / 2, 0.86, "Emulator", ha="center", va="center", fontsize=7.5, weight="bold", color=INK, zorder=6)
    ax.text(ex_ + ew / 2, 0.45, "expectation values,\ndevice noise model", ha="center", va="center",
            fontsize=7, color=INK, zorder=6, linespacing=1.1)
    arrow(ax, ex_ + ew / 2, Y, ex_ + ew / 2, 1.17)

    # ── flow arrows and end labels ──
    ym = Y + H / 2
    ax.text(0.5, ym, "analog\nprogram", ha="center", va="center", fontsize=7, color=INK)
    arrow(ax, 0.9, ym, 1.15, ym)
    for x0, x1 in [(3.25, 3.6), (5.9, 6.25), (8.75, 9.1), (11.4, 11.75)]:
        arrow(ax, x0, ym, x1, ym)
    arrow(ax, 13.8, ym, 13.98, ym)
    ax.text(12.78, Y - 0.22, "device-ready waveforms", ha="center", va="top", fontsize=7, color=INK)

    # boundary of existing analog toolchains
    xb = 11.58
    ax.plot([xb, xb], [0.1, 4.3], color="#999999", lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.text(xb - 0.1, 4.25, "existing analog\ntoolchains end here", ha="right", va="top", fontsize=7,
            color=SEC, linespacing=1.05)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    for out in (FIGDIR, OUT2, OUT3):
        os.makedirs(out, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out, f"F_lowering.{ext}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote F_lowering.pdf/.png ->", FIGDIR, OUT2, OUT3)


if __name__ == "__main__":
    main()
