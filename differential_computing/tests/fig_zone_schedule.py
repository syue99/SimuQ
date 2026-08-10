"""
fig_zone_schedule.py — Fig 4 (A3): zone architecture + one branch's compiled
schedule + physical channel activity.

Top: the three tweezer zones (interaction / gate / idle) atoms move between via
AOD transport.  Bottom: one PSR branch's three-segment schedule (ev, ±kick, ev)
on a time axis, with the five physical channels' activity blocks aligned to the
segments.  Schematic only (redrawn, no SimuQ reuse).
Saves figures/fig_zone_schedule.png.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Ellipse, Circle

FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
TEAL, ORANGE = "#00897b", "#e8710a"


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.6, 7.6), dpi=150)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ══ TOP: zone architecture ══
    ax.text(0.5, 0.975, "Tweezer zone architecture  (atoms move between zones by AOD transport)",
            ha="center", fontsize=11, fontweight="bold", color="#333")
    zy0, zy1 = 0.63, 0.93

    def zone(x0, x1, fc, ec, title, sub):
        ax.add_patch(FancyBboxPatch((x0, zy0), x1 - x0, zy1 - zy0,
                     boxstyle="round,pad=0.005,rounding_size=0.02", fc=fc, ec=ec, lw=1.6))
        ax.text((x0 + x1) / 2, zy1 - 0.028, title, ha="center", fontsize=10.5,
                fontweight="bold", color=ec)
        ax.text((x0 + x1) / 2, zy0 + 0.022, sub, ha="center", fontsize=8.3, color="#444")

    # interaction zone — cluster + dressing beam
    zone(0.05, 0.35, "#e8f0fe", "#2f5aa0", "INTERACTION (0,0)", "dressing + 1-qubit gates")
    ax.add_patch(Ellipse((0.20, 0.785), 0.16, 0.075, fc="#ffe9b0", ec="#e0a800",
                 lw=1, alpha=0.6, zorder=2))
    ax.text(0.20, 0.845, "dressing beam", ha="center", fontsize=7, color="#a06d00")
    for dx in (-0.05, -0.017, 0.017, 0.05):
        ax.add_patch(Circle((0.20 + dx, 0.785), 0.011, fc="#2f5aa0", ec="k", lw=0.6, zorder=3))
    # gate zone — a pair at R_target
    zone(0.39, 0.62, "#f3e8fd", "#7b3fb0", "GATE (1000,1000)", "2-qubit ZZ at R_target")
    for dx in (-0.028, 0.028):
        ax.add_patch(Circle((0.505 + dx, 0.78), 0.012, fc="#7b3fb0", ec="k", lw=0.6, zorder=3))
    ax.annotate("", xy=(0.533, 0.80), xytext=(0.477, 0.80),
                arrowprops=dict(arrowstyle="<->", color="#7b3fb0", lw=1)); \
        ax.text(0.505, 0.815, "R_target", ha="center", fontsize=7, color="#7b3fb0")
    # idle zone — parked line
    zone(0.66, 0.95, "#eeeeee", "#666666", "IDLE (-1000, …)", "parked qubits")
    for dy in (-0.03, 0, 0.03):
        ax.add_patch(Circle((0.805, 0.78 + dy), 0.011, fc="#666", ec="k", lw=0.6, zorder=3))
    # AOD transport arrows between zones
    for (xa, xb) in [(0.35, 0.39), (0.62, 0.66)]:
        ax.add_patch(FancyArrowPatch((xa, 0.755), (xb, 0.755), arrowstyle="<|-|>",
                     mutation_scale=12, lw=1.4, color="#c9700f", ls=(0, (4, 2))))
    ax.text(0.5, 0.605, "AOD transport", ha="center", fontsize=8, color="#c9700f", style="italic")

    # ══ BOTTOM: one branch's schedule ══
    ax.text(0.5, 0.545, "One PSR branch's compiled schedule",
            ha="center", fontsize=11, fontweight="bold", color="#333")
    xL, xR = 0.20, 0.93
    tau = xL + 0.42 * (xR - xL); kick_w = 0.05
    segy, segh = 0.46, 0.05
    ax.add_patch(Rectangle((xL, segy), tau - xL, segh, fc=TEAL, ec="k", lw=1))
    ax.text((xL + tau) / 2, segy + segh / 2, "ev(0, τ, H)", ha="center", va="center",
            color="w", family="monospace", fontsize=9)
    ax.add_patch(Rectangle((tau, segy), kick_w, segh, fc=ORANGE, ec="k", lw=1))
    ax.text(tau + kick_w / 2, segy + segh + 0.016, "kick H_j", ha="center", va="bottom",
            color=ORANGE, fontsize=8.5, fontweight="bold")
    ax.add_patch(Rectangle((tau + kick_w, segy), xR - tau - kick_w, segh, fc=TEAL, ec="k", lw=1))
    ax.text((tau + kick_w + xR) / 2, segy + segh / 2, "ev(τ, T, H)", ha="center",
            va="center", color="w", family="monospace", fontsize=9)
    # time axis
    ax.annotate("", xy=(xR + 0.01, segy - 0.03), xytext=(xL, segy - 0.03),
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.1))
    for xx, lab in [(xL, "0"), (tau, "τ"), (tau + kick_w, "τ+"), (xR, "T")]:
        ax.plot([xx, xx], [segy - 0.038, segy - 0.024], color="#555", lw=1)
        ax.text(xx, segy - 0.052, lab, ha="center", va="top", fontsize=8.5)

    # channel activity grid (rows) — . = idle, block = active
    rows = [  # (name, [ev1, kick, ev2]) activity, color
        ("DRESSING_AOM", [1, 0, 1], "#2f5aa0"),
        ("GATE_AOM",     [1, 0, 1], "#7b3fb0"),
        ("ADDR_RABI",    [1, 1, 1], "#00897b"),
        ("ADDR_DET",     [1, 0, 1], "#00897b"),
        ("TRANSPORT_AOD",[0, 1, 0], "#c9700f"),
    ]
    ry = 0.335; rh = 0.032; rgap = 0.010
    segs = [(xL, tau), (tau, tau + kick_w), (tau + kick_w, xR)]
    for name, act, col in rows:
        ax.text(xL - 0.015, ry + rh / 2, name, ha="right", va="center", fontsize=8,
                family="monospace", color="#333")
        for (sx0, sx1), a in zip(segs, act):
            if a:
                ax.add_patch(Rectangle((sx0, ry), sx1 - sx0, rh, fc=col, ec="w",
                             lw=0.8, alpha=0.85))
        ry -= (rh + rgap)

    ax.text(0.5, 0.05, "1-qubit-generator kick → ADDR_RABI in the interaction zone;   "
            "ZZ-generator kick → AOD transport to the gate zone (TRANSPORT_AOD + GATE_AOM)",
            ha="center", fontsize=7.8, color="#555", style="italic")
    ax.text(0.5, 0.018, "the PulseLedger records positions, zones, and per-segment channel "
            "meta-data — the verification IR", ha="center", fontsize=7.8, color="#888", style="italic")
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig_zone_schedule.png")
    fig.savefig(out); print(f"saved: {out}")


if __name__ == "__main__":
    main()
