"""
build_F_cycle.py — App F (fig:cycle): the experiment cycle a programmed window
sits inside, and which apparatus class drives what.

REDRAWN, NOT COPIED, AND NOT THIS PAPER'S DEVICE.  The layout follows the
published sequence of \\cite{cycle-source} at the level of "what happens in what
order", and nothing else: every lab-specific label is replaced by its FUNCTION
(load / image & repair / cool / set qubits to |0> / readout), every duration is
an order-of-magnitude RANGE rather than that lab's measured value, and no trace
reproduces a measured curve.  The guard for this is a unit test that fails if
any source-specific vocabulary reappears in a label.

\\cite{cycle-source} is a DIFFERENT apparatus from \\cite{device} — another Cs
atom array — taken as a representative example of the cycle a neutral-atom
register runs, not as a description of the target machine.  Do not merge the two
citations, and do not read any number here as a device parameter: the figure is
a generalization, and its content is the ORDER of the phases and which
apparatus class drives each, both of which are common to this class of machine.

The point of the figure: the two apparatus classes have disjoint jobs.

  slow control  (multifunction-I/O class, ms updates) — trap depth, cooling
                light, bias field, pump light, camera trigger.  Active through
                the whole frame and STATIC across the programmed window: these
                lines set up and read out a register, they do not compute.
  fast synthesis (AWG/RFSoC class, ns resolution, phase-coherent) — the window
                lines, collapsed here into one band.  SILENT through the entire
                frame and active only inside the boxed operation phase.

So everything this paper compiles lives in ONE phase of the cycle, driven by
hardware that is idle for all the others; that phase is what fig:schedules
expands.

Illustrative, not measured: this figure plots no data and has no cache.  The
axis is broken — phase widths are drawn for legibility, not in proportion to
durations that span some three decades — but the widths PRESERVE THE ORDER of
the durations, which a test enforces.  The durations themselves are printed on
the phase headers and the axis says what it is doing.

Run:  conda run -n qec_pg python differential_computing/tests/build_F_cycle.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle

FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
OUT3 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "paper_fig_3", "figs"))

INK, SEC, GRID, SURFACE = "#0b0b0b", "#52514e", "#d8d7d0", "#fcfcfb"

# Colour language follows the source figure: pastel phase bands, one hue per
# control line, the operation window a plain white box.  The GRAMMAR is
# borrowed, the content is not — each hue here is attached to a function.
BAND = {"load": "#f6e7f4", "image": "#f9e2e0", "cool": "#dcecf7",
        "prep": "#f7f5da", "op": "#ffffff", "readout": "#f9e2e0"}
HDR = {"load": "#a4589f", "image": "#c0554a", "cool": "#3f92c0",
       "prep": "#8a8a2b", "op": INK, "readout": "#c0554a"}
C_TRAP, C_COOL, C_FIELD, C_PUMP, C_CAM = ("#7b3fa0", "#8b1a1a", "#7d7d7d",
                                          "#c99a3f", "#c0554a")
C_FAST = "#26268f"

# Number of fast lines the collapsed band stands for.  Keep this equal to the
# line count drawn in fig:schedules; the thumbnail draws that many.
N_FAST_LINES = 8

# (key, header, duration text, drawn width, representative ms).  WIDTHS ARE NOT
# DURATIONS: the durations span some three decades, so proportional widths would
# leave every phase but the first invisible.  They do, however, PRESERVE THE
# ORDER — a phase drawn wider than another is longer than it — which a test
# enforces against the representative-ms column.  That column exists only to
# order the phases; it is never drawn and is not a device number.
PHASES = [
    ("load",    "load register",           r"$\sim$100s of ms", 3.00, 300.0),
    ("image",   "image & repair register", r"$\sim$100 ms",     2.45, 100.0),
    ("readout", "readout image",           r"$\sim$10s of ms",  2.20,  50.0),
    ("cool",    "cool",                    r"$\sim$10 ms",      1.35,  10.0),
    ("op",      "Operation",               r"$\sim$1–10 ms",    1.10,   5.0),
    ("prep",    r"set qubits to $|0\rangle$", r"$\sim$1 ms",    0.85,   1.0),
]
# Drawing order along the axis (PHASES above is written longest-first so the
# width/duration ordering is readable at a glance).
SEQUENCE = ["load", "image", "cool", "prep", "op", "readout"]

# Slow-control rows: (label, colour, level per phase in [0, 1]).  A level says
# "this line is doing something", not a calibrated amplitude.  Every row holds
# ONE level across the operation window: that is the claim being drawn.
SLOW_ROWS = [
    ("trap depth",     C_TRAP,  dict(load=0.55, image=0.85, cool=0.35,
                                     prep=0.85, op=0.85, readout=0.85)),
    ("cooling light",  C_COOL,  dict(load=0.80, image=0.00, cool=0.80,
                                     prep=0.00, op=0.00, readout=0.00)),
    ("bias field",     C_FIELD, dict(load=0.30, image=0.75, cool=0.75,
                                     prep=0.75, op=0.75, readout=0.75)),
    ("pump light",     C_PUMP,  dict(load=0.00, image=0.00, cool=0.00,
                                     prep=0.80, op=0.00, readout=0.00)),
    ("camera trigger", C_CAM,   dict(load=0.00, image=0.85, cool=0.00,
                                     prep=0.00, op=0.00, readout=0.85)),
]

# The fast band: zero everywhere but the operation window.  Asserted in tests.
FAST_ROW = ("window lines", dict(load=0.0, image=0.0, cool=0.0, prep=0.0,
                                 op=1.0, readout=0.0))


def phase_spans():
    """[(key, x0, x1)] in drawn coordinates."""
    width = {k: w for k, _h, _d, w, _ms in PHASES}
    out, x = [], 0.0
    for key in SEQUENCE:
        out.append((key, x, x + width[key]))
        x += width[key]
    return out


def step_trace(levels, spans, ramp=0.10):
    """Envelope across the frame: hold per phase, short ramp at each edge."""
    xs, ys = [], []
    for i, (key, x0, x1) in enumerate(spans):
        lv = levels[key]
        r = min(ramp, 0.35 * (x1 - x0))
        if i == 0:
            xs.append(x0); ys.append(lv)
        else:
            xs.append(x0 + r); ys.append(lv)
        xs.append(x1 - r if i < len(spans) - 1 else x1)
        ys.append(lv)
    return np.array(xs), np.array(ys)


def render():
    spans = phase_spans()
    span = {k: (a, b) for k, a, b in spans}
    total = spans[-1][2]
    ox0, ox1 = span["op"]
    oxm = 0.5 * (ox0 + ox1)

    fig, ax = plt.subplots(figsize=(7.0, 3.45), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(-0.195 * total, total * 1.02)
    ax.set_ylim(0, 11.8)
    ax.axis("off")

    Y_HDR = 10.45                            # phase name; duration below it
    TOP, BOT = 9.85, 3.95                    # phase-band extent
    y_slow = [9.20, 8.45, 7.70, 6.95, 6.20]
    y_fast = 4.60                            # its own row, below a clear gap
    H = 0.26

    # ── phase bands, boundaries, headers ──
    for key, x0, x1 in spans:
        ax.add_patch(Rectangle((x0, BOT), x1 - x0, TOP - BOT,
                               facecolor=BAND[key],
                               edgecolor=INK if key == "op" else "none",
                               lw=0.8 if key == "op" else 0, zorder=0))
    for _key, x0, _x1 in spans[1:]:
        ax.plot([x0, x0], [BOT, TOP], color="#9a9890", lw=0.6,
                ls=(0, (2.5, 2.0)), zorder=1)

    for key, hdr, dur, _w, _ms in PHASES:
        x0, x1 = span[key]
        xm = 0.5 * (x0 + x1)
        if key == "op":                      # named inside its box, as in the
            continue                         # source figure
        ax.text(xm, Y_HDR, hdr, ha="center", va="bottom", fontsize=6.8,
                color=HDR[key])
        ax.text(xm, Y_HDR - 0.34, dur, ha="center", va="bottom", fontsize=6.3,
                color=HDR[key])
    ax.text(oxm, 0.5 * (BOT + TOP) + 0.30, "Operation", rotation=90,
            ha="center", va="center", fontsize=7.0, color=INK, zorder=6)
    ax.text(oxm, Y_HDR - 0.34, r"$\sim$1–10 ms", ha="center", va="bottom",
            fontsize=6.3, color=INK)

    # ── slow-control rows ──
    for y, (label, colour, levels) in zip(y_slow, SLOW_ROWS):
        ax.plot([0, total], [y - H, y - H], color="#c9c8c2", lw=0.45,
                ls=(0, (1.2, 2.2)), zorder=1)
        xs, ys = step_trace(levels, spans)
        ax.plot(xs, y - H + 2 * H * ys, color=colour, lw=1.3,
                solid_joinstyle="round", zorder=3)
        ax.text(-0.022 * total, y, label, ha="right", va="center",
                fontsize=6.6, color=colour)
    # group bracket sits in its own gutter column, clear of the row labels;
    # the class descriptors are too long to rotate here and go under the axis
    ax.plot([-0.132 * total, -0.132 * total],
            [y_slow[-1] - 0.42, y_slow[0] + 0.42], color=SEC, lw=0.9)
    ax.text(-0.150 * total, 0.5 * (y_slow[0] + y_slow[-1]), "slow control",
            rotation=90, ha="center", va="center", fontsize=6.8, color=SEC)

    # ── fast-synthesis row, in the row stack where it belongs ──
    ax.plot([0, total], [y_fast - H, y_fast - H], color="#c9c8c2", lw=0.45,
            ls=(0, (1.2, 2.2)), zorder=1)
    ax.plot([0, ox0], [y_fast - H, y_fast - H], color=C_FAST, lw=1.3, zorder=3)
    ax.plot([ox1, total], [y_fast - H, y_fast - H], color=C_FAST, lw=1.3,
            zorder=3)
    ax.add_patch(Rectangle((ox0, y_fast - H), ox1 - ox0, 2 * H,
                           facecolor=C_FAST, lw=0, zorder=4))
    ax.text(-0.022 * total, y_fast, f"{N_FAST_LINES} window lines\n(collapsed)",
            ha="right", va="center", fontsize=6.6, color=C_FAST,
            linespacing=1.3)
    ax.plot([-0.132 * total, -0.132 * total], [y_fast - 0.75, y_fast + 0.75],
            color=C_FAST, lw=0.9)
    ax.text(-0.150 * total, y_fast, "fast synthesis", rotation=90,
            ha="center", va="center", fontsize=6.8, color=C_FAST)

    # ── time arrow, labelled as in the source figure ──
    ax.annotate("", xy=(total * 1.015, 3.45), xytext=(0, 3.45),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1))
    ax.text(total * 1.02, 3.22, "time", ha="right", va="top", fontsize=6.6,
            color=INK)
    ax.text(0, 3.22, "broken axis: widths keep the ORDER of the durations, "
            "not their ratios", ha="left", va="top", fontsize=6.2, color=SEC)
    ax.text(0, 2.72, "slow control: multifunction-I/O class, ms updates",
            ha="left", va="top", fontsize=6.3, color=SEC)
    ax.text(0, 2.30, "fast synthesis: AWG/RFSoC class, ns resolution, "
            "phase-coherent", ha="left", va="top", fontsize=6.3, color=C_FAST)

    # ── zoom callout: the window expands into the schedule ──
    tx0, tx1, ty0, ty1 = 0.545 * total, 0.995 * total, 0.30, 2.35
    ax.add_patch(Rectangle((tx0, ty0), tx1 - tx0, ty1 - ty0,
                           facecolor="white", edgecolor=C_FAST, lw=0.7,
                           zorder=4))
    rng = np.random.default_rng(7)
    for i in range(N_FAST_LINES):
        yy = ty1 - 0.26 - i * (ty1 - ty0 - 0.55) / max(N_FAST_LINES - 1, 1)
        ax.plot([tx0 + 0.14, tx1 - 0.14], [yy, yy], color="#dedcd6", lw=0.4,
                zorder=5)
        x = tx0 + 0.14
        while x < tx1 - 0.32:
            w = 0.12 + 0.46 * rng.random()
            if rng.random() < 0.60:
                ax.plot([x, min(x + w, tx1 - 0.14)], [yy, yy], color=C_FAST,
                        lw=1.3, solid_capstyle="butt", zorder=6)
            x += w + 0.07
    ax.text(0.5 * (tx0 + tx1), ty0 - 0.06,
            "the window schedule (Figure 13)", ha="center", va="top",
            fontsize=6.3, color=C_FAST)
    for xw, xt in ((ox0, tx0), (ox1, tx1)):
        ax.add_artist(ConnectionPatch((xw, y_fast - H), (xt, ty1),
                                      "data", "data", axesA=ax, axesB=ax,
                                      color=C_FAST, lw=0.6, alpha=0.6,
                                      ls=(0, (3, 2)), zorder=2))

    fig.tight_layout(pad=0.2)
    os.makedirs(FIGDIR, exist_ok=True)
    for out in (FIGDIR, OUT3):
        os.makedirs(out, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out, f"F_cycle.{ext}"),
                        bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote F_cycle.pdf/.png -> {FIGDIR}, {OUT3}")


CAPTION = """Figure (App F). The experiment cycle a programmed window sits inside.
A generalization, not a device description: the sequence is redrawn after
\\cite{cycle-source} -- a different Cs atom array, taken as a representative
example of this class of machine, NOT the apparatus of \\cite{device} -- with
every lab-specific label replaced by its function and every duration given as an
order of magnitude rather than that lab's measured value. No trace here is data,
and no number should be read as a device parameter. The colour language -- pastel
phase bands, one hue per control line, the operation window a plain white box --
follows the source; the content attached to each hue does not. The time axis is
broken: phases are drawn at legible widths, not in proportion to durations that
span some four decades. What generalizes, and what the figure is for, is the
ORDER of the phases and which apparatus class drives each.

The figure is about the division of labour between two apparatus classes. The
slow-control lines -- trap depth, cooling light, bias field, pump light, camera
trigger, on multifunction-I/O hardware with millisecond updates -- run through
the whole frame and hold ONE level across the programmed window: they prepare
and read out a register, they do not compute. The fast-synthesis lines -- the
window lines, on AWG/RFSoC-class hardware with nanosecond resolution and
phase coherence, collapsed here into a single band -- are silent for the entire
cycle and active only inside the operation window, the boxed phase of order
1--10 ms.

Everything this paper compiles therefore lives in one phase of the cycle, driven
by hardware that is idle for all the others, and it is that phase which
fig:schedules expands.
"""

BIB = """% A DIFFERENT apparatus from \\cite{device} -- another Cs atom array,
% cited as a representative example of the experiment cycle, not as the target
% machine.  Do NOT merge with \\cite{device}.
@article{cycle-source,
  note = {Nature, 2025; doi:10.1038/s41586-025-09641-4}
}
"""


def main():
    render()
    with open(os.path.join(FIGDIR, "F_cycle_caption.txt"), "w") as f:
        f.write(CAPTION)
    with open(os.path.join(FIGDIR, "F_cycle.bib"), "w") as f:
        f.write(BIB)
    print(f"wrote F_cycle_caption.txt and F_cycle.bib -> {FIGDIR}")


if __name__ == "__main__":
    main()
