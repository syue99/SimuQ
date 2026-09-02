"""
F_waveform_render.py — render App F's two-lane emission figure from the cache
written by build_F_waveform.extract().  Plot-only: it never touches the
pipeline, so styling changes cost nothing.

Layout: PSR left, NSR right, six physical channel rows each, on their own time
axes with the COLUMN WIDTHS carrying the duration difference — a wide PSR
column beside a narrow NSR one.  The axes are labelled with their real spans,
so the durations are readable without the figure asserting anything about them.

Deliberately NOT on this figure (owner ruling, 09-02): per-branch speedup
factors and percentages.  Per-branch wall clock is not the operative cost on a
real machine — measurement and atom loading / rearrangement dominate, and the
us-vs-ms difference between the two branches is not what decides anything.  The
claim this figure supports is STRUCTURAL: which channels a branch keys at all.
No title either; titles live in the LaTeX caption.

Transport rows show the AOD tone FREQUENCY mapped to atom position
(constant-amplitude tones carry no envelope information); the four drive/gate
rows show |A(t)|, peak-held to display resolution so the addressing comb's beat
is drawn rather than aliased.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

FIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
OUT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "paper_fig_2"))
OUT3 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "paper_fig_3", "figs"))

C_PSR, C_NSR = "#0072B2", "#009E73"
C_INK, C_SEC, C_GRID, C_SURFACE = "#0b0b0b", "#52514e", "#d8d7d0", "#fcfcfb"
C_MOVE, C_GATE = "#d9822b", "#d62728"

# Column widths: wide PSR, narrow NSR.  Not the true 24:1 — at that ratio the
# NSR column is 4% of the figure and unreadable, which is the failure mode this
# layout exists to fix.  The axis spans carry the actual numbers.
WIDTH_RATIO = (3.4, 1.0)

# display order groups the two transport axes together; (channel id, label, kind)
ORDER = [(0, "transport AOD $x$", "pos"), (5, "transport AOD $y$", "pos"),
         (1, "addressing: detuning", "env"), (2, "addressing: Rabi", "env"),
         (3, "dressing AOM", "env"), (4, "gate AOM", "env")]


def _peak_hold(t, y, nbins=700):
    """Envelope down to display resolution by PEAK HOLD, the way a scope draws.

    Plain decimation of |A| would alias the addressing comb's ~100 ns beat into
    a fake ragged envelope; the max over each display bin keeps the true upper
    envelope at any zoom.
    """
    if len(t) <= nbins:
        return t, y
    span = t[-1] - t[0]
    if span <= 0:
        return t, y
    idx = np.minimum(((t - t[0]) / span * nbins).astype(int), nbins - 1)
    out = np.zeros(nbins)
    np.maximum.at(out, idx, y)
    return t[0] + (np.arange(nbins) + 0.5) * span / nbins, out


def render(meta, arrays):
    lanes = meta["lanes"]
    ends = {t: lanes[t]["t_end_ns"] / 1e3 for t in ("psr", "nsr")}
    bounds_us = [b / 1e3 for b in lanes["psr"]["bounds_ns"]]
    gw = lanes.get("gate_window_ns")

    fig = plt.figure(figsize=(7.0, 2.55), dpi=300)
    fig.patch.set_facecolor(C_SURFACE)
    gs = GridSpec(6, 2, figure=fig, width_ratios=list(WIDTH_RATIO),
                  hspace=0.22, wspace=0.055,
                  left=0.152, right=0.995, top=0.90, bottom=0.145)

    # per-axis position scale: x spans the 100 um zone hop, y only the 5 um
    # transit lane, so one shared scale would hide the lane
    pos_max = {}
    for axis in ("x", "y"):
        m = 0.0
        for tag in ("psr", "nsr"):
            for i in range(lanes[tag].get(f"n_tones_{axis}", 0)):
                m = max(m, float(np.abs(arrays[f"{tag}_tone{axis}{i}_um"]).max()))
        pos_max[axis] = max(m, 1.0) * 1.18

    cols = {}
    for col, (tag, colour) in enumerate((("psr", C_PSR), ("nsr", C_NSR))):
        t_us = arrays[f"{tag}_t"] / 1e3
        rows = []
        for i, (cid, label, kind) in enumerate(ORDER):
            ax = fig.add_subplot(gs[i, col])
            rows.append(ax)
            ax.set_xlim(0, ends[tag])
            ax.set_facecolor(C_SURFACE)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.spines["bottom"].set_visible(True)
            ax.spines["bottom"].set_color(C_GRID)
            ax.spines["bottom"].set_linewidth(0.6)
            ax.tick_params(labelsize=6, colors=C_SEC, length=2, pad=1.5)
            if col == 0:
                ax.set_ylabel(label, rotation=0, ha="right", va="center",
                              fontsize=6.5, color=C_INK, labelpad=4)

            if kind == "pos":
                axis = "x" if cid == 0 else "y"
                ntone = lanes[tag].get(f"n_tones_{axis}", 0)
                for k in range(ntone):
                    ax.plot(arrays[f"{tag}_tone{axis}{k}_t"] / 1e3,
                            np.abs(arrays[f"{tag}_tone{axis}{k}_um"]),
                            color=colour, lw=0.9, solid_capstyle="round")
                pm = pos_max[axis]
                ax.set_ylim(-0.06 * pm, pm)
                top = pm / 1.18
                step = 10 ** np.floor(np.log10(top))
                ax.set_yticks([0, float(np.floor(top / step) * step)])
                ax.tick_params(axis="y", labelsize=5.4)
                if col == 1:
                    ax.set_yticklabels([])
                if ntone == 0:
                    ax.text(0.5, 0.5, "silent", transform=ax.transAxes,
                            fontsize=5.8, color="#9a9890", ha="center",
                            va="center", style="italic")
                elif tag == "psr" and cid == 0:
                    ax.text(0.012, 0.66, r"$\mu$m", transform=ax.transAxes,
                            fontsize=5.4, color=C_SEC, va="center")
            else:
                w = arrays[f"{tag}_ch{cid}"]
                other = arrays[f"{'nsr' if tag == 'psr' else 'psr'}_ch{cid}"]
                norm = max(np.abs(w).max(), np.abs(other).max(), 1e-30)
                tb, y = _peak_hold(t_us, np.abs(w) / norm)
                ax.fill_between(tb, 0.0, y, color=colour, lw=0.0, alpha=0.85)
                ax.set_ylim(0, 1.18)
                ax.set_yticks([])
                if not lanes[tag]["active"][cid]:
                    ax.text(0.5, 0.5, "silent", transform=ax.transAxes,
                            fontsize=5.8, color="#9a9890", ha="center",
                            va="center", style="italic")

            if tag == "psr":                     # transport window and the gate
                ax.axvspan(bounds_us[1], bounds_us[-2], color=C_MOVE,
                           alpha=0.06, lw=0, zorder=0)
                if gw:
                    ax.axvspan(gw[0] / 1e3, gw[1] / 1e3, color=C_GATE,
                               alpha=0.25, lw=0, zorder=0)
            if i < len(ORDER) - 1:
                ax.set_xticklabels([])
        rows[-1].set_xlabel(r"$\mu$s", fontsize=6.5, color=C_INK, labelpad=1)
        cols[tag] = rows

    cols["psr"][0].text(0.0, 1.30, "PSR (kick)",
                        transform=cols["psr"][0].transAxes, fontsize=7.5,
                        color=C_PSR, va="bottom", weight="bold")
    cols["nsr"][0].text(0.0, 1.30, "NSR (waveform shift)",
                        transform=cols["nsr"][0].transAxes, fontsize=7.5,
                        color=C_NSR, va="bottom", weight="bold")

    os.makedirs(FIG_DIR, exist_ok=True)
    for out in (FIG_DIR, OUT2, OUT3):
        os.makedirs(out, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out, f"F_waveform.{ext}"),
                        bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"rendered F_waveform.pdf/.png -> {FIG_DIR}, {OUT2}, {OUT3}")
