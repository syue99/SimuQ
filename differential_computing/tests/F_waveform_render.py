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

EVERY ROW IS THE REAL WAVEFORM — the compiled AWG samples themselves, Re w(t),
not an envelope and not a derived quantity.  The figure is evidence of an
end-to-end path, so what it shows has to be what the hardware would play.

At 120 us across a page a ~100 MHz carrier cannot be resolved, so each row is
drawn the way an instrument draws a sampled record: MIN AND MAX of the real
samples in every display bin, filled between.  The band's outline is therefore
the true excursion of the signal at that time, and no sample is invented or
smoothed away.  The zoom panels underneath resolve the carrier from the same
schedules at 0.5 ns, so the bands can be read as the oscillations they are:
the transport AOD's chirped tone mid-move, the addressing comb, and the measured
gate pulse.
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

# Column widths: wide PSR, narrow NSR.  The NSR column can be kept thin because
# its content is four uniform blocks that stay readable at any width, so the
# widths can lean closer to the real duration ratio than the layout first did.
# Still not the true 24:1 — the axis spans carry the actual numbers.
WIDTH_RATIO = (6.0, 1.0)

# the two transport channels, and which axis each steers
AOD_CHANNELS = {0: "x", 5: "y"}

# display order groups the two transport axes together; (channel id, label, kind)
ORDER = [(0, "transport AOD $x$", "pos"), (5, "transport AOD $y$", "pos"),
         (1, "addressing: detuning", "env"), (2, "addressing: Rabi", "env"),
         (3, "dressing AOM", "env"), (4, "gate AOM", "env")]


def _aod_record(meta, arrays, lanes, tag, axis, t_us, home):
    """The transport AOD record: one constant-amplitude tone per atom, ALWAYS ON.

    The tweezer tone is what holds the atoms; it is a standing condition of the
    trap, not a scheduled operation, so the compiler emits transport entries only
    where atoms MOVE.  Reading the sample array literally therefore shows silence
    wherever the pair is simply being held — including the whole NSR branch,
    which moves nothing — and a silent AOD would mean a dropped register.

    So this row, alone in the figure, is reconstructed rather than read: the tone
    frequency comes from the real schedule where it exists and sits at the atom's
    held position elsewhere, and the record is the sum of those tones.  MHz x us
    = cycles, so the phase is 2*pi times the running integral of f over t.
    """
    base = float(meta["aod_base_mhz"])
    kappa = float(meta["aod_kappa_mhz_per_um"])
    ntone = lanes[tag].get(f"n_tones_{axis}", 0)
    w = np.zeros_like(t_us)
    for atom in range(2):
        if ntone:
            segs = sorted(((arrays[f"{tag}_tone{axis}{k}_t"] / 1e3,
                            arrays[f"{tag}_tone{axis}{k}_mhz"])
                           for k in range(atom, ntone, 2)),
                          key=lambda sg: sg[0][0])
            tt = np.concatenate([sg[0] for sg in segs])
            ff = np.concatenate([sg[1] for sg in segs])
            order = np.argsort(tt)
            f = np.interp(t_us, tt[order], ff[order])     # holds at both ends
        else:
            f = np.full_like(t_us, base + kappa * home[atom])
        phase = 2.0 * np.pi * np.concatenate(
            ([0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(t_us))))
        w = w + np.cos(phase)
    return w


def _minmax(t, y, nbins=900):
    """Min and max of the real samples in each display bin — how a scope or an
    AWG viewer draws a record too dense to resolve.  Keeps the true excursion of
    the signal; never smooths, never interpolates."""
    if len(t) == 0:
        return t, y, y
    span = t[-1] - t[0]
    if span <= 0 or len(t) <= nbins:
        return t, y, y
    idx = np.minimum(((t - t[0]) / span * nbins).astype(int), nbins - 1)
    hi = np.full(nbins, -np.inf)
    lo = np.full(nbins, np.inf)
    np.maximum.at(hi, idx, y)
    np.minimum.at(lo, idx, y)
    seen = np.isfinite(hi)
    centres = t[0] + (np.arange(nbins) + 0.5) * span / nbins
    return centres[seen], lo[seen], hi[seen]


def render(meta, arrays):
    lanes = meta["lanes"]
    ends = {t: lanes[t]["t_end_ns"] / 1e3 for t in ("psr", "nsr")}
    bounds_us = [b / 1e3 for b in lanes["psr"]["bounds_ns"]]
    gw = lanes.get("gate_window_ns")

    fig = plt.figure(figsize=(7.0, 3.9), dpi=300)
    fig.patch.set_facecolor(C_SURFACE)
    gs = GridSpec(8, 2, figure=fig, width_ratios=list(WIDTH_RATIO),
                  height_ratios=[1.25, 1.25] + [1] * 4 + [0.62, 1.25],
                  hspace=0.24, wspace=0.055,
                  left=0.152, right=0.995, top=0.93, bottom=0.07)

    # one amplitude scale per channel, shared by the lanes, so the two columns
    # are directly comparable row by row
    scale = {}
    for cid, _label, _kind in ORDER:
        m = max(float(np.abs(arrays[f"{t}_ch{cid}"].real).max())
                for t in ("psr", "nsr"))
        scale[cid] = max(m, 1e-30)

    # atom positions: the interaction-zone geometry, and the AOD calibration
    home = {"x": [0.0, float(meta["sol_gvars"][0])],    # atom 0 at the origin
            "y": [0.0, float(meta["sol_gvars"][1])]}
    plim = [min(home["x"]) - 14.0, 118.0]

    cols = {}
    for col, (tag, colour) in enumerate((("psr", C_PSR), ("nsr", C_NSR))):
        t_us = arrays[f"{tag}_t"] / 1e3

        rows = []
        for i, (cid, label, _kind) in enumerate(ORDER):
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

            if cid in AOD_CHANNELS:            # always-on trap tone
                axis = AOD_CHANNELS[cid]
                w = _aod_record(meta, arrays, lanes, tag, axis, t_us,
                                home[axis]) / 2.0
            else:
                w = arrays[f"{tag}_ch{cid}"].real / scale[cid]
            nb = max(int(900 * WIDTH_RATIO[col] / sum(WIDTH_RATIO)), 60)
            tb, lo, hi = _minmax(t_us, w, nbins=nb)
            # span zero so a baseband channel (no carrier, min == max) is drawn
            # as a block rather than a zero-height fill.  The tweezer rows are
            # faint: there the record is context for the position line drawn
            # over it, so it has to be present but must not compete with it.
            ax.fill_between(tb, np.minimum(lo, 0.0), np.maximum(hi, 0.0),
                            color=colour, lw=0.0,
                            alpha=0.26 if cid in AOD_CHANNELS else 0.9)
            ax.set_ylim(-1.22, 1.22)
            ax.set_yticks([])
            ax.axhline(0.0, color=C_GRID, lw=0.35, zorder=0)
            if not lanes[tag]["active"][cid] and cid not in AOD_CHANNELS:
                ax.text(0.5, 0.5, "silent", transform=ax.transAxes,
                        fontsize=5.8, color="#9a9890", ha="center",
                        va="center", style="italic")

            if cid in AOD_CHANNELS:
                # SECOND AXIS on the same row: where the atoms are, over the
                # record that puts them there.  Same trace as the tone, through
                # the device's linear calibration.
                axis = AOD_CHANNELS[cid]
                axq = ax.twinx()
                lim = (plim if axis == "x" else
                       [min(home["y"]) - 1.4, max(home["y"]) + 6.4])
                axq.set_ylim(*lim)
                axq.set_xlim(0, ends[tag])
                for sp in axq.spines.values():
                    sp.set_visible(False)
                axq.tick_params(labelsize=5.4, colors=C_INK, length=2, pad=1.2)
                ntone = lanes[tag].get(f"n_tones_{axis}", 0)
                if ntone:
                    for atom in range(min(2, ntone)):
                        segs = sorted(
                            ((arrays[f"{tag}_tone{axis}{k}_t"] / 1e3,
                              arrays[f"{tag}_tone{axis}{k}_um"])
                             for k in range(atom, ntone, 2)),
                            key=lambda sg: sg[0][0])
                        axq.plot(np.concatenate([sg[0] for sg in segs]),
                                 np.concatenate([sg[1] for sg in segs]),
                                 color=C_INK, lw=1.0, zorder=6,
                                 solid_joinstyle="round")
                else:
                    for h in home[axis]:          # held, not moved
                        axq.plot([0, ends[tag]], [h, h], color=C_INK, lw=1.0,
                                 zorder=6)
                if col == 1:
                    axq.set_ylabel(r"$\mu$m", rotation=0, ha="left",
                                   va="center", fontsize=6.0, color=C_INK,
                                   labelpad=3)
                else:
                    axq.set_yticklabels([])

            if tag == "psr":                     # transport window and the gate
                ax.axvspan(bounds_us[1], bounds_us[-2], color=C_MOVE,
                           alpha=0.05, lw=0, zorder=0)
                if gw:
                    ax.axvspan(gw[0] / 1e3, gw[1] / 1e3, color=C_GATE,
                               alpha=0.22, lw=0, zorder=0)
            if i < len(ORDER) - 1:
                ax.set_xticklabels([])
        rows[-1].set_xlabel(r"$\mu$s", fontsize=6.5, color=C_INK, labelpad=1)
        cols[tag] = rows

    fig.text(0.16, 0.965, "PSR (kick)", fontsize=7.5, color=C_PSR,
             va="bottom", weight="bold")
    fig.text(0.865, 0.965, "NSR (waveform shift)", fontsize=7.5, color=C_NSR,
             va="bottom", ha="right", weight="bold")

    # ── carrier resolved, same schedules, 0.5 ns ──
    zooms = []
    if "aod_t" in arrays:
        zooms.append((arrays["aod_t"], arrays["aod_w"], C_PSR,
                      "transport AOD, mid-move"))
    zooms.append((arrays["psr_inset_t"], arrays["psr_inset_w"], C_PSR,
                  "addressing comb"))
    if "gate_t" in arrays:
        zooms.append((arrays["gate_t"], arrays["gate_w"], C_GATE,
                      f"gate pulse, {meta['gate_us'] * 1e3:.0f} ns"))
    zgs = gs[7, :].subgridspec(1, len(zooms), wspace=0.22)
    for j, (tz, wz, cz, title) in enumerate(zooms):
        az = fig.add_subplot(zgs[0, j])
        az.plot(tz, np.real(wz), color=cz, lw=0.45)
        az.set_facecolor(C_SURFACE)
        az.set_xlim(tz[0], tz[-1])
        az.tick_params(labelsize=5.6, colors=C_SEC, length=2, pad=1.2)
        az.set_yticks([])
        for sp in az.spines.values():
            sp.set_color(C_GRID)
            sp.set_linewidth(0.6)
        az.set_title(title + "  (0.5 ns)", fontsize=6.2, color=C_INK, pad=2)
        az.set_xlabel("ns", fontsize=6.0, color=C_SEC, labelpad=0.5)
    os.makedirs(FIG_DIR, exist_ok=True)
    for out in (FIG_DIR, OUT2, OUT3):
        os.makedirs(out, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out, f"F_waveform.{ext}"),
                        bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"rendered F_waveform.pdf/.png -> {FIG_DIR}, {OUT2}, {OUT3}")
