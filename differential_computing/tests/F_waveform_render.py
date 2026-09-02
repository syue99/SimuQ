"""
F_waveform_render.py — render App F's two-lane emission figure from the cache
written by build_F_waveform.extract().  Plot-only: it never touches the
pipeline, so styling changes cost nothing.

Layout: the two lanes SIDE BY SIDE, PSR left and NSR right, six physical
channel rows each, each column on its own time axis — the NSR branch is 24x
shorter, and on a shared absolute axis it collapses into a sliver at the origin
where none of its structure is readable.  The scale difference is then stated
rather than implied: the duration strip under the columns is the one place the
two lanes are commensurable, and it carries the 24x.

Transport rows show the AOD tone FREQUENCY mapped to atom position
(constant-amplitude tones carry no envelope information); the four drive/gate
rows show |A(t)|.  Two carrier-resolved insets show what the envelope rows
compress.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

FIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
OUT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "paper_fig_2"))
OUT3 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "paper_fig_3", "figs"))

C_PSR, C_NSR = "#0072B2", "#009E73"
C_INK, C_SEC, C_GRID, C_SURFACE = "#0b0b0b", "#52514e", "#d8d7d0", "#fcfcfb"
C_MOVE, C_GATE = "#d9822b", "#d62728"
HALO = [pe.withStroke(linewidth=1.9, foreground="white")]

# display order groups the two transport axes together; (channel id, label, kind)
ORDER = [(0, "transport AOD $x$", "pos"), (5, "transport AOD $y$", "pos"),
         (1, "addressing: detuning", "env"), (2, "addressing: Rabi", "env"),
         (3, "dressing AOM", "env"), (4, "gate AOM", "env")]


def _peak_hold(t, y, nbins=900):
    """Envelope down to display resolution by PEAK HOLD, the way a scope draws.

    Plain decimation of |A| would alias the addressing comb's ~100 ns beat into
    a fake ragged envelope; taking the max over each display bin keeps the true
    upper envelope at any zoom.
    """
    if len(t) <= nbins:
        return t, y
    span = t[-1] - t[0]
    if span <= 0:
        return t, y
    idx = np.minimum(((t - t[0]) / span * nbins).astype(int), nbins - 1)
    out = np.zeros(nbins)
    np.maximum.at(out, idx, y)
    centres = t[0] + (np.arange(nbins) + 0.5) * span / nbins
    return centres, out


def _frame(ax, label=None):
    ax.set_facecolor(C_SURFACE)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(C_GRID)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(labelsize=6, colors=C_SEC, length=2)
    if label is not None:
        ax.set_ylabel(label, rotation=0, ha="right", va="center",
                      fontsize=6.4, color=C_INK, labelpad=5)


def render(meta, arrays):
    lanes = meta["lanes"]
    ends = {t: lanes[t]["t_end_ns"] / 1e3 for t in ("psr", "nsr")}
    bounds_us = [b / 1e3 for b in lanes["psr"]["bounds_ns"]]
    gw = lanes.get("gate_window_ns")
    ratio = ends["psr"] / ends["nsr"]

    fig = plt.figure(figsize=(7.1, 6.1), dpi=300)
    fig.patch.set_facecolor(C_SURFACE)
    gs = GridSpec(10, 2, figure=fig,
                  height_ratios=[1] * 6 + [0.95, 0.55, 0.55, 1.30],
                  hspace=0.24, wspace=0.09,
                  left=0.155, right=0.985, top=0.875, bottom=0.055)

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
            _frame(ax, label if col == 0 else None)

            if kind == "pos":
                axis = "x" if cid == 0 else "y"
                ntone = lanes[tag].get(f"n_tones_{axis}", 0)
                for k in range(ntone):
                    ax.plot(arrays[f"{tag}_tone{axis}{k}_t"] / 1e3,
                            np.abs(arrays[f"{tag}_tone{axis}{k}_um"]),
                            color=colour, lw=1.0, solid_capstyle="round")
                pm = pos_max[axis]
                ax.set_ylim(-0.06 * pm, pm)
                top = pm / 1.18
                step = 10 ** np.floor(np.log10(top))
                ax.set_yticks([0, float(np.floor(top / step) * step)])
                ax.tick_params(axis="y", labelsize=5.6)
                if col == 1:
                    ax.set_yticklabels([])
                if ntone == 0:
                    ax.text(0.5, 0.55, "silent", transform=ax.transAxes,
                            fontsize=6.0, color="#9a9890", ha="center",
                            va="center", style="italic")
                elif tag == "psr" and cid == 0:
                    ax.text(0.015, 0.62, r"atom position ($\mu$m)",
                            transform=ax.transAxes, fontsize=5.6, color=C_SEC,
                            va="center", path_effects=HALO)
            else:
                w = arrays[f"{tag}_ch{cid}"]
                other = arrays[f"{'nsr' if tag == 'psr' else 'psr'}_ch{cid}"]
                norm = max(np.abs(w).max(), np.abs(other).max(), 1e-30)
                tb, y = _peak_hold(t_us, np.abs(w) / norm)
                ax.fill_between(tb, 0.0, y, color=colour, lw=0.0, alpha=0.85)
                ax.plot(tb, y, color=colour, lw=0.4)
                ax.set_ylim(0, 1.20)
                ax.set_yticks([])
                if not lanes[tag]["active"][cid]:
                    ax.text(0.5, 0.55, "silent", transform=ax.transAxes,
                            fontsize=6.0, color="#9a9890", ha="center",
                            va="center", style="italic")
                elif tag == "nsr" and cid == 1:
                    # the ripple is physical, not sampling noise: label it once
                    ax.text(0.5, 1.06,
                            r"COMB: one tone per atom, 10 MHz apart $\Rightarrow$ "
                            r"$|A|$ beats at ${\sim}100$ ns",
                            transform=ax.transAxes, fontsize=5.6, color=C_SEC,
                            ha="center", va="bottom")

            if tag == "psr":                     # transport window and the gate
                ax.axvspan(bounds_us[1], bounds_us[-2], color=C_MOVE,
                           alpha=0.06, lw=0, zorder=0)
                if gw:
                    ax.axvspan(gw[0] / 1e3, gw[1] / 1e3, color=C_GATE,
                               alpha=0.22, lw=0, zorder=0)
            if i < len(ORDER) - 1:
                ax.set_xticklabels([])
        rows[-1].set_xlabel(rf"time within one branch  ($\mu$s)"
                            rf"  —  axis spans 0–{ends[tag]:.4g}",
                            fontsize=6.6, color=C_INK, labelpad=2)
        cols[tag] = rows

    # ── column headers ──
    tr_us = meta["psr_transport_ns"] / 1e3
    cols["psr"][0].text(0.0, 1.46, "PSR (kick)",
                        transform=cols["psr"][0].transAxes, fontsize=8.4,
                        color=C_PSR, va="bottom", weight="bold")
    cols["psr"][0].text(
        0.0, 1.19,
        f"one branch = {ends['psr']:.2f} " + r"$\mu$s;  "
        + f"{100 * tr_us / ends['psr']:.0f}% of it transport + gate",
        transform=cols["psr"][0].transAxes, fontsize=6.5, color="#8a4b00",
        va="bottom")
    cols["nsr"][0].text(0.0, 1.46, "NSR (waveform shift)",
                        transform=cols["nsr"][0].transAxes, fontsize=8.4,
                        color=C_NSR, va="bottom", weight="bold")
    cols["nsr"][0].text(
        0.0, 1.19,
        f"one branch = {ends['nsr']:.2f} " + r"$\mu$s $=T$;  "
        + f"{ratio:.0f}" + r"$\times$ shorter, same geometry",
        transform=cols["nsr"][0].transAxes, fontsize=6.5, color="#0f6b52",
        va="bottom")

    # ── the one commensurable view: both branch durations on a shared axis ──
    axS = fig.add_subplot(gs[7, :])
    axS.set_facecolor(C_SURFACE)
    axS.barh([1], [ends["psr"]], height=0.5, color=C_PSR, alpha=0.85)
    axS.barh([1], [tr_us], left=bounds_us[1], height=0.5, color=C_MOVE,
             alpha=0.6)
    axS.barh([0], [ends["nsr"]], height=0.5, color=C_NSR, alpha=0.9)
    axS.set_xlim(0, ends["psr"] * 1.004)
    axS.set_ylim(-0.6, 1.6)
    axS.set_yticks([0, 1])
    axS.set_yticklabels(["NSR", "PSR"], fontsize=6.4, color=C_INK)
    axS.tick_params(labelsize=6, colors=C_SEC, length=2)
    for s in axS.spines.values():
        s.set_visible(False)
    axS.spines["bottom"].set_visible(True)
    axS.spines["bottom"].set_color(C_GRID)
    axS.set_xlabel(r"the columns above are on their own axes; this strip is the"
                   r" common one  ($\mu$s)", fontsize=6.4, color=C_SEC,
                   labelpad=1)
    axS.text(ends["nsr"] + ends["psr"] * 0.012, 0,
             f"{ends['nsr']:.2f} " + r"$\mu$s — one PSR branch buys "
             + f"{ratio:.0f} of these", fontsize=6.2, color="#0f6b52",
             va="center", path_effects=HALO)
    axS.text(bounds_us[1] + tr_us * 0.5, 1.34,
             f"transport + gate  {tr_us:.1f} " + r"$\mu$s", fontsize=6.2,
             color="#4a2800", ha="center", va="center")

    # ── carrier-resolved insets ──
    axg = fig.add_subplot(gs[9, 0])
    if "gate_t" in arrays:
        tg, wg = arrays["gate_t"], arrays["gate_w"]
        axg.plot(tg, np.real(wg), color=C_GATE, lw=0.3, alpha=0.85)
        axg.plot(tg, np.abs(wg), color=C_INK, lw=0.9)
        axg.set_xlabel("ns after gate start", fontsize=6.4, color=C_SEC,
                       labelpad=1)
        axg.set_title(f"gate AOM: measured shape, "
                      f"{meta['gate_us'] * 1e3:.0f} ns "
                      r"($|A|$ over the resolved carrier)",
                      fontsize=6.8, color=C_INK, pad=3)

    axd = fig.add_subplot(gs[9, 1])
    tp, wp = arrays["psr_inset_t"], arrays["psr_inset_w"]
    td, wd = arrays["nsr_inset_t"], arrays["nsr_inset_w"]
    win = 40.0                                   # ns — resolve the oscillation
    mp, md = tp <= win, td <= win
    axd.plot(tp[mp], np.real(wp[mp]), color=C_PSR, lw=2.2, alpha=0.9,
             label="PSR (source amplitude)")
    axd.plot(td[md], np.real(wd[md]), color=C_NSR, lw=1.0, ls=(0, (2.6, 1.8)),
             label="NSR (shifted)")
    peak = float(np.abs(np.real(wp[mp])).max())
    axd.set_xlabel("ns from start", fontsize=6.4, color=C_SEC, labelpad=1)
    axd.set_title("addressing Rabi comb: the shift is an amplitude change",
                  fontsize=6.8, color=C_INK, pad=3)
    axd.legend(fontsize=5.8, frameon=False, loc="lower center", ncol=2,
               handlelength=1.7, borderpad=0.15, labelspacing=0.2,
               columnspacing=1.1)
    axd.text(0.5, 0.985,
             r"identical carrier and phase; amplitude $\times$"
             + f"{meta['scale']:.3f}", transform=axd.transAxes, fontsize=5.9,
             color=C_SEC, va="top", ha="center", path_effects=HALO)
    axd.set_ylim(-1.75 * peak, 1.45 * peak)

    for ax in (axg, axd):
        ax.set_facecolor(C_SURFACE)
        ax.tick_params(labelsize=6, colors=C_SEC, length=2)
        for s in ax.spines.values():
            s.set_color(C_GRID)
            s.set_linewidth(0.6)

    fig.text(0.5, 0.985,
             r"running example  $H(x)=\sin(2x)\,(Z_0Z_1+X_0+X_1)$,  $x=0.7$,"
             r"  $T=5.0\ \mu$s, 2 qubits — ONE branch of each strategy, as "
             r"emitted on the six physical channels", fontsize=7.8,
             color=C_INK, ha="center", va="top")
    fig.text(0.5, 0.960,
             f"Nyquist mode $n$={meta['nsr_mode']},  shift $s$="
             f"{meta['s']:.4f}  ($K$={meta['K']:.3f});  machine residual "
             f"{meta['residual_source']:.1e} (source) / "
             f"{meta['residual_nsr']:.1e} (shifted).   A gradient needs many "
             r"branches either way — PSR spends them on a $\tau$ quadrature, "
             r"NSR on drawn shifts; how many is Fig. 10's question, not this "
             r"figure's.", fontsize=6.5, color=C_SEC, ha="center", va="top")

    os.makedirs(FIG_DIR, exist_ok=True)
    for out in (FIG_DIR, OUT2, OUT3):
        os.makedirs(out, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out, f"F_waveform.{ext}"),
                        bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"rendered F_waveform.pdf/.png -> {FIG_DIR}, {OUT2}, {OUT3}")
