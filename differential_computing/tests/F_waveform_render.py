"""
F_waveform_render.py — render App F's two-lane emission figure from the cache
written by build_F_waveform.extract().  Plot-only: it never touches the
pipeline, so styling changes cost nothing.

Layout: six physical channels x two lanes = twelve rows on ONE shared absolute
wall-clock axis, so the PSR lane's transport legs and the NSR lane's single
evolution block are drawn to the same scale.  Transport rows show the AOD tone
FREQUENCY mapped to atom position (constant-amplitude tones carry no envelope
information); the four drive/gate rows show |A(t)|.  Two carrier-resolved
insets underneath show what the envelope rows compress.
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


def _frame(ax, label, colour_label=C_INK):
    ax.set_facecolor(C_SURFACE)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(C_GRID)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(labelsize=6, colors=C_SEC, length=2)
    ax.set_ylabel(label, rotation=0, ha="right", va="center",
                  fontsize=6.4, color=colour_label, labelpad=5)


def render(meta, arrays):
    lanes = meta["lanes"]
    t_end_us = max(lanes["psr"]["t_end_ns"], lanes["nsr"]["t_end_ns"]) / 1e3
    bounds_us = [b / 1e3 for b in lanes["psr"]["bounds_ns"]]
    gw = lanes.get("gate_window_ns")

    fig = plt.figure(figsize=(7.1, 6.9), dpi=300)
    fig.patch.set_facecolor(C_SURFACE)
    gs = GridSpec(15, 2, figure=fig,
                  height_ratios=[1] * 6 + [0.75] + [1] * 6 + [1.05, 1.45],
                  hspace=0.18, wspace=0.17,
                  left=0.175, right=0.965, top=0.90, bottom=0.06)

    # each transport axis gets its OWN scale: x spans the 100 um zone hop,
    # y only the 5 um transit lane, so one shared scale would hide the lane
    pos_max = {}
    for axis in ("x", "y"):
        m = 0.0
        for tag in ("psr", "nsr"):
            for i in range(lanes[tag].get(f"n_tones_{axis}", 0)):
                m = max(m, float(np.abs(arrays[f"{tag}_tone{axis}{i}_um"]).max()))
        pos_max[axis] = max(m, 1.0) * 1.18

    blocks = {}
    for tag, colour, r0 in (("psr", C_PSR, 0), ("nsr", C_NSR, 7)):
        t_us = arrays[f"{tag}_t"] / 1e3
        rows = []
        for i, (cid, label, kind) in enumerate(ORDER):
            ax = fig.add_subplot(gs[r0 + i, :])
            rows.append(ax)
            ax.set_xlim(0, t_end_us)
            _frame(ax, label)

            if kind == "pos":
                axis = "x" if cid == 0 else "y"
                ntone = lanes[tag].get(f"n_tones_{axis}", 0)
                for k in range(ntone):
                    tt = arrays[f"{tag}_tone{axis}{k}_t"] / 1e3
                    uu = arrays[f"{tag}_tone{axis}{k}_um"]
                    ax.plot(tt, np.abs(uu), color=colour, lw=1.0,
                            solid_capstyle="round")
                pm = pos_max[axis]
                ax.set_ylim(-0.06 * pm, pm)
                top = pm / 1.18                       # a round tick, not 101
                step = 10 ** np.floor(np.log10(top))
                ax.set_yticks([0, float(np.floor(top / step) * step)])
                ax.tick_params(axis="y", labelsize=5.6)
                if ntone == 0:
                    ax.text(0.995, 0.55, "silent", transform=ax.transAxes,
                            fontsize=5.8, color="#9a9890", ha="right",
                            va="center", style="italic")
                elif tag == "psr" and cid == 0:
                    ax.text(0.012, 0.60, r"atom position ($\mu$m)",
                            transform=ax.transAxes, fontsize=5.6,
                            color=C_SEC, va="center", path_effects=HALO)
            else:
                w = arrays[f"{tag}_ch{cid}"]
                other = arrays[f"{'nsr' if tag == 'psr' else 'psr'}_ch{cid}"]
                norm = max(np.abs(w).max(), np.abs(other).max(), 1e-30)
                y = np.abs(w) / norm
                ax.fill_between(t_us, 0.0, y, color=colour, lw=0.0, alpha=0.85)
                ax.plot(t_us, y, color=colour, lw=0.4)
                ax.set_ylim(0, 1.20)
                ax.set_yticks([])
                if not lanes[tag]["active"][cid]:
                    ax.text(0.995, 0.55, "silent", transform=ax.transAxes,
                            fontsize=5.8, color="#9a9890", ha="right",
                            va="center", style="italic")

            # stage shading: the PSR lane's transport window and the gate
            if tag == "psr":
                ax.axvspan(bounds_us[1], bounds_us[-2], color=C_MOVE,
                           alpha=0.06, lw=0, zorder=0)
                if gw:
                    ax.axvspan(gw[0] / 1e3, gw[1] / 1e3, color=C_GATE,
                               alpha=0.22, lw=0, zorder=0)
                ax.set_xticklabels([])          # one time axis, at the bottom
        blocks[tag] = rows

    blocks["nsr"][-1].set_xlabel(r"wall-clock time  ($\mu$s)", fontsize=7.6,
                                 color=C_INK, labelpad=2)
    for ax in blocks["nsr"][:-1]:
        ax.set_xticklabels([])

    # ── lane headers + the two numbers the figure exists to show ──
    tr_us = meta["psr_transport_ns"] / 1e3
    wall_us = meta["psr_wall_ns"] / 1e3
    blocks["psr"][0].text(
        0.0, 1.34, "PSR (kick)", transform=blocks["psr"][0].transAxes,
        fontsize=8.0, color=C_PSR, va="bottom", ha="left", weight="bold")
    blocks["psr"][0].text(
        1.0, 1.34,
        f"transport + gate + transport = {tr_us:.1f} " + r"$\mu$s of "
        + f"{wall_us:.1f} " + r"$\mu$s" + f"  ({100 * tr_us / wall_us:.0f}%)",
        transform=blocks["psr"][0].transAxes, fontsize=6.8, color="#8a4b00",
        va="bottom", ha="right")
    blocks["nsr"][0].text(
        0.0, 1.34, "NSR (waveform shift)",
        transform=blocks["nsr"][0].transAxes, fontsize=8.0, color=C_NSR,
        va="bottom", ha="left", weight="bold")
    blocks["nsr"][0].text(
        1.0, 1.34,
        f"same geometry, one segment: {meta['nsr_wall_ns'] / 1e3:.3f} "
        + r"$\mu$s $=T$;  amplitudes $\times$" + f"{meta['scale']:.3f}",
        transform=blocks["nsr"][0].transAxes, fontsize=6.8, color="#0f6b52",
        va="bottom", ha="right")

    # ── carrier-resolved insets ──
    axg = fig.add_subplot(gs[14, 0])
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

    axd = fig.add_subplot(gs[14, 1])
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

    fig.text(0.5, 0.975,
             r"running example  $H(x)=\sin(2x)\,(Z_0Z_1+X_0+X_1)$,  $x=0.7$,"
             r"  $T=5.0\ \mu$s, 2 qubits — as emitted on the six physical "
             r"channels", fontsize=8.0, color=C_INK, ha="center", va="top")
    fig.text(0.5, 0.949,
             f"Nyquist mode $n$={meta['nsr_mode']},  shift $s$="
             f"{meta['s']:.4f}  (bandwidth $K$={meta['K']:.3f});  machine "
             f"residual {meta['residual_source']:.1e} (source) / "
             f"{meta['residual_nsr']:.1e} (shifted)",
             fontsize=6.9, color=C_SEC, ha="center", va="top")

    os.makedirs(FIG_DIR, exist_ok=True)
    for out in (FIG_DIR, OUT2, OUT3):
        os.makedirs(out, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out, f"F_waveform.{ext}"),
                        bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"rendered F_waveform.pdf/.png -> {FIG_DIR}, {OUT2}, {OUT3}")
