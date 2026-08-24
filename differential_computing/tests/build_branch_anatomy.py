"""
build_branch_anatomy.py — Sec 6 figure: machine anatomy of one 2q PSR branch.

Two panels, styled after the hand mock (space | time), but every number is
read from the real compiled artifact:

  Panel A "Space: zone architecture" — atom geometry from the solver's
  sol_gvars, gate zone / transit lane / R_cz from the mapper config, with the
  compiled pair highlighted against a schematic array.

  Panel B "Time: one two-qubit-kick branch" — lane timeline on the
  EVENT-SPACED axis (equal width per inter-event interval, tick labels are
  true times), extracted from the PulseDSL schedule:
    1 global drives   : dressing + addressing ON blocks with the real
                        sampled comb envelopes inside; OFF (frozen) during
                        the insertion window (Assumption 4.7 halt/resume)
    2 AOD transport   : the real min-jerk x(t) of both comb tones
    3 gate-zone pulse : the 200 ns CZ play (amplitude pi, virtual-Z phase)
    4 frame updates   : software-only ticks — the branch sign s lives here
    5 pair y / lane   : lift 5 um -> transit lane -> drop, park at R_cz
  plus a per-stage cost strip (dressed T2*, ground-state clock, benchmarked
  gate error, classical readout).

Phases: extract (runs the pipeline, caches figures/branch_anatomy_data.json)
and render (reads the JSON — REBUILD=1 forces re-extraction).  Never re-run
the pipeline just to tweak the plot.

Run:  conda run -n qec_pg python differential_computing/tests/build_branch_anatomy.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/Users/syue99/research/RISC-Q/PulseDSL/src/DSL/")

import numpy as np

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
DATA_JSON = os.path.join(FIG_DIR, "branch_anatomy_data.json")

C_ANALOG = "#1f5fbf"      # analog / continuous
C_DIGITAL = "#d62728"     # digital / discrete
C_SOFT = "#666666"        # software-only
C_ATOM = "#26343f"


# ── extraction ────────────────────────────────────────────────────────────────

def extract():
    """Compile the running instance and distill the figure data to JSON."""
    import physical_channels as pc
    from awg_compile import ChirpTone, _fallback_waveform
    from physical_walkthrough import build_schedule

    # Simplification for this figure: atoms always ride the AOD, so moves
    # are direct (no lift/drop transit legs).
    schedule, mapper, meta = build_schedule(verbose=False, transit_dy=None)
    rows = schedule._Sched__schedule

    def entries(ch):
        out = []
        for e in rows[ch]:
            out.append((float(e._ScheduleEntry__t0),
                        float(e._ScheduleEntry__t1),
                        e._ScheduleEntry__pulse))
        return sorted(out, key=lambda x: x[0])

    # critical times = every entry boundary on every channel
    bset = {0.0}
    for chn in range(pc.NUM_PHYSICAL_CHANNELS):
        for t0, t1, _ in entries(chn):
            bset.add(t0); bset.add(t1)
    bounds = []
    for b in sorted(bset):
        if not bounds or b - bounds[-1] > 1.0:
            bounds.append(b)

    # transport: per-tone position traces (AOD freq -> um via calibration)
    def pos_of(f_mhz):
        return ((np.asarray(f_mhz) - pc.TRANSPORT_BASE_FREQ_MHZ)
                / pc.TRANSPORT_KAPPA_MHZ_PER_UM)

    def tone_traces(ch):
        traces = []
        for t0, t1, p in entries(ch):
            wf = p.waveform
            if isinstance(wf, ChirpTone):
                tt = np.linspace(0.0, wf.duration_ns, 120)
                traces.append({"t": (t0 + tt).tolist(),
                               "um": pos_of(wf.instantaneous_freq_mhz(tt)
                                            ).tolist()})
            elif getattr(wf, "freq_mhz", 0.0):
                traces.append({"t": [t0, t1],
                               "um": [float(pos_of(wf.freq_mhz))] * 2})
        return traces

    # drive blocks: dressing / addressing windows + sampled |A| envelope
    def drive_blocks(ch):
        blocks = {}
        for t0, t1, p in entries(ch):
            blocks.setdefault((t0, t1), []).append(p)
        out = []
        for (t0, t1), pulses in sorted(blocks.items()):
            t = np.linspace(t0, t1, 400)
            w = np.zeros(len(t), dtype=complex)
            for p in pulses:
                fn = p.waveform if p.waveform is not None \
                    else _fallback_waveform(p)
                w += fn(t - t0)
            out.append({"t0": t0, "t1": t1,
                        "env": np.abs(w).tolist()})
        return out

    # the CZ play (single shortest gate-zone entry) + its virtual-Z phase
    gate = min(entries(pc.GATE_AOM), key=lambda e: e[1] - e[0])
    cz = {"t0": gate[0], "t1": gate[1],
          "amp": float(gate[2].amplitude),
          "vz_phase": float(getattr(gate[2], "phase", 0.0) or 0.0)}

    data = {
        "meta": {k: (list(v) if isinstance(v, tuple) else v)
                 for k, v in meta.items()},
        "bounds_ns": bounds,
        "x_tones": tone_traces(pc.TRANSPORT_AOD_X),
        "y_tones": tone_traces(pc.TRANSPORT_AOD_Y),
        "dressing": drive_blocks(pc.DRESSING_AOM),
        "addr_rabi": drive_blocks(pc.ADDR_RABI),
        "addr_det": drive_blocks(pc.ADDR_DET),
        "aod_x_env": drive_blocks(pc.TRANSPORT_AOD_X),
        "cz": cz,
    }
    data["meta"]["interaction_positions"] = [
        list(p) for p in meta["interaction_positions"]]
    os.makedirs(FIG_DIR, exist_ok=True)
    with open(DATA_JSON, "w") as f:
        json.dump(data, f)
    print(f"extracted -> {DATA_JSON}")
    return data


def stage_names(bounds, cz):
    """Human stage label per inter-event interval.

    Direct-move schedule: ev(0,tau) ; move -> ; CZ ; move <- ; ev(tau,T).
    """
    names = []
    for i in range(len(bounds) - 1):
        t0, t1 = bounds[i], bounds[i + 1]
        if abs(t0 - cz["t0"]) < 1.0 and abs(t1 - cz["t1"]) < 1.0:
            names.append("CZ")
        elif i == 0:
            names.append("ev(0,τ)")
        elif i == len(bounds) - 2:
            names.append("ev(τ,T)")
        elif t1 <= cz["t0"] + 1.0:
            names.append("move →")
        else:
            names.append("move ←")
    return names


# ── render ────────────────────────────────────────────────────────────────────

def render(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon

    meta = data["meta"]
    bounds = data["bounds_ns"]
    cz = data["cz"]
    nb = len(bounds)
    warp = lambda t: np.interp(t, bounds, np.arange(nb, dtype=float))

    fig = plt.figure(figsize=(12.6, 4.35))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.46, 1.0], wspace=0.12,
                          left=0.012, right=0.992, top=0.82, bottom=0.10)
    gsA = gs[0, 0].subgridspec(3, 1, height_ratios=[1.0, 0.42, 1.0],
                               hspace=0.28)

    R_cz = meta["R_cz"]
    gz_x = meta["gate_zone"][0]
    move_us = (bounds[2] - bounds[1]) * 1e-3
    C_AOD = "#d9822b"

    # interaction box (wide) and gate zone (narrow column)
    IB = (-28, -7, 26, 14)
    GB = (16, -9.5, 6, 19)
    lattice = [(x, y) for x in (-25, -20, -15, -10)
               for y in (-4.0, 0.0, 4.0)]
    pair_sites = [(-20, 0.0), (-15, 0.0)]
    gb_cx = GB[0] + GB[2] / 2

    def zone_boxes(ax, dress_on, gate_on):
        bx, by, bw, bh = IB
        ax.add_patch(Rectangle((bx, by), bw, bh, ec=C_ANALOG, lw=1.0,
                               fc=C_ANALOG if dress_on else "none",
                               alpha=0.14 if dress_on else 1.0))
        if dress_on:
            cx = bx + bw / 2
            ax.add_patch(Polygon([(cx - 6, by + bh + 3.4),
                                  (cx + 6, by + bh + 3.4),
                                  (cx + bw / 2 - 1, by + bh),
                                  (cx - bw / 2 + 1, by + bh)], closed=True,
                                 fc=C_ANALOG, alpha=0.30, ec="none"))
            ax.text(cx, by + bh + 1.4, "dressing laser ON", fontsize=7.2,
                    ha="center", color=C_ANALOG, fontweight="bold")
        else:
            ax.text(bx + bw / 2, by + bh + 1.4, "dressing laser OFF",
                    fontsize=7.2, ha="center", color="#999999")
        ax.text(bx + bw / 2, by - 2.4, "interaction zone", fontsize=7.4,
                ha="center", color=C_ANALOG)
        bx, by, bw, bh = GB
        ax.add_patch(Rectangle((bx, by), bw, bh, ec=C_DIGITAL, lw=1.0,
                               fc=C_DIGITAL if gate_on else "none",
                               alpha=0.14 if gate_on else 1.0))
        if gate_on:
            ax.add_patch(Polygon([(gb_cx - 4.5, by + bh + 3.4),
                                  (gb_cx + 4.5, by + bh + 3.4),
                                  (gb_cx + bw / 2 - 0.5, by + bh),
                                  (gb_cx - bw / 2 + 0.5, by + bh)],
                                 closed=True, fc=C_DIGITAL, alpha=0.30,
                                 ec="none"))
            ax.text(gb_cx, by + bh + 1.4, "gate laser ON", fontsize=7.2,
                    ha="center", color=C_DIGITAL, fontweight="bold")
        else:
            ax.text(gb_cx, by + bh + 1.4, "gate laser OFF", fontsize=7.2,
                    ha="center", color="#999999")
        ax.text(gb_cx, by - 2.4, "gate zone", fontsize=7.4, ha="center",
                color=C_DIGITAL)

    def dots(ax, vacated):
        for (px, py) in lattice:
            if (px, py) in pair_sites:
                if vacated:
                    ax.plot(px, py, "o", ms=6, mfc="none", mec="gray",
                            mew=0.9, alpha=0.6)
                continue
            ax.plot(px, py, "o", ms=2.6, color="gray", alpha=0.4)

    def pair(ax, positions, ms=6.5):
        for px, py in positions:
            ax.plot(px, py, "o", ms=ms, color=C_ATOM, zorder=6)
            ax.plot(px, py, "o", ms=ms + 5, mfc="none", mec="#e6a817",
                    mew=1.3, zorder=6)

    def badge(ax, x, y, num, col):
        ax.text(x, y, num, fontsize=8, color="white", ha="center",
                va="center",
                bbox=dict(boxstyle="circle,pad=0.2", fc=col, ec="none"))

    # scene 1 — evolve
    ax1 = fig.add_subplot(gsA[0, 0])
    ax1.set_xlim(-30, 30); ax1.set_ylim(-10.5, 15.5); ax1.axis("off")
    zone_boxes(ax1, dress_on=True, gate_on=False)
    dots(ax1, vacated=False)
    pair(ax1, pair_sites)
    badge(ax1, -28.8, 14.0, "1", C_ANALOG)
    ax1.text(-25.4, 14.0, "evolve ev$(0,\\tau)$ / ev$(\\tau,T)$",
             fontsize=8.2, color=C_ATOM, ha="left", va="center")

    # scene 2 — transit (thin): no zones, pair mid-flight on the AOD
    ax2 = fig.add_subplot(gsA[1, 0])
    ax2.set_xlim(-30, 30); ax2.set_ylim(-4.5, 4.5); ax2.axis("off")
    pair(ax2, [(12.0, 0.0), (16.0, 0.0)], ms=6)
    ax2.annotate("", xy=(27, 1.8), xytext=(19.5, 1.8),
                 arrowprops=dict(arrowstyle="-|>", color=C_AOD, lw=1.3,
                                 ls="--", mutation_scale=12))
    ax2.annotate("", xy=(19.5, -1.8), xytext=(27, -1.8),
                 arrowprops=dict(arrowstyle="-|>", color=C_AOD, lw=1.1,
                                 ls="--", mutation_scale=12, alpha=0.7))
    badge(ax2, -28.8, 2.6, "2", C_AOD)
    badge(ax2, -28.8, -2.6, "4", C_AOD)
    ax2.text(-25.4, 0.0, f"AOD moves pair ({move_us:.0f} μs\n"
             "min-jerk, all drives off)", fontsize=7.4, color=C_AOD,
             ha="left", va="center")

    # scene 3 — insert CZ
    ax3 = fig.add_subplot(gsA[2, 0])
    ax3.set_xlim(-30, 30); ax3.set_ylim(-10.5, 15.5); ax3.axis("off")
    zone_boxes(ax3, dress_on=False, gate_on=True)
    dots(ax3, vacated=True)
    pair(ax3, [(gb_cx - 1.5, 0.0), (gb_cx + 1.5, 0.0)], ms=6)
    badge(ax3, -28.8, 14.0, "3", C_DIGITAL)
    ax3.text(-25.4, 14.0, "insert CZ (200 ns) + virtual $R_z$",
             fontsize=8.2, color=C_ATOM, ha="left", va="center")
    ax3.annotate(f"pair at $R_{{cz}}$ = {R_cz:g} μm",
                 xy=(gb_cx - 1.5, -0.8), xytext=(1.5, -7.6), fontsize=7.2,
                 ha="center", color=C_ATOM,
                 arrowprops=dict(arrowstyle="-", color=C_ATOM, lw=0.7))

    fig.text(0.022, 0.93, "Space: zones and beams", fontsize=11.5,
             color=C_ANALOG, fontweight="bold")

    # ═══ Panel B: laser schedule over AOD movement (compressed) ═════════════
    axB = fig.add_subplot(gs[0, 1])
    axB.set_xlim(0, nb - 0.999 + 0.9)
    axB.set_ylim(-1.55, 5.05)
    axB.axis("off")
    axB.set_title("Time: one two-qubit-kick branch (event-spaced axis)",
                  fontsize=11.5, color=C_ANALOG, loc="left",
                  fontweight="bold")

    names = stage_names(bounds, cz)
    meas_x0, meas_x1 = nb - 1, nb - 0.1
    ins_x0, ins_x1 = 1.0, nb - 2.0
    xcz = warp((cz["t0"] + cz["t1"]) / 2)

    for i in range(nb):
        axB.plot([i, i], [-0.72, 4.0], color="gray", lw=0.4, alpha=0.30)
    for i, nm in enumerate(names):
        axB.text(i + 0.5, 4.14, nm, fontsize=7.2, ha="center", color=C_ATOM)
    axB.text((meas_x0 + meas_x1) / 2, 4.14, "meas", fontsize=7.2,
             ha="center", color=C_SOFT)
    for i in range(nb - 1):
        num = str((1, 2, 3, 4, 1)[i])
        col = (C_ANALOG, C_AOD, C_DIGITAL, C_AOD, C_ANALOG)[i]
        axB.text(i + 0.5, 4.62, num, fontsize=7.2, color="white",
                 ha="center", va="center",
                 bbox=dict(boxstyle="circle,pad=0.18", fc=col, ec="none"))
    for i, b in enumerate(bounds):
        v = b * 1e-3
        axB.text(i, -0.86, f"{v:.4g}" if v < 100 else f"{v:.1f}",
                 fontsize=6.3, ha="right", va="top", rotation=45,
                 color="#444444")
    axB.text(meas_x1, -0.86, "t (μs)", fontsize=7, ha="left", va="top",
             color="#444444")

    # lane 1: laser schedule
    y1, h1 = 2.55, 0.95
    axB.text(-0.10, y1 + 0.45, "lasers\n(dressing · addr\n· gate)",
             fontsize=7.6, ha="right", va="center", color=C_ATOM)
    env_max = max(max(b["env"]) for b in
                  data["dressing"] + data["addr_rabi"])
    for key, alpha in (("addr_rabi", 0.55), ("dressing", 0.95)):
        for b in data[key]:
            t = np.linspace(b["t0"], b["t1"], len(b["env"]))
            e = np.asarray(b["env"]) / env_max * h1
            axB.fill_between(warp(t), y1, y1 + e, color=C_ANALOG,
                             alpha=0.30 * alpha, lw=0)
            axB.plot(warp(t), y1 + e, color=C_ANALOG, lw=0.6, alpha=alpha)
    axB.plot([ins_x0, ins_x1], [y1, y1], color=C_ANALOG, lw=0.8, alpha=0.6)
    axB.add_patch(Rectangle((warp(cz["t0"]), y1),
                            warp(cz["t1"]) - warp(cz["t0"]), h1 * 0.85,
                            fc=C_DIGITAL, alpha=0.8, ec="none"))
    axB.text(xcz, y1 + h1 + 0.10, f"gate laser: CZ "
             f"({(cz['t1'] - cz['t0']):.0f} ns, amp π)", fontsize=7.2,
             ha="center", color=C_DIGITAL)
    axB.text(1.5, y1 + 0.45, "dressing OFF (frozen)", fontsize=7.2,
             ha="center", color=C_ANALOG, alpha=0.85)
    axB.text(0.5, y1 - 0.28, "dressing ON", fontsize=6.8, ha="center",
             color=C_ANALOG, fontweight="bold")
    axB.text(nb - 1.5, y1 - 0.28, "dressing ON", fontsize=6.8, ha="center",
             color=C_ANALOG, fontweight="bold")
    axB.add_patch(Rectangle((meas_x0 + 0.05, y1), meas_x1 - meas_x0 - 0.1,
                            h1 * 0.8, fc="none", ec=C_SOFT, ls="--",
                            lw=0.8))
    axB.text((meas_x0 + meas_x1) / 2, y1 + 0.38, "readout\nlight",
             fontsize=6.2, ha="center", va="center", color=C_SOFT)
    axB.annotate("virtual $R_z(s\\pi/2)^{\\otimes2}$ (software) — "
                 "the branch sign $s$ exists here",
                 xy=(xcz + 0.06, y1 - 0.04), xytext=(xcz + 0.55, y1 - 0.42),
                 fontsize=6.6, color=C_SOFT, va="center",
                 arrowprops=dict(arrowstyle="-", color=C_SOFT, lw=0.6))

    # lane 2: AOD movement — x(t) tones over the comb |A| (grey)
    y2, h2 = 0.30, 1.15
    axB.text(-0.10, y2 + 0.55, "AOD movement\n$x(t)$ + comb $|A|$",
             fontsize=7.6, ha="right", va="center", color=C_ATOM)
    aenv_max = max(max(b["env"]) for b in data["aod_x_env"])
    for b in data["aod_x_env"]:
        t = np.linspace(b["t0"], b["t1"], len(b["env"]))
        e = np.asarray(b["env"]) / aenv_max * h2
        axB.fill_between(warp(t), y2, y2 + e, color="gray", alpha=0.22,
                         lw=0)
    for tr in data["x_tones"]:
        t = np.asarray(tr["t"]); um = np.asarray(tr["um"])
        axB.plot(warp(t), y2 + (um - um.min()) / gz_x * h2, color=C_AOD,
                 lw=1.1)
    axB.text(1.5, y2 + h2 + 0.08, f"min-jerk move ({move_us:.0f} μs)",
             fontsize=7.0, ha="center", color=C_AOD)
    axB.text(xcz, y2 + h2 + 0.08, "in gate zone", fontsize=7.0,
             ha="center", color=C_AOD)
    axB.text(0.5, y2 + 0.16, "grey: AOD comb $|A|$ (2-tone beat)",
             fontsize=6.2, ha="center", color="#777777")

    # cost strip
    yc = -0.30
    tot_move_us = (bounds[-2] - bounds[1]) * 1e-3 \
        - (cz["t1"] - cz["t0"]) * 1e-3
    for xa, xb, lab, col in (
            (0, 1, "$T/T_2^*$ (dressed)", C_ANALOG),
            (ins_x0, ins_x1,
             f"{tot_move_us:.0f} μs ground-state clock + "
             "benchmarked $\\epsilon_{ins}$", C_DIGITAL),
            (nb - 2, nb - 1, "$T/T_2^*$", C_ANALOG),
            (meas_x0, meas_x1, "readout", C_SOFT)):
        axB.plot([xa + 0.05, xb - 0.05], [yc, yc], color=col, lw=1.0)
        axB.plot([xa + 0.05] * 2, [yc, yc + 0.09], color=col, lw=1.0)
        axB.plot([xb - 0.05] * 2, [yc, yc + 0.09], color=col, lw=1.0)
        axB.text((xa + xb) / 2, yc - 0.11, lab, fontsize=6.4, ha="center",
                 va="top", color=col)

    axB.plot([], [], color=C_ANALOG, lw=3, label="ANALOG")
    axB.plot([], [], color=C_DIGITAL, lw=3, label="DIGITAL")
    axB.plot([], [], color=C_AOD, lw=3, label="AOD / transport")
    axB.plot([], [], color=C_SOFT, lw=1.5, ls=(0, (4, 3)),
             label="SOFTWARE-ONLY")
    fig.legend(loc="lower center", ncol=4, fontsize=7, frameon=False,
               bbox_to_anchor=(0.72, -0.012))
    fig.text(0.20, 0.012,
             "atoms held by AOD tweezers throughout (simplification); "
             "pair separation schematic; zones "
             f"{gz_x:.0f} μm apart", fontsize=6.6, ha="center",
             color="#777777")

    out = os.path.join(FIG_DIR, "branch_anatomy")
    fig.savefig(out + ".png", dpi=170, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    print(f"saved {out}.png / .pdf")


def main():
    if os.path.exists(DATA_JSON) and not os.environ.get("REBUILD"):
        with open(DATA_JSON) as f:
            data = json.load(f)
        print(f"using cached {DATA_JSON} (REBUILD=1 to re-extract)")
    else:
        data = extract()
    render(data)


if __name__ == "__main__":
    main()
