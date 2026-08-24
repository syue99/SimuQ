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

    schedule, mapper, meta = build_schedule(verbose=False)
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
    """Human stage label per inter-event interval, classified by time."""
    names = []
    travel_seen = 0
    for i in range(len(bounds) - 1):
        t0, t1 = bounds[i], bounds[i + 1]
        dur = t1 - t0
        if abs(t0 - cz["t0"]) < 1.0 and abs(t1 - cz["t1"]) < 1.0:
            names.append("CZ")
        elif i == 0:
            names.append("ev(0,τ)")
        elif i == len(bounds) - 2:
            names.append("ev(τ,T)")
        elif dur > 10000:
            travel_seen += 1
            names.append("move →" if travel_seen == 1 else "move ←")
        elif t1 <= cz["t0"] + 1.0:
            names.append("lift" if travel_seen == 0 else "drop")
        else:
            names.append("lift" if travel_seen == 1 else "drop")
        # lift/drop disambiguation: first pre-travel small leg = lift,
        # post-travel = drop (per direction)
    # fix the small legs around each travel explicitly
    for i in range(len(names)):
        if names[i] in ("lift", "drop"):
            prev_travel = any(n.startswith("move") for n in names[:i])
            next_travel = any(n.startswith("move") for n in names[i + 1:])
            if not prev_travel:
                names[i] = "lift"
            elif prev_travel and next_travel and names[i + 1].startswith("move"):
                names[i] = "lift"
            elif not next_travel and i > 0 and names[i - 1] == "ev(τ,T)":
                names[i] = "drop"
            elif i > 0 and names[i - 1].startswith("move"):
                names[i] = "drop"
    return names


# ── render ────────────────────────────────────────────────────────────────────

def render(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch

    meta = data["meta"]
    bounds = data["bounds_ns"]
    cz = data["cz"]
    nb = len(bounds)
    warp = lambda t: np.interp(t, bounds, np.arange(nb, dtype=float))

    fig = plt.figure(figsize=(13.2, 5.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.42, 1.0], wspace=0.14,
                          left=0.012, right=0.99, top=0.86, bottom=0.075)
    gsA = gs[0, 0].subgridspec(2, 1, hspace=0.42)

    R_cz = meta["R_cz"]
    dy = meta["transit_dy"]
    gz_x = meta["gate_zone"][0]
    pair_sep = float(np.hypot(
        meta["interaction_positions"][1][0]
        - meta["interaction_positions"][0][0],
        meta["interaction_positions"][1][1]
        - meta["interaction_positions"][0][1]))

    def badge(ax, num, text, col, y=18.6):
        ax.text(-25.5, y, num, fontsize=8.5, color="white", ha="center",
                va="center", zorder=6,
                bbox=dict(boxstyle="circle,pad=0.22", fc=col, ec="none"))
        ax.text(-21.5, y, text, fontsize=8.6, color=C_ATOM, ha="left",
                va="center")

    def atom_pair(ax, positions, ms=7.5):
        for px, py in positions:
            ax.plot(px, py, "o", ms=ms, color=C_ATOM, zorder=6)
            ax.plot(px, py, "o", ms=ms + 5.5, mfc="none", mec="#e6a817",
                    mew=1.4, zorder=6)

    # ── scene 1: interaction zone ────────────────────────────────────────────
    ax1 = fig.add_subplot(gsA[0, 0])
    ax1.set_xlim(-28, 28); ax1.set_ylim(-17.5, 21); ax1.axis("off")
    # tidy 10 um lattice, compiled pair on adjacent sites (footnote below)
    for xg in (-20, -10, 0, 10, 20):
        for yg in (-9, 0, 9):
            ax1.plot(xg, yg, "o", ms=3, color="gray", alpha=0.35, zorder=3)
    # global dressing: broad sheet of light from the top
    ax1.add_patch(Polygon([(-26, 21), (26, 21), (22, -11), (-22, -11)],
                          closed=True, fc=C_ANALOG, alpha=0.10, ec="none"))
    ax1.text(0, 14.6, "global dressing beam (ZZ)", fontsize=7.4,
             ha="center", color=C_ANALOG)
    # addressing AOD: projector box + focused cones onto the pair
    ax1.add_patch(Rectangle((-7, -17), 14, 3.2, fc="#dfe7f2", ec=C_ANALOG,
                            lw=0.8))
    ax1.text(0, -15.4, "addr AOD", fontsize=7, ha="center", color=C_ANALOG)
    for px in (-10, 0):
        ax1.add_patch(Polygon([(-2.0, -13.8), (2.0, -13.8), (px + 0.9, -1.0),
                               (px - 0.9, -1.0)], closed=True, fc=C_ANALOG,
                              alpha=0.22, ec="none"))
    ax1.text(11.5, -7.5, "addressing\ncombs (X)", fontsize=7.4,
             ha="center", color=C_ANALOG)
    atom_pair(ax1, [(-10, 0), (0, 0)])
    badge(ax1, "1", "ev$(0,\\tau)$ / ev$(\\tau,T)$ — analog evolution",
          C_ANALOG)
    ax1.text(0, -20.6, "interaction zone", fontsize=8, ha="center",
             color=C_ANALOG)

    # ── scene 2: gate zone ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gsA[1, 0])
    ax2.set_xlim(-28, 28); ax2.set_ylim(-17.5, 21); ax2.axis("off")
    ax2.add_patch(Rectangle((-11, -11), 22, 26, fc=C_DIGITAL, alpha=0.06,
                            ec="none"))
    # focused CZ beam from the top
    ax2.add_patch(Polygon([(-4.2, 21), (4.2, 21), (1.7, 1.3), (-1.7, 1.3)],
                          closed=True, fc=C_DIGITAL, alpha=0.28, ec="none"))
    ax2.text(13.5, 8.0, "focused\nCZ beam", fontsize=7.4, ha="center",
             color=C_DIGITAL)
    # transport AOD tweezers holding the pair
    ax2.add_patch(Rectangle((-7, -17), 14, 3.2, fc="#f7e9d9", ec="#d9822b",
                            lw=0.8))
    ax2.text(0, -15.4, "transport AOD", fontsize=7, ha="center",
             color="#d9822b")
    for px in (-R_cz / 2, R_cz / 2):
        ax2.add_patch(Polygon([(-1.6, -13.8), (1.6, -13.8), (px + 0.7, -0.9),
                               (px - 0.7, -0.9)], closed=True, fc="#d9822b",
                              alpha=0.30, ec="none"))
    ax2.text(11.5, -8.0, "moving\ntweezers", fontsize=7.4, ha="center",
             color="#d9822b")
    atom_pair(ax2, [(-R_cz / 2, 0.0), (R_cz / 2, 0.0)], ms=7)
    ax2.annotate(f"$R_{{cz}}$ = {R_cz:g} μm", xy=(0, -1.6),
                 xytext=(-17, -6.5), fontsize=7.4, color=C_ATOM,
                 arrowprops=dict(arrowstyle="-", color=C_ATOM, lw=0.7))
    badge(ax2, "3", "CZ (200 ns) + virtual $R_z$", C_DIGITAL)
    ax2.text(0, -20.6, "gate zone", fontsize=8, ha="center", color=C_DIGITAL)

    # ── step arrows between the scenes ───────────────────────────────────────
    from matplotlib.patches import ConnectionPatch
    move_lab = (f"2  pickup, $+{dy:g}$ μm lane,\n"
                f"    {gz_x:.0f} μm min-jerk move (≈52 μs)")
    cp = ConnectionPatch(xyA=(-19, -12.5), coordsA=ax1.transData,
                         xyB=(-19, 19.5), coordsB=ax2.transData,
                         arrowstyle="-|>", mutation_scale=13,
                         color=C_DIGITAL, ls="--", lw=1.3)
    fig.add_artist(cp)
    ax2.text(-27.6, 26.5, move_lab, fontsize=7.6, color=C_DIGITAL,
             ha="left", va="center")
    cp2 = ConnectionPatch(xyA=(21, 19.5), coordsA=ax2.transData,
                          xyB=(21, -12.5), coordsB=ax1.transData,
                          arrowstyle="-|>", mutation_scale=13,
                          color=C_DIGITAL, ls="--", lw=1.1, alpha=0.65)
    fig.add_artist(cp2)
    ax2.text(27.6, 26.5, "4  return\n    (drives resume)", fontsize=7.6,
             color=C_DIGITAL, ha="right", va="center", alpha=0.85)

    fig.text(0.022, 0.93, "Space: zones and beams", fontsize=12,
             color=C_ANALOG, fontweight="bold")
    fig.text(0.185, 0.012,
             "compiled pair $n{=}2$ on a schematic lattice "
             f"(solver pair separation {pair_sep:.1f} μm); $y$ not to scale",
             fontsize=7.2, ha="center", color="#777777")

    # ═══ Panel B: two lanes — lasers and AOD movement ════════════════════════
    axB = fig.add_subplot(gs[0, 1])
    axB.set_xlim(0, nb - 0.999 + 0.9)
    axB.set_ylim(-1.85, 5.9)
    axB.axis("off")
    axB.set_title("Time: one two-qubit-kick branch (event-spaced axis)",
                  fontsize=12, color=C_ANALOG, loc="left",
                  fontweight="bold")

    names = stage_names(bounds, cz)
    meas_x0, meas_x1 = nb - 1, nb - 0.1
    ins_x0, ins_x1 = 1.0, nb - 2.0
    xcz = warp((cz["t0"] + cz["t1"]) / 2)

    for i in range(nb):
        axB.plot([i, i], [-0.9, 4.75], color="gray", lw=0.4, alpha=0.30)
    for i, nm in enumerate(names):
        axB.text(i + 0.5, 4.92, nm, fontsize=7.2, ha="center", color=C_ATOM)
    axB.text((meas_x0 + meas_x1) / 2, 4.92, "meas", fontsize=7.2,
             ha="center", color=C_SOFT)
    for xm, num, col in ((0.5, "1", C_ANALOG), (2.5, "2", C_DIGITAL),
                         (xcz, "3", C_DIGITAL), (6.5, "4", C_DIGITAL),
                         (nb - 1.5, "1", C_ANALOG)):
        axB.text(xm, 5.45, num, fontsize=7.5, color="white", ha="center",
                 va="center",
                 bbox=dict(boxstyle="circle,pad=0.2", fc=col, ec="none"))
    for i, b in enumerate(bounds):
        v = b * 1e-3
        axB.text(i, -1.05, f"{v:.4g}" if v < 100 else f"{v:.1f}",
                 fontsize=6.3, ha="right", va="top", rotation=45,
                 color="#444444")
    axB.text(meas_x1, -1.05, "t (μs)", fontsize=7, ha="left", va="top",
             color="#444444")

    # ── lane 1: laser schedule (dressing + addressing + CZ + readout) ────────
    y1, h1 = 3.1, 1.1
    axB.text(-0.12, y1 + 0.5, "lasers\n(dressing · addr\n· CZ)",
             fontsize=8, ha="right", va="center", color=C_ATOM)
    env_max = max(max(b["env"]) for b in
                  data["dressing"] + data["addr_rabi"])
    for key, alpha in (("addr_rabi", 0.55), ("dressing", 0.95)):
        for b in data[key]:
            t = np.linspace(b["t0"], b["t1"], len(b["env"]))
            e = np.asarray(b["env"]) / env_max * h1
            axB.fill_between(warp(t), y1, y1 + e, color=C_ANALOG,
                             alpha=0.30 * alpha, lw=0)
            axB.plot(warp(t), y1 + e, color=C_ANALOG, lw=0.7, alpha=alpha)
    axB.plot([ins_x0, ins_x1], [y1, y1], color=C_ANALOG, lw=0.9, alpha=0.6)
    axB.add_patch(Rectangle((warp(cz["t0"]), y1),
                            warp(cz["t1"]) - warp(cz["t0"]), h1 * 0.85,
                            fc=C_DIGITAL, alpha=0.8, ec="none"))
    axB.text(xcz, y1 + h1 + 0.13, f"CZ ({(cz['t1'] - cz['t0']):.0f} ns, "
             "amp π)", fontsize=7.5, ha="center", color=C_DIGITAL)
    axB.text((ins_x0 + xcz - 0.5) / 2 - 0.4, y1 + 0.5, "OFF (frozen)",
             fontsize=8, ha="center", color=C_ANALOG, alpha=0.85)
    axB.text(0.5, y1 - 0.33, "ON", fontsize=7.5, ha="center",
             color=C_ANALOG, fontweight="bold")
    axB.text(nb - 1.5, y1 - 0.33, "ON", fontsize=7.5, ha="center",
             color=C_ANALOG, fontweight="bold")
    axB.add_patch(Rectangle((meas_x0 + 0.05, y1), meas_x1 - meas_x0 - 0.1,
                            h1 * 0.8, fc="none", ec=C_SOFT, ls="--",
                            lw=0.9))
    axB.text((meas_x0 + meas_x1) / 2, y1 + 0.44, "readout\nlight",
             fontsize=6.6, ha="center", va="center", color=C_SOFT)
    axB.annotate("virtual $R_z(s\\pi/2)^{\\otimes2}$ — software only;\n"
                 "the branch sign $s$ exists here",
                 xy=(xcz + 0.06, y1 - 0.05), xytext=(xcz + 1.35, y1 - 0.52),
                 fontsize=7, color=C_SOFT, va="center",
                 arrowprops=dict(arrowstyle="-", color=C_SOFT, lw=0.7))

    # ── lane 2: AOD movement ────────────────────────────────────────────────
    y2, h2 = 0.55, 1.35
    axB.text(-0.12, y2 + 0.65, "AOD\nmovement $x(t)$", fontsize=8,
             ha="right", va="center", color=C_ATOM)
    for tr in data["x_tones"]:
        t = np.asarray(tr["t"]); um = np.asarray(tr["um"])
        axB.plot(warp(t), y2 + (um - um.min()) / gz_x * h2, color="#d9822b",
                 lw=1.2)
    axB.text(0.5, y2 + 0.22, "idle", fontsize=7.2, ha="center",
             color="#d9822b", alpha=0.85)
    axB.text(warp(bounds[2] + 0.45 * (bounds[3] - bounds[2])), y2 + h2 + 0.1,
             f"min-jerk move ({(bounds[3] - bounds[2]) * 1e-3:.0f} μs), "
             f"$+{dy:g}$ μm transit lane", fontsize=7.3, ha="center",
             color="#d9822b")
    axB.text(xcz, y2 + 0.16, "park at $R_{cz}$", fontsize=7.3,
             ha="center", color="#d9822b")

    # ── cost strip ──────────────────────────────────────────────────────────
    yc = -0.45
    tot_move_us = (bounds[-2] - bounds[1]) * 1e-3 \
        - (cz["t1"] - cz["t0"]) * 1e-3
    for xa, xb, lab, col in (
            (0, 1, "$T/T_2^*$ (dressed)", C_ANALOG),
            (ins_x0, ins_x1,
             f"{tot_move_us:.0f} μs ground-state clock + "
             "benchmarked $\\epsilon_{ins}$", C_DIGITAL),
            (nb - 2, nb - 1, "$T/T_2^*$", C_ANALOG),
            (meas_x0, meas_x1, "readout", C_SOFT)):
        axB.plot([xa + 0.05, xb - 0.05], [yc, yc], color=col, lw=1.1)
        axB.plot([xa + 0.05] * 2, [yc, yc + 0.1], color=col, lw=1.1)
        axB.plot([xb - 0.05] * 2, [yc, yc + 0.1], color=col, lw=1.1)
        axB.text((xa + xb) / 2, yc - 0.13, lab, fontsize=6.6, ha="center",
                 va="top", color=col)

    axB.plot([], [], color=C_ANALOG, lw=3, label="ANALOG (continuous)")
    axB.plot([], [], color=C_DIGITAL, lw=3, label="DIGITAL (discrete)")
    axB.plot([], [], color="#d9822b", lw=3, label="AOD / transport")
    axB.plot([], [], color=C_SOFT, lw=1.5, ls=(0, (4, 3)),
             label="SOFTWARE-ONLY")
    fig.legend(loc="lower center", ncol=4, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.70, -0.008))

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
