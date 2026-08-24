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
    from matplotlib.patches import Rectangle, Polygon, ConnectionPatch

    meta = data["meta"]
    bounds = data["bounds_ns"]
    cz = data["cz"]
    nb = len(bounds)
    warp = lambda t: np.interp(t, bounds, np.arange(nb, dtype=float))

    fig = plt.figure(figsize=(12.8, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.44, 1.0], wspace=0.13,
                          left=0.012, right=0.99, top=0.85, bottom=0.08)
    gsA = gs[0, 0].subgridspec(2, 1, hspace=0.34)

    R_cz = meta["R_cz"]
    gz_x = meta["gate_zone"][0]
    pair_sep = float(np.hypot(
        meta["interaction_positions"][1][0]
        - meta["interaction_positions"][0][0],
        meta["interaction_positions"][1][1]
        - meta["interaction_positions"][0][1]))
    C_AOD = "#d9822b"

    # ── one scene = BOTH zone boxes with their beam states ───────────────────
    IB = (-28, -8, 23, 16)     # interaction box (x, y, w, h)
    GB = (5, -8, 23, 16)       # gate box
    lattice = [(x, y) for x in (-25, -20, -15, -10)
               for y in (-4.5, 0.0, 4.5)]
    pair_sites = [(-20, 0.0), (-15, 0.0)]

    def zone_boxes(ax, dress_on, gate_on):
        for (bx, by, bw, bh), on, col, name, lab in (
                (IB, dress_on, C_ANALOG, "interaction zone",
                 "dressing laser"),
                (GB, gate_on, C_DIGITAL, "gate zone", "gate laser")):
            ax.add_patch(Rectangle((bx, by), bw, bh, ec=col, lw=1.0,
                                   fc=col if on else "none",
                                   alpha=0.14 if on else 1.0))
            if on:   # beam of light from above
                cx = bx + bw / 2
                ax.add_patch(Polygon([(cx - 6, by + bh + 3.9),
                                      (cx + 6, by + bh + 3.9),
                                      (cx + bw / 2 - 1, by + bh),
                                      (cx - bw / 2 + 1, by + bh)],
                                     closed=True, fc=col, alpha=0.30,
                                     ec="none"))
                ax.text(cx, by + bh + 1.7, f"{lab} ON", fontsize=7.6,
                        ha="center", color=col, fontweight="bold")
            else:
                ax.text(bx + bw / 2, by + bh + 1.7, f"{lab} OFF",
                        fontsize=7.6, ha="center", color="#999999")
            ax.text(bx + bw / 2, by - 2.6, name, fontsize=7.6, ha="center",
                    color=col)

    def atoms(ax, pair_in_gate):
        for (px, py) in lattice:
            if (px, py) in pair_sites and pair_in_gate:
                ax.plot(px, py, "o", ms=6.5, mfc="none", mec="gray",
                        mew=0.9, alpha=0.6)      # vacated sites
            elif (px, py) in pair_sites:
                continue
            else:
                ax.plot(px, py, "o", ms=2.8, color="gray", alpha=0.4)
        pos = ([(gz_x_draw - 2.0, 0.0), (gz_x_draw + 2.0, 0.0)]
               if pair_in_gate else pair_sites)
        for px, py in pos:
            ax.plot(px, py, "o", ms=7, color=C_ATOM, zorder=6)
            ax.plot(px, py, "o", ms=12.5, mfc="none", mec="#e6a817",
                    mew=1.4, zorder=6)

    gz_x_draw = GB[0] + GB[2] / 2

    def badge(ax, num, title, sub, col):
        ax.text(-28.5, 15.6, num, fontsize=8.5, color="white", ha="center",
                va="center",
                bbox=dict(boxstyle="circle,pad=0.22", fc=col, ec="none"))
        ax.text(-24.8, 15.6, title, fontsize=8.8, color=C_ATOM, ha="left",
                va="center")
        ax.text(-24.8, 12.4, sub, fontsize=7.6, color=col, ha="left",
                va="center", style="italic")

    ax1 = fig.add_subplot(gsA[0, 0])
    ax1.set_xlim(-30, 30); ax1.set_ylim(-12.5, 17.5); ax1.axis("off")
    zone_boxes(ax1, dress_on=True, gate_on=False)
    atoms(ax1, pair_in_gate=False)
    badge(ax1, "1", "evolve  ev$(0,\\tau)$ / ev$(\\tau,T)$",
          "dressing ON · gate laser OFF", C_ANALOG)

    ax2 = fig.add_subplot(gsA[1, 0])
    ax2.set_xlim(-30, 30); ax2.set_ylim(-12.5, 17.5); ax2.axis("off")
    zone_boxes(ax2, dress_on=False, gate_on=True)
    atoms(ax2, pair_in_gate=True)
    badge(ax2, "3", "insert  CZ (200 ns) + virtual $R_z$",
          "dressing OFF · gate laser ON", C_DIGITAL)
    ax2.text(gz_x_draw, 4.0, f"pair at $R_{{cz}}$ = {R_cz:g} μm",
             fontsize=7.2, color=C_ATOM, ha="center")
    # AOD carries the pair between the boxes (steps 2 and 4)
    ax2.annotate("", xy=(GB[0] + 3, 3.8), xytext=(IB[0] + IB[2] - 3, 3.8),
                 arrowprops=dict(arrowstyle="-|>", color=C_AOD, lw=1.4,
                                 ls="--", mutation_scale=13))
    ax2.annotate("", xy=(IB[0] + IB[2] - 3, -3.8), xytext=(GB[0] + 3, -3.8),
                 arrowprops=dict(arrowstyle="-|>", color=C_AOD, lw=1.2,
                                 ls="--", mutation_scale=13, alpha=0.7))
    move_us = (bounds[2] - bounds[1]) * 1e-3
    ax2.text(0, 6.0, "2", fontsize=7.5, color="white", ha="center",
             va="center",
             bbox=dict(boxstyle="circle,pad=0.2", fc=C_AOD, ec="none"))
    ax2.text(0, -6.0, "4", fontsize=7.5, color="white", ha="center",
             va="center",
             bbox=dict(boxstyle="circle,pad=0.2", fc=C_AOD, ec="none"))
    ax2.text(0, 0.6, f"AOD ({move_us:.0f} μs\nmin-jerk)", fontsize=7,
             ha="center", color=C_AOD)

    fig.text(0.022, 0.925, "Space: zones and beams", fontsize=12,
             color=C_ANALOG, fontweight="bold")
    fig.text(0.185, 0.012,
             "atoms held by AOD tweezers throughout (simplification); "
             f"pair separation schematic (solver: {pair_sep:.1f} μm); "
             f"zones {gz_x:.0f} μm apart", fontsize=7, ha="center",
             color="#777777")

    # ═══ Panel B: two lanes — laser schedule and AOD movement ═══════════════
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
        axB.text(i + 0.5, 4.92, nm, fontsize=7.4, ha="center", color=C_ATOM)
    axB.text((meas_x0 + meas_x1) / 2, 4.92, "meas", fontsize=7.4,
             ha="center", color=C_SOFT)
    for i in range(nb - 1):
        num = str((1, 2, 3, 4, 1)[i]) if nb == 6 else ""
        col = (C_ANALOG, C_AOD, C_DIGITAL, C_AOD, C_ANALOG)[i] \
            if nb == 6 else C_ATOM
        if num:
            axB.text(i + 0.5, 5.45, num, fontsize=7.5, color="white",
                     ha="center", va="center",
                     bbox=dict(boxstyle="circle,pad=0.2", fc=col,
                               ec="none"))
    for i, b in enumerate(bounds):
        v = b * 1e-3
        axB.text(i, -1.05, f"{v:.4g}" if v < 100 else f"{v:.1f}",
                 fontsize=6.5, ha="right", va="top", rotation=45,
                 color="#444444")
    axB.text(meas_x1, -1.05, "t (μs)", fontsize=7, ha="left", va="top",
             color="#444444")

    # ── lane 1: laser schedule ──────────────────────────────────────────────
    y1, h1 = 3.1, 1.1
    axB.text(-0.10, y1 + 0.5, "lasers\n(dressing · addr\n· gate)",
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
    axB.text(xcz, y1 + h1 + 0.13, f"gate laser: CZ "
             f"({(cz['t1'] - cz['t0']):.0f} ns, amp π)", fontsize=7.5,
             ha="center", color=C_DIGITAL)
    axB.text(1.5, y1 + 0.5, "dressing OFF (frozen)", fontsize=7.6,
             ha="center", color=C_ANALOG, alpha=0.85)
    axB.text(0.5, y1 - 0.33, "dressing ON", fontsize=7.2, ha="center",
             color=C_ANALOG, fontweight="bold")
    axB.text(nb - 1.5, y1 - 0.33, "dressing ON", fontsize=7.2, ha="center",
             color=C_ANALOG, fontweight="bold")
    axB.add_patch(Rectangle((meas_x0 + 0.05, y1), meas_x1 - meas_x0 - 0.1,
                            h1 * 0.8, fc="none", ec=C_SOFT, ls="--",
                            lw=0.9))
    axB.text((meas_x0 + meas_x1) / 2, y1 + 0.44, "readout\nlight",
             fontsize=6.6, ha="center", va="center", color=C_SOFT)
    axB.annotate("virtual $R_z(s\\pi/2)^{\\otimes2}$ — software only;\n"
                 "the branch sign $s$ exists here",
                 xy=(xcz + 0.06, y1 - 0.05), xytext=(xcz + 0.75, y1 - 0.62),
                 fontsize=7, color=C_SOFT, va="center",
                 arrowprops=dict(arrowstyle="-", color=C_SOFT, lw=0.7))

    # ── lane 2: AOD movement ────────────────────────────────────────────────
    y2, h2 = 0.55, 1.35
    axB.text(-0.10, y2 + 0.65, "AOD\nmovement $x(t)$", fontsize=8,
             ha="right", va="center", color=C_ATOM)
    for tr in data["x_tones"]:
        t = np.asarray(tr["t"]); um = np.asarray(tr["um"])
        axB.plot(warp(t), y2 + (um - um.min()) / gz_x * h2, color=C_AOD,
                 lw=1.2)
    axB.text(0.5, y2 + 0.22, "in interaction zone", fontsize=7.2,
             ha="center", color=C_AOD, alpha=0.85)
    axB.text(1.5, y2 + h2 + 0.1,
             f"min-jerk move ({move_us:.0f} μs)", fontsize=7.3,
             ha="center", color=C_AOD)
    axB.text(xcz, y2 + h2 + 0.1, "in gate zone", fontsize=7.3, ha="center",
             color=C_AOD)

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
    axB.plot([], [], color=C_AOD, lw=3, label="AOD / transport")
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
