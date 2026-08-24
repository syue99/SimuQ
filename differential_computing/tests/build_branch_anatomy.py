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
    from matplotlib.patches import Rectangle, FancyArrowPatch, Circle

    meta = data["meta"]
    bounds = data["bounds_ns"]
    cz = data["cz"]
    nb = len(bounds)
    warp = lambda t: np.interp(t, bounds, np.arange(nb, dtype=float))

    fig = plt.figure(figsize=(15.0, 8.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.62, 1.0], wspace=0.16,
                          left=0.012, right=0.99, top=0.90, bottom=0.06)
    gsA = gs[0, 0].subgridspec(4, 1, hspace=0.10)

    # ═══ Panel A: space, step by step ════════════════════════════════════════
    gz_x, _gz_y = meta["gate_zone"]
    dy = meta["transit_dy"]
    R_cz = meta["R_cz"]
    atoms = meta["interaction_positions"]
    x0 = float(np.mean([a[0] for a in atoms]))
    y0 = float(np.mean([a[1] for a in atoms]))
    lane_y = y0 + dy

    def scene(idx, blue_on, red_on):
        ax = fig.add_subplot(gsA[idx, 0])
        ax.set_xlim(-26, 116); ax.set_ylim(-13.5, 15)
        ax.axis("off")
        ax.add_patch(Rectangle((-21, -9.5), 42, 21, ec="none", fc=C_ANALOG,
                               alpha=0.14 if blue_on else 0.045))
        ax.add_patch(Rectangle((gz_x - 6, -9.5), 12, 21, ec="none",
                               fc=C_DIGITAL,
                               alpha=0.16 if red_on else 0.045))
        for xg in np.arange(-17, 18, 5.0):
            for yg in np.arange(-7.5, 10.5, 4.5):
                ax.plot(xg, yg, "o", ms=2.4, color="gray", alpha=0.30)
        return ax

    def pair(ax, positions, ms=7):
        for px, py in positions:
            ax.plot(px, py, "o", ms=ms, color=C_ATOM, zorder=5)
            ax.plot(px, py, "o", ms=ms + 5.5, mfc="none", mec="#e6a817",
                    mew=1.4, zorder=5)

    def badge(ax, num, text, col):
        ax.text(-24, 13.2, num, fontsize=9, color="white", ha="center",
                va="center", zorder=6,
                bbox=dict(boxstyle="circle,pad=0.25", fc=col, ec="none"))
        ax.text(-19.5, 13.2, text, fontsize=8.6, color=C_ATOM, ha="left",
                va="center")

    # step 1 — evolve: dressing + combs ON, pair in the array
    ax1 = scene(0, blue_on=True, red_on=False)
    badge(ax1, "1", "ev$(0,\\tau)$ — dressing + addressing ON", C_ANALOG)
    pair(ax1, atoms)
    ax1.text(0, -12.2, "interaction zone (global analog beams)",
             fontsize=7.3, ha="center", color=C_ANALOG)
    ax1.text(gz_x, -12.2, "gate zone (idle)", fontsize=7.3, ha="center",
             color=C_DIGITAL, alpha=0.7)

    # step 2 — drives frozen: pickup, +5 um lift, min-jerk move
    ax2 = scene(1, blue_on=False, red_on=False)
    badge(ax2, "2", "drives OFF — pickup, lift, move", C_DIGITAL)
    for (px, py) in atoms:
        ax2.plot(px, py, "o", ms=7, mfc="none", mec=C_ATOM, mew=0.9,
                 alpha=0.5)
    ax2.plot([x0, x0], [y0, lane_y], ls="--", color=C_DIGITAL, lw=1.1)
    ax2.plot([x0, gz_x], [lane_y, lane_y], ls="--", color=C_DIGITAL,
             lw=1.1)
    ax2.plot([gz_x, gz_x], [lane_y, 0.0], ls=":", color=C_DIGITAL, lw=1.0,
             alpha=0.7)
    pair(ax2, [(46, lane_y), (48.4, lane_y)], ms=6)
    ax2.add_patch(FancyArrowPatch((54, lane_y), (70, lane_y),
                                  arrowstyle="-|>", mutation_scale=12,
                                  color=C_DIGITAL, lw=0))
    ax2.annotate("", xy=(x0, lane_y - 0.4), xytext=(x0, y0 + 0.4),
                 arrowprops=dict(arrowstyle="-|>", color=C_DIGITAL, lw=1.0))
    ax2.text(x0 - 2.5, (y0 + lane_y) / 2, f"+{dy:g} μm", fontsize=7,
             ha="right", color=C_DIGITAL)
    ax2.annotate("", xy=(gz_x, -11.6), xytext=(x0, -11.6),
                 arrowprops=dict(arrowstyle="<|-|>", color=C_ATOM, lw=0.9))
    ax2.text((x0 + gz_x) / 2, -10.4,
             f"d = {gz_x:.0f} μm — min-jerk, "
             f"{meta['v_max']:g} m/s, ≈52 μs", fontsize=7.3, ha="center",
             color=C_ATOM)

    # step 3 — CZ in the gate zone
    ax3 = scene(2, blue_on=False, red_on=True)
    badge(ax3, "3", "CZ in gate zone (200 ns) + virtual $R_z$",
          C_DIGITAL)
    for (px, py) in atoms:
        ax3.plot(px, py, "o", ms=7, mfc="none", mec=C_ATOM, mew=0.9,
                 alpha=0.5)
    pair(ax3, [(gz_x - R_cz / 2, 0.0), (gz_x + R_cz / 2, 0.0)], ms=6)
    ax3.annotate(f"pair at $R_{{cz}}$ = {R_cz:g} μm",
                 xy=(gz_x - 1, -1.6), xytext=(gz_x - 42, -6.5), fontsize=7.3,
                 color=C_ATOM,
                 arrowprops=dict(arrowstyle="-", color=C_ATOM, lw=0.7))
    ax3.text(gz_x, 12.6, "focused CZ beam", fontsize=7.3, ha="center",
             color=C_DIGITAL)
    ax3.annotate("", xy=(gz_x, 9.5), xytext=(gz_x, 12),
                 arrowprops=dict(arrowstyle="-|>", color=C_DIGITAL,
                                 lw=1.2))

    # step 4 — return, resume evolution, measure
    ax4 = scene(3, blue_on=True, red_on=False)
    badge(ax4, "4", "return — resume ev$(\\tau,T)$, then measure",
          C_ANALOG)
    pair(ax4, atoms)
    ax4.plot([gz_x, gz_x, x0 + 1.5, x0 + 1.5],
             [0.0, -lane_y, -lane_y, y0 - 0.5], ls="--", color=C_DIGITAL,
             lw=1.0, alpha=0.6)
    ax4.add_patch(FancyArrowPatch((70, -lane_y), (54, -lane_y),
                                  arrowstyle="-|>", mutation_scale=12,
                                  color=C_DIGITAL, lw=0, alpha=0.6))
    ax4.text(58, -lane_y - 2.6, "return lane", fontsize=7, ha="center",
             color=C_DIGITAL, alpha=0.75)
    ax4.text(0, -12.2, "dressing + combs back ON", fontsize=7.3,
             ha="center", color=C_ANALOG)
    ax4.text(84, -11.8, "then: global readout light (meas)",
             fontsize=7, ha="center", color=C_SOFT)

    fig.text(0.03, 0.955, "Space: zone architecture, step by step",
             fontsize=13, color=C_ANALOG, fontweight="bold")
    fig.text(0.19, 0.012,
             r"$H(x)=\sin 2x\,(Z_0Z_1+X_0+X_1)$ — dressing (ZZ) + "
             "addressing combs (X);  compiled pair $n{=}2$, faded atoms "
             "schematic, $y$ not to scale", fontsize=8, ha="center",
             color=C_ATOM)

    # ═══ Panel B: time ═══════════════════════════════════════════════════════
    axB = fig.add_subplot(gs[0, 1])
    axB.set_xlim(0, nb - 0.999 + 0.9)      # room for "meas" stage at right
    axB.set_ylim(-1.9, 12.9)
    axB.axis("off")
    axB.set_title("Time: one two-qubit-kick branch (event-spaced axis)",
                  fontsize=13, color=C_ANALOG, loc="left",
                  fontweight="bold")
    # step badges tying the timeline to the storyboard at left
    for xm, num, col in ((0.5, "1", C_ANALOG),
                         (2.5, "2", C_DIGITAL),
                         (warp((cz["t0"] + cz["t1"]) / 2), "3", C_DIGITAL),
                         (6.5, "4", C_ANALOG)):
        axB.text(xm, 12.35, num, fontsize=8, color="white", ha="center",
                 va="center",
                 bbox=dict(boxstyle="circle,pad=0.22", fc=col, ec="none"))

    names = stage_names(bounds, cz)
    meas_x0, meas_x1 = nb - 1, nb - 0.1    # synthetic readout stage

    # interval scaffolding + stage headers + true-time ticks
    for i in range(nb):
        axB.plot([i, i], [-1.0, 11.4], color="gray", lw=0.4, alpha=0.30)
    for i, nm in enumerate(names):
        axB.text(i + 0.5, 11.65, nm, fontsize=7.5, ha="center",
                 color=C_ATOM)
    axB.text((meas_x0 + meas_x1) / 2, 11.65, "meas", fontsize=7.5,
             ha="center", color=C_SOFT)
    for i, b in enumerate(bounds):
        v = b * 1e-3
        axB.text(i, -1.25, f"{v:.4g}" if v < 100 else f"{v:.1f}",
                 fontsize=6.5, ha="right", va="top", rotation=45,
                 color="#444444")
    axB.text(meas_x1, -1.25, "t (μs)", fontsize=7, ha="left",
             va="top", color="#444444")

    # program-level brace: ev ; insertion ; ev ; meas
    ins_x0, ins_x1 = 1.0, nb - 2.0
    for (xa, xb, lab, col) in ((0, 1, "", C_ANALOG),
                               (ins_x0, ins_x1, "insertion window",
                                C_DIGITAL),
                               (nb - 2, nb - 1, "", C_ANALOG),
                               (meas_x0, meas_x1, "", C_SOFT)):
        axB.plot([xa + 0.03, xb - 0.03], [11.15, 11.15], color=col, lw=1.6)
        if lab:
            axB.text((xa + xb) / 2, 10.75, lab, fontsize=8, ha="center",
                     color=col)

    def lane_label(y, num, text, col):
        axB.add_patch(Circle((-0.35, y), 0.17, fc=col, ec="none",
                             clip_on=False))
        axB.text(-0.35, y, num, fontsize=8, color="white", ha="center",
                 va="center", clip_on=False)
        axB.text(-0.62, y, text, fontsize=8, color=C_ATOM, ha="right",
                 va="center", clip_on=False)

    # ── lane 1: global drives ------------------------------------------------
    y1, h1 = 9.6, 1.05
    lane_label(y1 + 0.4, "1", "global drives\n(dressing +\naddressing)",
               C_ANALOG)
    env_max = max(max(b["env"]) for b in
                  data["dressing"] + data["addr_rabi"])
    for key, alpha in (("addr_rabi", 0.55), ("dressing", 0.95)):
        for b in data[key]:
            t = np.linspace(b["t0"], b["t1"], len(b["env"]))
            e = np.asarray(b["env"]) / env_max * h1
            axB.fill_between(warp(t), y1, y1 + e, color=C_ANALOG,
                             alpha=0.30 * alpha, lw=0)
            axB.plot(warp(t), y1 + e, color=C_ANALOG, lw=0.7, alpha=alpha)
    axB.text(warp((bounds[0] + bounds[1]) / 2), y1 - 0.42, "ON",
             fontsize=8, ha="center", color=C_ANALOG, fontweight="bold")
    axB.text(warp((bounds[-2] + bounds[-1]) / 2), y1 - 0.42, "ON",
             fontsize=8, ha="center", color=C_ANALOG, fontweight="bold")
    axB.text((ins_x0 + ins_x1) / 2, y1 + 0.45, "OFF (frozen)", fontsize=9,
             ha="center", color=C_ANALOG, alpha=0.85)
    axB.plot([ins_x0, ins_x1], [y1, y1], color=C_ANALOG, lw=1.0, alpha=0.6)

    # ── lane 2: AOD transport (x position of both tones) ---------------------
    y2, h2 = 7.0, 1.5
    lane_label(y2 + 0.7, "2", "AOD transport\n$x(t)$, both tones",
               C_DIGITAL)
    gz_x = meta["gate_zone"][0]
    for tr in data["x_tones"]:
        t = np.asarray(tr["t"]); um = np.asarray(tr["um"])
        axB.plot(warp(t), y2 + um / gz_x * h2, color=C_DIGITAL, lw=1.2)
    axB.text(warp(bounds[2] + 0.4 * (bounds[3] - bounds[2])), y2 + 1.75,
             "move (min-jerk,\n"
             f"{(bounds[3] - bounds[2]) * 1e-3:.0f} μs)", fontsize=7.5,
             ha="center", color=C_DIGITAL)
    axB.text(warp((cz["t0"] + cz["t1"]) / 2), y2 + h2 + 0.22,
             "park in gate zone", fontsize=7.5, ha="center",
             color=C_DIGITAL)
    axB.text(0.5, y2 + 0.25, "idle", fontsize=7.5, ha="center",
             color=C_DIGITAL, alpha=0.8)

    # ── lane 3: gate-zone pulse ----------------------------------------------
    y3, h3 = 5.35, 0.95
    lane_label(y3 + 0.4, "3", "gate-zone\npulse", C_DIGITAL)
    axB.plot([0, warp(cz["t0"])], [y3, y3], color=C_DIGITAL, lw=1.0)
    axB.add_patch(Rectangle((warp(cz["t0"]), y3),
                            warp(cz["t1"]) - warp(cz["t0"]), h3,
                            fc=C_DIGITAL, alpha=0.75, ec="none"))
    axB.plot([warp(cz["t1"]), nb - 1], [y3, y3], color=C_DIGITAL, lw=1.0)
    axB.text(warp((cz["t0"] + cz["t1"]) / 2), y3 + h3 + 0.18,
             f"CZ  ({(cz['t1'] - cz['t0']):.0f} ns, amp π)",
             fontsize=8, ha="center", color=C_DIGITAL)
    axB.text(warp((cz["t0"] + cz["t1"]) / 2), y3 - 0.42,
             "costs benchmarked $\\epsilon_{ins}$", fontsize=7.5,
             ha="center", color=C_DIGITAL, alpha=0.9)

    # ── lane 4: frame updates (software-only) --------------------------------
    y4 = 4.0
    lane_label(y4, "4", "frame updates\n(software)", C_SOFT)
    axB.plot([0, meas_x1], [y4, y4], ls=(0, (4, 3)), color=C_SOFT, lw=0.9)
    xcz = warp((cz["t0"] + cz["t1"]) / 2)
    axB.plot([xcz, xcz], [y4 - 0.18, y4 + 0.18], color=C_SOFT, lw=1.6)
    axB.annotate("virtual $R_z(s\\pi/2)^{\\otimes 2}$ — the branch\n"
                 "sign $s$ exists only here",
                 xy=(xcz + 0.05, y4), xytext=(xcz + 2.1, y4 + 0.35),
                 fontsize=7.5, color=C_SOFT, va="center",
                 arrowprops=dict(arrowstyle="-", color=C_SOFT, lw=0.7))

    # ── lane 5: pair y-position (transit lane) -------------------------------
    y5, h5 = 1.6, 1.15
    lane_label(y5 + 0.5, "5", "pair position\n$y(t)$ (lane)", C_DIGITAL)
    dy = meta["transit_dy"]
    for tr in data["y_tones"]:
        t = np.asarray(tr["t"]); um = np.asarray(tr["um"])
        axB.plot(warp(t), y5 + um / dy * h5 * 0.8, color=C_DIGITAL, lw=1.2)
    axB.text(0.5, y5 - 0.30, "in array ($y{=}0$)", fontsize=7.5,
             ha="center", color=C_DIGITAL, alpha=0.85)
    axB.text(warp(bounds[3]), y5 + h5 + 0.05,
             f"transit lane ($y{{=}}{dy:+.0f}$ μm)", fontsize=7.5,
             ha="center", color=C_DIGITAL)
    axB.text(xcz, y5 - 0.30, "drop to $y{=}0$,\npair at $R_{cz}$",
             fontsize=7, ha="center", color=C_DIGITAL, alpha=0.9)

    # ── cost strip -----------------------------------------------------------
    yc = -0.55
    tot_move_us = sum((bounds[i + 1] - bounds[i]) * 1e-3
                      for i, nm in enumerate(names)
                      if nm in ("lift", "drop") or nm.startswith("move"))
    costs = ((0, 1, "costs $T/T_2^*$\n(dressed)", C_ANALOG),
             (ins_x0, ins_x1,
              f"{tot_move_us:.0f} μs ground-state clock "
              "(no dressing on)", C_DIGITAL),
             (nb - 2, nb - 1, "costs $T/T_2^*$\n(dressed)", C_ANALOG),
             (meas_x0, meas_x1, "classical\nreadout", C_SOFT))
    for xa, xb, lab, col in costs:
        axB.plot([xa + 0.05, xb - 0.05], [yc, yc], color=col, lw=1.2)
        axB.plot([xa + 0.05] * 2, [yc, yc + 0.12], color=col, lw=1.2)
        axB.plot([xb - 0.05] * 2, [yc, yc + 0.12], color=col, lw=1.2)
        axB.text((xa + xb) / 2, yc - 0.16, lab, fontsize=7, ha="center",
                 va="top", color=col)

    # meas stage marker
    axB.add_patch(Rectangle((meas_x0 + 0.05, y1), meas_x1 - meas_x0 - 0.1,
                            h1, fc="none", ec=C_SOFT, ls="--", lw=0.9))
    axB.text((meas_x0 + meas_x1) / 2, y1 + 0.45, "readout\nlight",
             fontsize=7, ha="center", va="center", color=C_SOFT)

    # legend
    axB.plot([], [], color=C_ANALOG, lw=3, label="ANALOG (continuous)")
    axB.plot([], [], color=C_DIGITAL, lw=3, label="DIGITAL (discrete)")
    axB.plot([], [], color=C_SOFT, lw=1.5, ls=(0, (4, 3)),
             label="SOFTWARE-ONLY")
    fig.legend(loc="lower center", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.72, -0.005))

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
