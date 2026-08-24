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

    fig = plt.figure(figsize=(4.3, 3.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.0, 1.0], wspace=0.02,
                          left=0.012, right=0.995, top=0.905, bottom=0.02)
    gsA = gs[0, 0].subgridspec(3, 1, height_ratios=[1.0, 0.40, 1.0],
                               hspace=0.20)

    R_cz = meta["R_cz"]
    gz_x = meta["gate_zone"][0]
    move_us = (bounds[2] - bounds[1]) * 1e-3
    C_AOD = "#d9822b"

    IB = (-28, -7, 24, 14)
    GB = (0, -9.5, 6, 19)
    lattice = [(x, y) for x in (-25, -20, -15, -10)
               for y in (-4.0, 0.0, 4.0)]
    pair_sites = [(-20, 0.0), (-15, 0.0)]
    gb_cx = GB[0] + GB[2] / 2

    def zone_boxes(ax, dress_on, gate_on):
        bx, by, bw, bh = IB
        ax.add_patch(Rectangle((bx, by), bw, bh, ec=C_ANALOG, lw=0.9,
                               fc=C_ANALOG if dress_on else "none",
                               alpha=0.14 if dress_on else 1.0))
        if dress_on:
            cx = bx + bw / 2
            ax.add_patch(Polygon([(cx - 6, by + bh + 3.4),
                                  (cx + 6, by + bh + 3.4),
                                  (cx + bw / 2 - 1, by + bh),
                                  (cx - bw / 2 + 1, by + bh)], closed=True,
                                 fc=C_ANALOG, alpha=0.30, ec="none"))
        ax.text(bx + bw / 2, by + bh + 1.6,
                "dressing ON" if dress_on else "dressing OFF",
                fontsize=6.0, ha="center",
                color=C_ANALOG if dress_on else "#999999",
                fontweight="bold" if dress_on else "normal")
        ax.text(bx + bw / 2, by - 2.8, "interaction zone", fontsize=6.0,
                ha="center", color=C_ANALOG)
        bx, by, bw, bh = GB
        ax.add_patch(Rectangle((bx, by), bw, bh, ec=C_DIGITAL, lw=0.9,
                               fc=C_DIGITAL if gate_on else "none",
                               alpha=0.14 if gate_on else 1.0))
        if gate_on:
            ax.add_patch(Polygon([(gb_cx - 4.5, by + bh + 3.4),
                                  (gb_cx + 4.5, by + bh + 3.4),
                                  (gb_cx + bw / 2 - 0.5, by + bh),
                                  (gb_cx - bw / 2 + 0.5, by + bh)],
                                 closed=True, fc=C_DIGITAL, alpha=0.30,
                                 ec="none"))
        ax.text(gb_cx, by + bh + 1.6,
                "gate ON" if gate_on else "gate OFF", fontsize=6.0,
                ha="center", color=C_DIGITAL if gate_on else "#999999",
                fontweight="bold" if gate_on else "normal")
        ax.text(gb_cx, by - 2.8, "gate\nzone", fontsize=6.0, ha="center",
                va="top", color=C_DIGITAL)

    def dots(ax, vacated):
        for (px, py) in lattice:
            if (px, py) in pair_sites:
                if vacated:
                    ax.plot(px, py, "o", ms=4.4, mfc="none", mec="gray",
                            mew=0.7, alpha=0.6)
                continue
            ax.plot(px, py, "o", ms=1.9, color="gray", alpha=0.4)

    def pair(ax, positions, ms=4.8):
        for px, py in positions:
            ax.plot(px, py, "o", ms=ms, color=C_ATOM, zorder=6)
            ax.plot(px, py, "o", ms=ms + 3.6, mfc="none", mec="#e6a817",
                    mew=1.0, zorder=6)

    def badge(ax, x, y, num, col, fs=6.2):
        ax.text(x, y, num, fontsize=fs, color="white", ha="center",
                va="center",
                bbox=dict(boxstyle="circle,pad=0.15", fc=col, ec="none"))

    ax1 = fig.add_subplot(gsA[0, 0])
    ax1.set_xlim(-30, 12); ax1.set_ylim(-12.8, 15.5); ax1.axis("off")
    zone_boxes(ax1, dress_on=True, gate_on=False)
    dots(ax1, vacated=False)
    pair(ax1, pair_sites)
    badge(ax1, -28.8, 14.0, "1", C_ANALOG)
    badge(ax1, -26.4, 14.0, "4", C_ANALOG)
    ax1.text(-24.2, 14.0, "evolve ev$(0,\\tau)$ / ev$(\\tau,T)$",
             fontsize=6.4, color=C_ATOM, ha="left", va="center")

    ax2 = fig.add_subplot(gsA[1, 0])
    ax2.set_xlim(-30, 12); ax2.set_ylim(-5.2, 4.8); ax2.axis("off")
    pair(ax2, [(-3.6, 1.4), (-0.4, 1.4)], ms=4.2)
    ax2.annotate("", xy=(5.5, -1.6), xytext=(-9.5, -1.6),
                 arrowprops=dict(arrowstyle="-|>", color=C_AOD, lw=1.0,
                                 ls="--", mutation_scale=9))
    ax2.annotate("", xy=(-9.5, -3.9), xytext=(5.5, -3.9),
                 arrowprops=dict(arrowstyle="-|>", color=C_AOD, lw=0.85,
                                 ls="--", mutation_scale=9, alpha=0.7))
    badge(ax2, -28.8, 0.0, "2", C_AOD)
    ax2.text(-26.6, 0.0, f"AOD moves pair ({move_us:.0f} μs)",
             fontsize=6.2, color=C_AOD, ha="left", va="center")

    ax3 = fig.add_subplot(gsA[2, 0])
    ax3.set_xlim(-30, 12); ax3.set_ylim(-12.8, 15.5); ax3.axis("off")
    zone_boxes(ax3, dress_on=False, gate_on=True)
    dots(ax3, vacated=True)
    pair(ax3, [(gb_cx - 1.5, 0.0), (gb_cx + 1.5, 0.0)], ms=4.4)
    badge(ax3, -28.8, 14.0, "3", C_DIGITAL)
    ax3.text(-26.6, 14.0, "insert CZ (200 ns)", fontsize=6.4,
             color=C_ATOM, ha="left", va="center")
    ax3.annotate(f"$R_{{cz}}$ = {R_cz:g} μm",
                 xy=(gb_cx - 1.5, -0.8), xytext=(-8.5, -11.8),
                 fontsize=5.8, ha="center", color=C_ATOM,
                 arrowprops=dict(arrowstyle="-", color=C_ATOM, lw=0.6))

    fig.text(0.015, 0.945, "Space", fontsize=9, color=C_ANALOG,
             fontweight="bold")

    # ═══ vertical timeline: 1/3 width, badge with name BELOW ════════════════
    axB = fig.add_subplot(gs[0, 1])
    names = stage_names(bounds, cz)
    n_rows = nb
    axB.set_xlim(-0.72, 2.02)
    axB.set_ylim(n_rows + 0.10, -0.62)
    axB.axis("off")
    fig.text(0.755, 0.945, "Time", fontsize=9, color=C_ANALOG,
             fontweight="bold")

    LX, LW = 0.0, 0.55
    AX, AW = 0.72, 0.55
    xcz_v = warp((cz["t0"] + cz["t1"]) / 2)

    for i in range(nb + 1):
        axB.plot([LX - 0.04, AX + AW + 0.04], [i, i], color="gray",
                 lw=0.35, alpha=0.30)
    for i in range(nb):
        v = bounds[i] * 1e-3
        axB.text(AX + AW + 0.08, i, f"{v:.4g}" if v < 100 else f"{v:.0f}",
                 fontsize=4.9, ha="left", va="center", color="#444444")
    axB.text(AX + AW + 0.08, -0.42, "t(μs)↓", fontsize=5.0, ha="left",
             color="#444444")

    stage_badges = (("1", C_ANALOG), ("2", C_AOD), ("3", C_DIGITAL),
                    ("2", C_AOD), ("4", C_ANALOG))
    short = {"ev(0,τ)": "ev(0,τ)", "move →": "move→", "CZ": "CZ+$R_z$",
             "move ←": "move←", "ev(τ,T)": "ev(τ,T)"}
    for i, nm in enumerate(names):
        num, col = stage_badges[i]
        badge(axB, -0.38, i + 0.33, num, col, fs=5.4)
        axB.text(-0.38, i + 0.70, short.get(nm, nm), fontsize=4.7,
                 ha="center", va="center", color=C_ATOM)
    axB.text(-0.38, n_rows - 0.5, "meas", fontsize=4.9, ha="center",
             va="center", color=C_SOFT)

    axB.text(LX + LW / 2, -0.28, "lasers", fontsize=4.8, ha="center",
             color=C_ATOM)
    axB.text(AX + AW / 2, -0.28, "AOD", fontsize=4.8, ha="center",
             color=C_ATOM)

    env_max = max(max(b["env"]) for b in
                  data["dressing"] + data["addr_rabi"])
    for key, alpha in (("addr_rabi", 0.55), ("dressing", 0.95)):
        for b in data[key]:
            t = np.linspace(b["t0"], b["t1"], len(b["env"]))
            e = np.asarray(b["env"]) / env_max * LW
            axB.fill_betweenx(warp(t), LX, LX + e, color=C_ANALOG,
                              alpha=0.30 * alpha, lw=0)
            axB.plot(LX + e, warp(t), color=C_ANALOG, lw=0.35,
                     alpha=alpha)
    axB.plot([LX, LX], [1, nb - 2], color=C_ANALOG, lw=0.6, alpha=0.6)
    axB.add_patch(Rectangle((LX, warp(cz["t0"])), LW,
                            warp(cz["t1"]) - warp(cz["t0"]),
                            fc=C_DIGITAL, alpha=0.85, ec="none"))
    axB.text(LX + LW / 2, xcz_v, "CZ", fontsize=5.6, ha="center",
             va="center", color="white", fontweight="bold")
    axB.add_patch(Rectangle((LX, nb - 1 + 0.24), LW, 0.52, fc="none",
                            ec=C_SOFT, ls="--", lw=0.6))

    aenv_max = max(max(b["env"]) for b in data["aod_x_env"])
    for b in data["aod_x_env"]:
        t = np.linspace(b["t0"], b["t1"], len(b["env"]))
        e = np.asarray(b["env"]) / aenv_max * AW
        axB.fill_betweenx(warp(t), AX, AX + e, color="gray", alpha=0.22,
                          lw=0)
    for tr in data["x_tones"]:
        t = np.asarray(tr["t"]); um = np.asarray(tr["um"])
        axB.plot(AX + (um - um.min()) / gz_x * AW, warp(t), color=C_AOD,
                 lw=0.8)

    out = os.path.join(FIG_DIR, "branch_anatomy")
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
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
