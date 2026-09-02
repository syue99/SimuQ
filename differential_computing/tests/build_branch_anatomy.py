"""
build_branch_anatomy.py — Fig 5 (fig:branch): one compiled PSR branch as one
artifact, per the 2026-08-25 redraw guide.

Three columns (figure*, full text width), stage numbers ①②③④ shared:

  Space  — UNCHANGED zone cartoon (interaction/gate zones, dressing ON/OFF,
           AOD move, CZ insert, R_cz).
  Time   — vertical wall-clock axis, NOT to scale (axis breaks between
           stages); per stage ONLY the active channels, as real traces from
           the compiled per-channel waveform output; a compact 8-channel ×
           4-stage coverage table under the axis.
  Ledger — monospace excerpt of the real PulseLedger rows for this branch:
           segment id · semantic clock · wall clock · active terms with
           provenance · frame state · transport plan · insertion marker.

Channel-name mapping to the App F inventory (emission-layer truth):
  move-AOD x/y  = TRANSPORT_AOD_X / _Y (tone frequency = tweezer position)
  drive I / Q   = Re / Im of the ADDR_RABI addressing comb waveform
  dressing      = DRESSING_AOM;   gate = GATE_AOM (measured 696 ns shape)
  addr-AOD x/y  = addressing-beam steering — idle for this branch (the comb
                  tone frequencies are static), shown blank in the coverage
                  table.

Phases: extract (runs the pipeline, caches figures/branch_anatomy_data.json)
and render (reads the JSON — REBUILD=1 forces re-extraction).  Never re-run
the pipeline just to tweak the plot.  Every number in the Time and Ledger
columns comes from the schedule / PulseLedger — nothing is invented.

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

C_ANALOG = "#1f5fbf"      # analog / continuous (stages ① ④) + drive I/Q
C_DIGITAL = "#d62728"     # digital / discrete  (stage ③)   + gate
C_AOD = "#d9822b"         # transport            (stage ②)   + move x/y
C_DRESS = "#2a9d8f"       # dressing field (channel color in the waveform col)
C_SOFT = "#666666"        # software-only / measure
C_ATOM = "#26343f"

DRIVE_EXCERPT_NS = 250.0  # length of the evolution-drive excerpt windows


# ── extraction ────────────────────────────────────────────────────────────────

def extract():
    """Compile the running instance and distill the figure data to JSON."""
    import physical_channels as pc
    from awg_compile import ChirpTone, SampledTone, _fallback_waveform
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
        """[{t, um, mhz}] per tone entry (chirps sampled, holds two-point)."""
        traces = []
        for t0, t1, p in entries(ch):
            wf = p.waveform
            if isinstance(wf, ChirpTone):
                tt = np.linspace(0.0, wf.duration_ns, 120)
                f = wf.instantaneous_freq_mhz(tt)
                traces.append({"t": (t0 + tt).tolist(),
                               "um": pos_of(f).tolist(),
                               "mhz": f.tolist()})
            elif getattr(wf, "freq_mhz", 0.0):
                f = float(wf.freq_mhz)
                traces.append({"t": [t0, t1],
                               "um": [float(pos_of(f))] * 2,
                               "mhz": [f, f]})
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

    def window_iq(ch, t0, dur_ns, dt=0.5):
        """Real I/Q excerpt of one channel summed over [t0, t0+dur)."""
        t = np.arange(0.0, dur_ns, dt)
        w = np.zeros(len(t), dtype=complex)
        for e0, e1, p in entries(ch):
            m = (t + t0 >= e0) & (t + t0 < e1)
            if not m.any():
                continue
            fn = p.waveform if p.waveform is not None \
                else _fallback_waveform(p)
            w[m] += fn(t[m] + t0 - e0)
        return t, w

    # guard the gate-shape routing: measured shape on GATE_AOM only
    assert all(isinstance(p.waveform, SampledTone)
               for _, _, p in entries(pc.GATE_AOM)), \
        "GATE_AOM play without the measured gate shape"
    for chn in (pc.DRESSING_AOM, pc.ADDR_RABI, pc.ADDR_DET):
        assert not any(isinstance(p.waveform, SampledTone)
                       for _, _, p in entries(chn)), \
            f"sampled gate shape leaked onto channel {chn}"

    gate = min(entries(pc.GATE_AOM), key=lambda e: e[1] - e[0])
    gt = np.linspace(0.0, gate[1] - gate[0], 400)
    gwf = gate[2].waveform            # SampledTone — the measured gate
    twf = np.arange(0.0, gate[1] - gate[0], 0.5)
    cz = {"t0": gate[0], "t1": gate[1],
          "amp": float(gate[2].amplitude),
          "vz_phase": float(getattr(gate[2], "phase", 0.0) or 0.0),
          "env": np.abs(gwf(gt)).tolist(),
          "phi": np.interp(gt, gwf.t_ns - gwf.t_ns[0],
                           gwf.phase_rad).tolist(),
          "wf_t": twf.tolist(),
          "wf": np.imag(gwf(twf)).tolist()}

    # evolution-stage drive excerpts (real I/Q of the ADDR_RABI comb) and
    # the dressing level, per evolution stage
    ev_stages = []
    for (s0, s1) in ((bounds[0], bounds[1]), (bounds[-2], bounds[-1])):
        te, we = window_iq(pc.ADDR_RABI, s0, DRIVE_EXCERPT_NS)
        _, wd = window_iq(pc.DRESSING_AOM, s0, DRIVE_EXCERPT_NS)
        ev_stages.append({"t0": s0, "t1": s1,
                          "ex_t": te.tolist(),
                          "I": we.real.tolist(),
                          "Q": we.imag.tolist(),
                          "dress_level": float(np.abs(wd).max())})

    # frame-table events: the CZ's virtual-Z on both qubits (from the ledger)
    cz_ledger = [e for e in mapper.ledger.entries
                 if e.channel_kind == "cz_gate"]
    assert len(cz_ledger) == 1
    frame = {"t": cz["t1"],
             "qubits": list(cz_ledger[0].target_qubits),
             "phase": float(cz_ledger[0].phase)}

    # ledger excerpt: one row per branch segment, straight off the ledger
    kick = [e for e in mapper.ledger.entries if e.channel_kind == "kick"]
    assert len(kick) == 1 and \
        [(str(p), c) for p, c in kick[0].hamiltonian.ham][0][1] in (1.0, -1.0)
    aods = [e for e in mapper.ledger.entries if e.op_type == "aod"]
    b_us = [b * 1e-3 for b in bounds]
    src_terms = "Ω·Xa Ω·Xb θJ·ZaZb (src)"
    ledger_rows = [
        {"stage": "1", "seg": "seg0", "sem": "[0,τ)",
         "wall": [b_us[0], b_us[1]], "terms": src_terms,
         "frame": "–", "transport": "–", "ins": "–"},
        {"stage": "2", "seg": "seg1", "sem": "–",
         "wall": [b_us[1], b_us[2]], "terms": "–",
         "frame": "–",
         "transport": "a,b→" + aods[0].zone[0], "ins": "–"},
        {"stage": "3", "seg": "seg2", "sem": "[0,π/4)",
         "wall": [b_us[2], b_us[3]],
         "terms": "s·ZaZb (insertion)",
         # The ledger stores the frame update as phi, the coefficient of Z in
         # e^{-i·phi·Z} (cz_kick_decomposition).  This row prints an Rz ANGLE,
         # and appendix eq:zz-lower fixes Rz(alpha) = e^{-i·alpha·Z/2}, so the
         # printed angle is alpha = 2·phi = s·0.5π.  The stored phase is right;
         # printing phi as if it were alpha was the bug (it halved the angle).
         "frame": f"Rz(s·{2 * frame['phase'] / np.pi:.2g}π) a,b",
         "transport": "–", "ins": "INS"},
        {"stage": "2", "seg": "seg3", "sem": "–",
         "wall": [b_us[3], b_us[4]], "terms": "–",
         "frame": "–",
         "transport": "a,b→" + {"interaction": "int"}.get(
             aods[1].zone[0], aods[1].zone[0]), "ins": "–"},
        {"stage": "4", "seg": "seg4", "sem": "[τ,T)",
         "wall": [b_us[4], b_us[5]], "terms": src_terms,
         "frame": "–", "transport": "–", "ins": "–"},
    ]

    # coverage: which App-F channels are active in stages ① ② ③ ④
    def active(ch, lo, hi):
        return any(e0 < hi - 1.0 and e1 > lo + 1.0 and
                   (np.abs(p.amplitude or 0.0) > 1e-12)
                   for e0, e1, p in entries(ch))

    ev_w = [(bounds[0], bounds[1]), (bounds[4], bounds[5])]
    mv_w = [(bounds[1], bounds[2]), (bounds[3], bounds[4])]
    cz_w = [(bounds[2], bounds[3])]

    def stages_of(ch):
        return [any(active(ch, lo, hi) for lo, hi in ev_w),
                any(active(ch, lo, hi) for lo, hi in mv_w),
                any(active(ch, lo, hi) for lo, hi in cz_w),
                any(active(ch, lo, hi) for lo, hi in ev_w)]

    coverage = [
        ("move-AOD x", stages_of(pc.TRANSPORT_AOD_X)),
        ("move-AOD y", stages_of(pc.TRANSPORT_AOD_Y)),
        ("addr-AOD x", [False] * 4),        # steering idle: static tone freqs
        ("addr-AOD y", [False] * 4),
        ("drive I", stages_of(pc.ADDR_RABI)),
        ("drive Q", stages_of(pc.ADDR_RABI)),
        ("dressing", stages_of(pc.DRESSING_AOM)),
        ("gate", stages_of(pc.GATE_AOM)),
    ]

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
        "aod_y_env": drive_blocks(pc.TRANSPORT_AOD_Y),
        "cz": cz,
        "ev_stages": ev_stages,
        "frame": frame,
        "ledger_rows": ledger_rows,
        "coverage": coverage,
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

    # compact layout, same canvas as the original working anatomy:
    # left = Space (top) + Ledger (bottom); right = ONE combined waveform
    # column (channels by color) with the coverage table below it
    # right column sized to END at the content's right edge (no white):
    # axT gets xlim (0, 0.81) with width scaled by the same 0.81, so every
    # element keeps its physical size while the axis stops at the content.
    fig = plt.figure(figsize=(3.98, 3.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.58, 0.731],
                          height_ratios=[0.72, 0.28],
                          wspace=0.02, hspace=0.08,
                          left=0.008, right=0.998, top=0.905, bottom=0.015)
    gsA = gs[0, 0].subgridspec(3, 1, height_ratios=[1.0, 0.40, 1.0],
                               hspace=0.20)

    R_cz = meta["R_cz"]
    gz_x = meta["gate_zone"][0]
    move_us = (bounds[2] - bounds[1]) * 1e-3

    # ═══ column 1 — Space (UNCHANGED drawing) ═══════════════════════════════
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
    cz_ns = cz["t1"] - cz["t0"]
    ax3.text(-26.6, 14.0, f"insert CZ ({cz_ns:.0f} ns)", fontsize=6.4,
             color=C_ATOM, ha="left", va="center")
    ax3.annotate(f"$R_{{cz}}$ = {R_cz:g} μm",
                 xy=(gb_cx - 1.5, -0.8), xytext=(-8.5, -11.8),
                 fontsize=5.8, ha="center", color=C_ATOM,
                 arrowprops=dict(arrowstyle="-", color=C_ATOM, lw=0.6))

    fig.text(0.015, 0.945, "Space", fontsize=9, color=C_ANALOG,
             fontweight="bold")

    # ═══ right column — Time: ONE combined waveform column ══════════════════
    # Time runs DOWN; every channel shares the SAME column, told apart by
    # color (drive blue, dressing teal, gate red, move orange).  Translucent
    # trace = the real (carrier-dense) waveform; opaque curve on top = the
    # salient content (envelope for drives, tone f(t) for transport).
    # The coverage table sits directly below, channel names in their colors.
    axT = fig.add_subplot(gs[:, 1])
    axT.set_xlim(0, 0.795); axT.set_ylim(0, 1); axT.axis("off")
    fig.text(0.70, 0.945, "Time", fontsize=9, color=C_ANALOG,
             fontweight="bold")

    CH_COLORS = {"drive I/Q": C_ANALOG, "dressing": C_DRESS,
                 "gate": C_DIGITAL, "move x/y": C_AOD}

    b_us = [b * 1e-3 for b in bounds]
    band_h = [0.115, 0.085, 0.100, 0.085, 0.115]
    GAP = 0.014
    stage_meta = [("1", C_ANALOG, "ev(0,τ)"), ("2", C_AOD, "move →"),
                  ("3", C_DIGITAL, "CZ+$R_z$"), ("2", C_AOD, "move ←"),
                  ("4", C_ANALOG, "ev(τ,T)")]
    AXX = 0.33                       # vertical axis x
    CX, HW = 0.62, 0.13             # single THIN waveform strip

    # channel colors are identified by the coverage-table names below
    y_top = 0.945

    def brk(y):                      # axis-break marks
        for dy in (0.004, -0.004):
            axT.plot([AXX - 0.014, AXX + 0.014], [y + dy - 0.003,
                                                  y + dy + 0.003],
                     color="#444444", lw=0.6)

    def tick(y, t_us, sym=None):
        lab = f"{t_us:.4g}" if t_us < 100 else f"{t_us:.1f}"
        if sym:
            lab += f" ({sym})"
        axT.text(AXX - 0.03, y, lab, fontsize=4.6, ha="right",
                 va="center", color="#444444")

    def vwave(y0, y1, vals, col, lw=0.25, alpha=0.9, norm=None):
        """Vertical waveform in the shared column, deflecting horizontally."""
        v = np.asarray(vals, dtype=float)
        n = norm if norm else max(np.abs(v).max(), 1e-12)
        yy = np.linspace(y1, y0, len(v))
        axT.plot(CX + v / n * HW, yy, color=col, lw=lw, alpha=alpha)

    y = y_top
    boundary_syms = {1: "τ", 5: "T"}
    tick(y, b_us[0])
    for si, ((num, scol, name), h) in enumerate(zip(stage_meta, band_h)):
        y1, y0 = y, y - h            # band spans [y0, y1]
        axT.plot([AXX, AXX], [y0, y1], color="#888888", lw=0.7)
        badge(axT, 0.05, (y0 + y1) / 2 + 0.016, num, scol, fs=5.0)
        axT.text(0.05, (y0 + y1) / 2 - 0.026, name, fontsize=4.3,
                 ha="center", va="center", color=C_ATOM)

        if si in (0, 4):             # ① ④ — dressing band + drive spindle
            ev = data["ev_stages"][0 if si == 0 else 1]
            I, Q = np.asarray(ev["I"]), np.asarray(ev["Q"])
            env = np.abs(I + 1j * Q)
            nrm = env.max()
            cd = CH_COLORS["dressing"]
            frac = ev["dress_level"] / nrm
            axT.fill_betweenx([y0, y1], CX - frac * HW, CX + frac * HW,
                              color=cd, alpha=0.16, lw=0)
            for sgn in (1, -1):
                axT.plot([CX + sgn * frac * HW] * 2, [y0, y1], color=cd,
                         lw=0.7)
            cdr = CH_COLORS["drive I/Q"]
            vwave(y0, y1, I, cdr, lw=0.22, alpha=0.30, norm=nrm)
            vwave(y0, y1, Q, cdr, lw=0.22, alpha=0.30, norm=nrm)
            vwave(y0, y1, env, cdr, lw=0.6, norm=nrm)
            vwave(y0, y1, -env, cdr, lw=0.6, norm=nrm)
            axT.text(CX + HW, y0 + 0.003, f"{DRIVE_EXCERPT_NS:.0f} ns",
                     fontsize=3.5, ha="right", va="bottom",
                     color="#888888")
        elif si in (1, 3):           # ② — move x/y in the same column
            lo_t, hi_t = bounds[si], bounds[si + 1]
            ca = CH_COLORS["move x/y"]
            for key in ("aod_x_env", "aod_y_env"):
                for blk in data[key]:
                    if blk["t0"] < lo_t - 1 or blk["t1"] > hi_t + 1:
                        continue
                    e = np.asarray(blk["env"])
                    e = e / max(e.max(), 1e-12)
                    yy = np.linspace(y1, y0, len(e))
                    axT.fill_betweenx(yy, CX - e * HW, CX + e * HW,
                                      color=ca, alpha=0.06, lw=0)
            trs_x = [tr for tr in data["x_tones"]
                     if tr["t"][0] >= lo_t - 1 and tr["t"][-1] <= hi_t + 1]
            trs_y = [tr for tr in data["y_tones"]
                     if tr["t"][0] >= lo_t - 1 and tr["t"][-1] <= hi_t + 1]
            f_all = [v for tr in trs_x + trs_y for v in tr["mhz"]]
            flo, fhi = min(f_all), max(f_all)
            fspan = (fhi - flo) if fhi > flo else 1.0
            for trs, lw_, al in ((trs_x, 0.75, 1.0), (trs_y, 0.45, 0.75)):
                for tr in trs:
                    ts = (np.asarray(tr["t"]) - lo_t) / (hi_t - lo_t)
                    fs = (np.asarray(tr["mhz"]) - flo) / fspan
                    axT.plot(CX + (fs * 2 - 1) * 0.88 * HW,
                             y1 - ts * (y1 - y0), color=ca, lw=lw_,
                             alpha=al)
            axT.text(CX + HW, y0 + 0.003, "f(t)", fontsize=3.5,
                     ha="right", va="bottom", color="#888888")
        else:                        # ③ — gate + frame ticks
            cg = CH_COLORS["gate"]
            env = np.asarray(cz["env"])
            nrm = env.max()
            vwave(y0, y1, np.asarray(cz["wf"]), cg, lw=0.20, alpha=0.55,
                  norm=nrm)
            vwave(y0, y1, env, cg, lw=0.6, norm=nrm)
            vwave(y0, y1, -env, cg, lw=0.6, norm=nrm)
            for dq in (0.0, 0.012):
                axT.plot([CX + HW + 0.020 + dq] * 2,
                         [y0 + 0.15 * h, y0 + 0.45 * h], color=C_SOFT,
                         lw=0.9)
            axT.text(CX + HW + 0.016, y0 + 0.62 * h, "$R_z$",
                     fontsize=3.8, ha="left", va="center", color=C_SOFT)

        y = y0
        tick(y, b_us[si + 1], boundary_syms.get(si + 1))
        if si < 4:
            brk(y - GAP / 2)
            y -= GAP

    # meas box
    y0m = y - 0.042
    axT.add_patch(Rectangle((CX - 0.10, y0m), 0.20, 0.034, fc="none",
                            ec=C_SOFT, ls="--", lw=0.6))
    axT.text(CX - 0.114, y0m + 0.017, "meas", fontsize=4.3,
             ha="right", va="center", color=C_SOFT)

    # coverage table below the column, channel names in channel colors
    name_col = {"move-AOD x": C_AOD, "move-AOD y": C_AOD,
                "addr-AOD x": "#999999", "addr-AOD y": "#999999",
                "drive I": C_ANALOG, "drive Q": C_ANALOG,
                "dressing": C_DRESS, "gate": C_DIGITAL}
    ty = y0m - 0.035
    cols_x = [0.36, 0.46, 0.56, 0.66]
    stage_cols = [C_ANALOG, C_AOD, C_DIGITAL, C_ANALOG]
    for k, num in enumerate("1234"):
        axT.text(cols_x[k], ty, num, fontsize=4.2, ha="center",
                 color=stage_cols[k], fontweight="bold")
    rh = (ty - 0.018) / 8.4
    for ri, (nm, acts) in enumerate(data["coverage"]):
        ry = ty - (ri + 1) * rh
        axT.text(0.28, ry, nm, fontsize=3.9, ha="right", va="center",
                 color=name_col.get(nm, C_ATOM))
        for k, a in enumerate(acts):
            if a:
                axT.text(cols_x[k], ry, "✓", fontsize=4.0, ha="center",
                         va="center", color=stage_cols[k])

    # ═══ footer left — Ledger excerpt (real PulseLedger rows) ═══════════════
    axL = fig.add_subplot(gs[1, 0])
    axL.set_xlim(0, 1); axL.set_ylim(0, 1); axL.axis("off")
    axL.text(0.015, 0.985, "Ledger", fontsize=7, color=C_ANALOG,
             fontweight="bold", va="top")

    # concise: one line per segment — seg · wall clock · what happened
    stage_color = {"1": C_ANALOG, "2": C_AOD, "3": C_DIGITAL, "4": C_ANALOG}
    mono = dict(family="monospace", fontsize=4.8, va="center")

    def content(r):
        if r["ins"] == "INS":
            return f"ins{r['sem']}: {r['terms'].replace(' (insertion)', '')}" \
                   f" · {r['frame']}"
        if r["transport"] != "–":
            return f"move {r['transport']}"
        return f"ev{r['sem']}: {r['terms']}"

    y = 0.83
    axL.text(0.075, y, "seg  wall clock (µs)  content", family="monospace",
             fontsize=4.2, va="center", color="#777777")
    y -= 0.115
    for r in data["ledger_rows"]:
        col = stage_color[r["stage"]]
        if r["ins"] == "INS":
            axL.add_patch(plt.Rectangle((0.008, y - 0.048), 0.987, 0.096,
                                        fc=C_DIGITAL, alpha=0.07,
                                        ec="none"))
        badge(axL, 0.033, y, r["stage"], col, fs=4.4)
        axL.text(0.075, y,
                 f"{r['seg'][-1]}  [{r['wall'][0]:6.1f},{r['wall'][1]:6.1f}]"
                 f"  {content(r)}",
                 color=(C_DIGITAL if r["ins"] == "INS" else C_ATOM), **mono)
        y -= 0.115
    y -= 0.010
    axL.text(0.075, y,
             f"total {b_us[-1]:.1f} µs = "
             f"{b_us[1] - b_us[0] + b_us[5] - b_us[4]:.1f} ev + "
             f"{b_us[2] - b_us[1] + b_us[4] - b_us[3]:.1f} move + "
             f"{b_us[3] - b_us[2]:.2f} gate",
             family="monospace", fontsize=4.2, color="#777777")

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
