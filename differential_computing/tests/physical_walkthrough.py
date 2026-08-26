"""
physical_walkthrough.py — one PSR branch through the PHYSICAL channel stack,
all the way to sampled AWG waveforms.

    compile (2q) -> map_hlist_tree (logical, per-qubit)
    -> physical_channels.to_physical (6 fixed AOM/AOD channels, COMBs,
       crossed X/Y transport chirps)
    -> to_pulsedsl_tree (CombNode -> COMB, PlayNode -> Play, waveforms attached)
    -> RUN -> schedule.view()
    -> awg_compile.compile_waveforms  (per-channel complex sample arrays)
    -> awg_waveforms_2q.png           (the end-to-end waveform figure)

Shows per-qubit detuning/Rabi consolidated into ADDR_DET / ADDR_RABI tone combs,
dressing on DRESSING_AOM, ZZ on GATE_AOM, and tweezer moves as frequency chirps
on TRANSPORT_AOD_X / TRANSPORT_AOD_Y.

Run:  conda run -n qec_pg python differential_computing/tests/physical_walkthrough.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/Users/syue99/research/RISC-Q/PulseDSL/src/DSL/")

import numpy as np
import sympy as sp

from simuq import QSystem, Qubit
from simuq.braket.diffQC_provider import diffQCProvider, to_pulsedsl_tree
from observable_program_generator import observable_program_generator
from tweezer_mapper import TweezerMapper
import pulse_tree as pt
import physical_channels as pc


# ── Realistic device timing (user config, 2026-08-24) ────────────────────────
# CZ gate = the measured 696 ns fixed shape (see GATE_SHAPE below); dressing
# segments on the order of 1–10 μs (T = 5 μs here,
# split by the kick's sampled τ); transport is SPEED-limited at v_max = 4 m/s
# (Cicali et al.'s Eq.-(6) min-jerk profile, whose peak speed is (15/8)·d/τ,
# so a zone hop of d takes τ = (15/8)·d/v_max and needs peak acceleration
# a_pk = 10d/(√3 τ²) = (128/45√3)·v_max²/d — the accel is derived, not a cap).
# At 4 m/s the ~100 μm zone hop takes (15/8)·d/v ≈ 47 μs — the transport
# budget in use.  The figure uses an EVENT-SPACED time axis (equal width
# between consecutive critical times; tick labels carry the true times) so
# the 200 ns CZ and the μs dressing segments stay readable beside the moves.
V_MAX_UM_US = 4.0        # peak tweezer speed, μm/μs (numerically = m/s)
D_ZONE_UM = 100.0        # interaction → gate zone separation
T_EVOLVE_US = 5.0        # evolution time (dressing-on window, split by kick)
AOD_SETTLE_US = 1.0      # floor on any move (AOD settle)
TRANSIT_DY_UM = 5.0      # y-offset lane: lift 5 um, travel, drop, so the
                         # moving pair never sweeps through parked atoms

# The 2q gate is the MEASURED fixed pulse (A(t), φ(t) tables @ 1 ns on an
# 80 MHz carrier — gate_amp_and_phase.csv), identical for every two-qubit
# gate.  The fixed shape owns the gate duration, so cz_gate_time is derived
# from the table, and to_pulsedsl_tree(gate_shape=...) attaches the sampled
# waveform to zz plays ONLY (dressing/addressing keep constant envelopes).
from awg_compile import GateShape
GATE_CSV = os.path.join(os.path.dirname(__file__), "..",
                        "gate_amp_and_phase.csv")
GATE_SHAPE = GateShape.from_csv(GATE_CSV, carrier_mhz=80.0)
CZ_GATE_US = GATE_SHAPE.duration_ns / 1000.0   # 0.696 μs measured gate


def build_schedule(verbose=True, transit_dy=TRANSIT_DY_UM):
    """Compile the 2q running instance end-to-end under the module config.

    transit_dy: y-offset lane for collision-free transit; None = direct
    single-leg moves (atoms ride the AOD straight to the gate zone).

    Returns (schedule, mapper, meta): the PulseDSL Sched, the TweezerMapper
    (its ledger holds per-segment provenance), and a dict of the compile
    context (n, tau split, atom geometry) for figure builders.
    """
    fifo = "/Users/syue99/research/RISC-Q/PulseDSL/src/DSL/tmp_pulse_mmio.txt"
    if os.path.exists(fifo):
        os.remove(fifo)

    # Gate zone 100 μm from the interaction zone (walkthrough geometry config)
    import pulse_ledger
    import tweezer_mapper as tm_mod
    pulse_ledger.GATE_ZONE = (D_ZONE_UM, 0.0)
    tm_mod.GATE_ZONE = (D_ZONE_UM, 0.0)

    x = sp.Symbol("x"); T, x_val = T_EVOLVE_US, 0.7
    qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    H = sp.sin(2 * x) * q[0].Z * q[1].Z + sp.sin(2 * x) * q[0].X \
        + sp.sin(2 * x) * q[1].X

    prov = diffQCProvider()
    qs_c = QSystem(); _ = [Qubit(qs_c) for _ in range(2)]
    qs_c.add_evolution(H.set_parameterizedHam({"x": x_val}), T)
    prov.compile(qs_c, "quera", "Aquila", "rydberg2d", tol=0.1, verbose=0)
    n, sol_gvars, boxes, _e, _ = prov.prog
    mapper = TweezerMapper(n_qubits=n, sol_gvars=sol_gvars, boxes=boxes,
                           ramp_time=AOD_SETTLE_US,
                           cz_gate_time=CZ_GATE_US,
                           aod_vmax=V_MAX_UM_US,
                           transit_dy=transit_dy)

    np.random.seed(1)
    programs = observable_program_generator(
        H, T, n_sample=1, n_repetition=1, diff_var="x", value=x_val)
    H_list = programs[0][0][0]
    tau_us = float(H_list[0][1])         # sampled insertion time (first seg)

    logical, _, _ = mapper.map_hlist_tree(H_list, T=T)
    physical = pc.to_physical(logical, n)

    if verbose:
        print("\n=== PHYSICAL op-tree (6 AOM/AOD channels) ===")
        print(pt.pretty(physical))

    from PulseDSL_py import Channels, Schedule, PulseLib
    from PulseDSL_py.pulselib import set_platform
    import PulseDSL_py.schedule as dsl_schedule

    dsl_schedule.sched = None            # fresh session (global singleton)
    ch, reg = Channels(pc.NUM_PHYSICAL_CHANNELS)
    schedule = Schedule()
    set_platform(PulseLib.Rydberg)
    aod_ch = ch[pc.TRANSPORT_AOD]

    if verbose:
        print("\n=== Translating to PulseDSL (COMB/Play) and RUN ===")
    to_pulsedsl_tree(physical, ch, aod_ch, run=True, gate_shape=GATE_SHAPE)

    meta = {
        "n": n,
        "T_us": T,
        "tau_us": tau_us,
        "x_val": x_val,
        "interaction_positions": mapper.interaction_positions(),
        "gate_zone": (D_ZONE_UM, 0.0),
        "R_cz": mapper.R_cz,
        "transit_dy": transit_dy,
        "v_max": V_MAX_UM_US,
        "cz_gate_us": CZ_GATE_US,
    }
    return schedule, mapper, meta


def main():
    schedule, mapper, meta = build_schedule(verbose=True)

    print("\n=== schedule.view() — channels:", pc.CHANNEL_NAMES, "===")
    schedule.view()

    # ── AWG compile: schedule → per-channel sample arrays ─────────────────────
    # The timeline is ~10 ms while the CZ is 200 ns, so one sample rate cannot
    # serve the whole figure: the overview is compiled at 10 MS/s (carriers
    # unresolved — envelope view), and the zoom panels re-sample their windows
    # at 1 GS/s.
    from awg_compile import (compile_waveforms, waveform_summary, ChirpTone,
                             _fallback_waveform)

    t_ns, waves = compile_waveforms(schedule,
                                    n_channels=pc.NUM_PHYSICAL_CHANNELS,
                                    dt_ns=100.0)
    print("=== AWG waveforms (overview grid, 10 MS/s) ===")
    print(waveform_summary(t_ns, waves, names=pc.CHANNEL_NAMES))

    rows = schedule._Sched__schedule

    def sample_window(ch_idx, t_lo, t_hi, dt=1.0):
        """Re-sample one channel's schedule window at fine dt (ns)."""
        t = np.arange(t_lo, t_hi, dt)
        w = np.zeros(len(t), dtype=complex)
        for e in rows[ch_idx]:
            t0, t1 = float(e._ScheduleEntry__t0), float(e._ScheduleEntry__t1)
            m = (t >= t0) & (t < t1)
            if not m.any():
                continue
            p = e._ScheduleEntry__pulse
            fn = p.waveform if p.waveform is not None else _fallback_waveform(p)
            w[m] += fn(t[m] - t0)
        return t, w

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_ch = pc.NUM_PHYSICAL_CHANNELS

    # ── event-spaced time axis ───────────────────────────────────────────────
    # Critical times = every entry boundary on every channel.  Each
    # inter-event interval gets EQUAL width on the axis; tick labels carry
    # the true times, so duration ratios are read off the labels, not the
    # spacing — this keeps the 200 ns CZ visible beside a 47 us move.
    bset = {0.0}
    for row in rows[:n_ch]:
        for e in row:
            bset.add(float(e._ScheduleEntry__t0))
            bset.add(float(e._ScheduleEntry__t1))
    bounds = []
    for b in sorted(bset):
        if not bounds or b - bounds[-1] > 1.0:   # merge <1 ns duplicates
            bounds.append(b)
    K = 160
    t_real = np.concatenate([
        np.linspace(bounds[i], bounds[i + 1], K, endpoint=False)
        for i in range(len(bounds) - 1)])
    t_warp = np.concatenate([
        i + np.linspace(0.0, 1.0, K, endpoint=False)
        for i in range(len(bounds) - 1)])

    def env_at(ch_idx, t):
        """|A| envelope of one channel on an arbitrary time grid (ns)."""
        w = np.zeros(len(t), dtype=complex)
        for e in rows[ch_idx]:
            t0 = float(e._ScheduleEntry__t0)
            t1 = float(e._ScheduleEntry__t1)
            m = (t >= t0) & (t < t1)
            if not m.any():
                continue
            p_ = e._ScheduleEntry__pulse
            fn = p_.waveform if p_.waveform is not None \
                else _fallback_waveform(p_)
            w[m] += fn(t[m] - t0)
        return np.abs(w)

    fig, (ax_f, ax_g) = plt.subplots(2, 1, sharex=True, figsize=(13, 7.5),
                                     height_ratios=[1.2, 1.0])

    # ── panel 1: AOD transport frequency, X and Y combined ───────────────────
    # Per-ENTRY plotting: both atoms ride the same ramp as co-temporal comb
    # tones, so each tone gets its own curve (a single per-channel array
    # would overwrite one tone with the other).
    warp = lambda t: np.interp(t, bounds, np.arange(len(bounds), dtype=float))
    for ch_idx, color, lab in ((pc.TRANSPORT_AOD_X, "C0", "AOD X tones"),
                               (pc.TRANSPORT_AOD_Y, "C1", "AOD Y tones")):
        seen = False
        for e in rows[ch_idx]:
            wf = e._ScheduleEntry__pulse.waveform
            t0 = float(e._ScheduleEntry__t0)
            t1 = float(e._ScheduleEntry__t1)
            if isinstance(wf, ChirpTone):
                tt = np.linspace(0.0, wf.duration_ns, 300)
                ax_f.plot(warp(t0 + tt), wf.instantaneous_freq_mhz(tt),
                          lw=1.5, color=color, alpha=0.9,
                          label=None if seen else lab)
                lin = wf.f0_mhz + (wf.f1_mhz - wf.f0_mhz) * tt / wf.duration_ns
                ax_f.plot(warp(t0 + tt), lin, lw=0.7, ls="--", color=color,
                          alpha=0.4)
                seen = True
            elif getattr(wf, "freq_mhz", 0.0):
                ax_f.plot(warp(np.array([t0, t1])), [wf.freq_mhz] * 2,
                          lw=1.5, color=color, alpha=0.9,
                          label=None if seen else lab)
                seen = True
    ax_f.plot([], [], lw=0.7, ls="--", color="gray", label="linear (old)")
    ax_f.legend(fontsize=8, loc="center right")
    ax_f.set_ylabel("AOD frequency (MHz)", fontsize=9)
    ax_f.tick_params(labelsize=8)

    # the physical channel signal: both tones summed on one modulator —
    # the multi-tone beat envelope of the X comb (right axis, gray)
    ax_b = ax_f.twinx()
    ax_b.fill_between(t_warp, env_at(pc.TRANSPORT_AOD_X, t_real), 0,
                      color="gray", alpha=0.18, lw=0)
    ax_b.plot([], [], color="gray", alpha=0.5, lw=4,
              label="|A| of X comb (2-tone beat)")
    ax_b.set_ylabel("|A| (X comb)", fontsize=8, color="gray")
    ax_b.tick_params(labelsize=7, colors="gray")
    ax_b.set_ylim(bottom=0)
    ax_b.legend(fontsize=7, loc="lower right")
    move_us = sorted({wf.duration_ns * 1e-3
                      for ch in (pc.TRANSPORT_AOD_X, pc.TRANSPORT_AOD_Y)
                      for e in rows[ch]
                      for wf in [e._ScheduleEntry__pulse.waveform]
                      if isinstance(wf, ChirpTone)})
    d_big = max(move_us) * V_MAX_UM_US * 8.0 / 15.0            # um
    a_pk_si = 1e6 * 128.0 * V_MAX_UM_US ** 2 / (
        45.0 * np.sqrt(3.0) * d_big)                           # m/s^2
    ax_f.set_title(
        f"transport: {d_big:.0f} um min-jerk hop, v_pk = "
        f"{V_MAX_UM_US:.0f} m/s -> {max(move_us):.0f} us travel + "
        f"{min(move_us):.1f} us lift/drop ({TRANSIT_DY_UM:.0f} um lane); "
        f"a_pk ~ {a_pk_si:.1e} m/s^2", fontsize=9)

    # ── panel 2: all drive channels, |A| envelope, one color each ────────────
    for ch_idx, color, lab in ((pc.DRESSING_AOM, "C2", "dressing (ZZ)"),
                               (pc.GATE_AOM, "C3", "gate zone (CZ)"),
                               (pc.ADDR_RABI, "C4", "addressing Rabi"),
                               (pc.ADDR_DET, "C5", "addressing detuning")):
        ax_g.plot(t_warp, env_at(ch_idx, t_real), lw=1.0, color=color,
                  label=lab)
    ax_g.legend(fontsize=8, loc="center right")
    ax_g.set_ylabel("|A| envelope", fontsize=9)
    ax_g.tick_params(labelsize=8)
    ax_g.set_title("drive channels", fontsize=9)

    # event-spaced ticks: one per critical time, labeled with the true time
    def fmt_us(b_ns):
        v = b_ns * 1e-3
        return f"{v:.4g}" if v < 100 else f"{v:.1f}"
    ax_g.set_xticks(np.arange(len(bounds)))
    ax_g.set_xticklabels([fmt_us(b) for b in bounds], rotation=45,
                         ha="right", fontsize=7)
    ax_g.set_xlim(0, len(bounds) - 1)
    ax_g.set_xlabel("t (us) — event-spaced axis: equal width per interval, "
                    "labels are true times", fontsize=9)
    for ax in (ax_f, ax_g):
        for i in range(len(bounds)):
            ax.axvline(i, color="gray", lw=0.4, alpha=0.25)

    fig.suptitle("One 2q PSR branch -- transport chirps vs drive channels "
                 "(event-spaced time). CZ = 200 ns, dressing ~ us, "
                 f"travel ~ {max(move_us):.0f} us", fontsize=11)
    out = os.path.join(os.path.dirname(__file__), "awg_waveforms_2q.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"\nSaved waveform figure: {out}")
    print("move durations (us):", [f"{m:.2f}" for m in move_us])


if __name__ == "__main__":
    main()
