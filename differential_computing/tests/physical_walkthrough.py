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
# CZ gate ~200 ns; dressing segments on the order of 1–10 μs (T = 5 μs here,
# split by the kick's sampled τ); transport is SPEED-limited at v_max = 4 m/s
# (Cicali et al.'s Eq.-(6) min-jerk profile, whose peak speed is (15/8)·d/τ,
# so a zone hop of d takes τ = (15/8)·d/v_max and needs peak acceleration
# a_pk = 10d/(√3 τ²) = (128/45√3)·v_max²/d — the accel is derived, not a cap).
# At 4 m/s a 100 μm hop would take only 47 μs; the 100–500 μs move budget
# implies a zone separation of ~200–1000 μm, so the walkthrough uses 500 μm:
# τ ≈ 235 μs per hop, a_pk ≈ 1.64·16/500e-6 ≈ 5.3e4 m/s².  Transport still
# dominates the schedule: ~2 orders over dressing, ~3 over the gate.
V_MAX_UM_US = 4.0        # peak tweezer speed, μm/μs (numerically = m/s)
D_ZONE_UM = 500.0        # interaction → gate zone separation
T_EVOLVE_US = 5.0        # evolution time (dressing-on window, split by kick)
CZ_GATE_US = 0.2         # 200 ns two-qubit gate
AOD_SETTLE_US = 1.0      # floor on any move (AOD settle)


def main():
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
                           aod_vmax=V_MAX_UM_US)

    np.random.seed(1)
    programs = observable_program_generator(
        H, T, n_sample=1, n_repetition=1, diff_var="x", value=x_val)
    H_list = programs[0][0][0]

    logical, _, _ = mapper.map_hlist_tree(H_list, T=T)
    physical = pc.to_physical(logical, n)

    print("\n=== PHYSICAL op-tree (6 AOM/AOD channels) ===")
    print(pt.pretty(physical))

    from PulseDSL_py import Channels, Schedule, PulseLib
    from PulseDSL_py.pulselib import set_platform

    ch, reg = Channels(pc.NUM_PHYSICAL_CHANNELS)
    schedule = Schedule()
    set_platform(PulseLib.Rydberg)
    aod_ch = ch[pc.TRANSPORT_AOD]

    print("\n=== Translating to PulseDSL (COMB/Play) and RUN ===")
    to_pulsedsl_tree(physical, ch, aod_ch, run=True)

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

    t_us = t_ns * 1e-3

    # locate the CZ gate window (shortest GATE_AOM entry) for annotation
    gate_entries = [(float(e._ScheduleEntry__t0), float(e._ScheduleEntry__t1))
                    for e in rows[pc.GATE_AOM]]
    cz_t0, cz_t1 = min(gate_entries, key=lambda w: w[1] - w[0])

    fig, (ax_f, ax_g) = plt.subplots(2, 1, sharex=True, figsize=(12, 7),
                                     height_ratios=[1.2, 1.0])

    # ── panel 1: AOD transport frequency, X and Y combined ───────────────────
    move_us = []
    for ch_idx, color, lab in ((pc.TRANSPORT_AOD_X, "C0", "AOD X"),
                               (pc.TRANSPORT_AOD_Y, "C1", "AOD Y")):
        seen = False
        for e in rows[ch_idx]:
            wf = e._ScheduleEntry__pulse.waveform
            if not isinstance(wf, ChirpTone):
                continue
            t0 = float(e._ScheduleEntry__t0)
            tt = np.linspace(0.0, wf.duration_ns, 400)
            ax_f.plot((t0 + tt) * 1e-3, wf.instantaneous_freq_mhz(tt),
                      lw=1.6, color=color, label=None if seen else lab)
            seen = True
            lin = wf.f0_mhz + (wf.f1_mhz - wf.f0_mhz) * tt / wf.duration_ns
            ax_f.plot((t0 + tt) * 1e-3, lin, lw=0.8, ls="--", color=color,
                      alpha=0.5)
            if ch_idx == pc.TRANSPORT_AOD_X:
                move_us.append(wf.duration_ns * 1e-3)
        if not seen:   # only constant hold tones on this axis
            for e in rows[ch_idx]:
                wf = e._ScheduleEntry__pulse.waveform
                f0 = getattr(wf, "freq_mhz", None)
                if not f0:
                    continue
                t0 = float(e._ScheduleEntry__t0)
                t1 = float(e._ScheduleEntry__t1)
                ax_f.plot([t0 * 1e-3, t1 * 1e-3], [f0, f0], lw=1.6,
                          color=color, label=None if seen else lab)
                seen = True
    ax_f.plot([], [], lw=0.8, ls="--", color="gray", label="linear (old)")
    ax_f.legend(fontsize=8, loc="center right")
    ax_f.set_ylabel("AOD frequency (MHz)", fontsize=9)
    ax_f.tick_params(labelsize=8)
    d_big = max(move_us) * V_MAX_UM_US * 8.0 / 15.0            # um
    a_pk_si = 1e6 * 128.0 * V_MAX_UM_US ** 2 / (
        45.0 * np.sqrt(3.0) * d_big)                           # m/s^2
    ax_f.set_title(
        f"transport: {d_big:.0f} um min-jerk hop, v_pk = "
        f"{V_MAX_UM_US:.0f} m/s -> {max(move_us):.0f} us per move "
        f"(needs a_pk ~ {a_pk_si:.1e} m/s^2)", fontsize=9)

    # ── panel 2: all gate/drive channels, |A| envelope, one color each ───────
    gate_chs = ((pc.DRESSING_AOM, "C2", "dressing (ZZ)"),
                (pc.GATE_AOM, "C3", "gate zone (CZ)"),
                (pc.ADDR_RABI, "C4", "addressing Rabi"),
                (pc.ADDR_DET, "C5", "addressing detuning"))
    for ch_idx, color, lab in gate_chs:
        ax_g.plot(t_us, np.abs(waves[ch_idx]), lw=1.0, color=color,
                  label=lab)
    ax_g.legend(fontsize=8, loc="center right")
    ax_g.set_ylabel("|A| envelope", fontsize=9)
    ax_g.set_xlabel("t (us)", fontsize=9)
    ax_g.tick_params(labelsize=8)
    ax_g.set_title("drive channels (10 MS/s envelope view; us/ns pulses "
                   "are subpixel -- see annotations)", fontsize=9)

    # subpixel events: annotate the 200 ns CZ and the us dressing segments
    ax_g.annotate("CZ 200 ns", xy=(cz_t0 * 1e-3, 0.80),
                  xycoords=("data", "axes fraction"), color="C3", fontsize=8,
                  ha="left", xytext=(8, 0), textcoords="offset points",
                  arrowprops=dict(arrowstyle="-", color="C3", lw=0.8))
    for e in rows[pc.DRESSING_AOM]:
        t0 = float(e._ScheduleEntry__t0)
        t1 = float(e._ScheduleEntry__t1)
        ax_g.annotate(f"dressing {(t1 - t0) * 1e-3:.1f} us",
                      xy=(t0 * 1e-3, 0.75),
                      xycoords=("data", "axes fraction"), color="C2",
                      fontsize=8, ha="left", xytext=(4, 8),
                      textcoords="offset points",
                      arrowprops=dict(arrowstyle="-", color="C2", lw=0.8))

    fig.suptitle("One 2q PSR branch -- transport chirps vs drive channels. "
                 "move ~ 100s of us >> dressing ~ us >> gate = 200 ns",
                 fontsize=11)
    out = os.path.join(os.path.dirname(__file__), "awg_waveforms_2q.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"\nSaved waveform figure: {out}")
    print("move durations (us):", [f"{m:.1f}" for m in move_us])


if __name__ == "__main__":
    main()
