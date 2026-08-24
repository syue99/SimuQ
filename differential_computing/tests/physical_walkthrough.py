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

    n_ch = pc.NUM_PHYSICAL_CHANNELS
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(n_ch + 2, 2, height_ratios=[1] * n_ch + [1.4, 1.6],
                          hspace=0.35, wspace=0.18)
    t_us = t_ns * 1e-3

    # locate the CZ gate window (shortest GATE_AOM entry) for markers/zoom
    gate_entries = [(float(e._ScheduleEntry__t0), float(e._ScheduleEntry__t1))
                    for e in rows[pc.GATE_AOM]]
    cz_t0, cz_t1 = min(gate_entries, key=lambda w: w[1] - w[0])
    # first dressing-on window for the us-scale zoom
    dr_t0, dr_t1 = min(((float(e._ScheduleEntry__t0),
                         float(e._ScheduleEntry__t1))
                        for e in rows[pc.DRESSING_AOM]), key=lambda w: w[0])

    # -- overview: |A| envelope per channel, x in ms --------------------------
    axes = [fig.add_subplot(gs[i, :]) for i in range(n_ch)]
    for ch_idx, ax in enumerate(axes):
        w = np.abs(waves[ch_idx])
        ax.fill_between(t_us, w, 0, color="C0", alpha=0.35, lw=0)
        ax.plot(t_us, w, lw=0.7, color="C0")
        ax.set_ylabel(pc.CHANNEL_NAMES[ch_idx], fontsize=8, rotation=0,
                      ha="right", va="center")
        ax.tick_params(labelsize=7)
        ax.set_xlim(t_us[0], t_us[-1])
        if ch_idx < n_ch - 1:
            ax.tick_params(labelbottom=False)
    axes[0].set_title("|A| envelope (10 MS/s view; carriers unresolved)",
                      fontsize=8, loc="right")
    # the 200 ns CZ is subpixel on the us axis -- mark it
    axes[pc.GATE_AOM].axvline(cz_t0 * 1e-3, color="C3", lw=1.0)
    axes[pc.GATE_AOM].annotate("CZ 200 ns (zoom below)",
                               xy=(cz_t0 * 1e-3, 0.5),
                               xycoords=("data", "axes fraction"),
                               xytext=(8, 0), fontsize=7,
                               textcoords="offset points", color="C3")
    # ... and so are the us-scale dressing segments
    for e in rows[pc.DRESSING_AOM]:
        t0 = float(e._ScheduleEntry__t0)
        t1 = float(e._ScheduleEntry__t1)
        axes[pc.DRESSING_AOM].axvline(t0 * 1e-3, color="C2", lw=1.0)
        axes[pc.DRESSING_AOM].annotate(
            f"dressing {(t1 - t0) * 1e-3:.1f} us",
            xy=(t0 * 1e-3, 0.6), xycoords=("data", "axes fraction"),
            xytext=(6, 0), fontsize=7, textcoords="offset points",
            color="C2")

    # -- transport frequency: min-jerk S-curves dominate the timeline --------
    ax_f = fig.add_subplot(gs[n_ch, :], sharex=axes[0])
    colors = {pc.TRANSPORT_AOD_X: "C0", pc.TRANSPORT_AOD_Y: "C1"}
    move_us = []
    for ch_idx, color in colors.items():
        for e in rows[ch_idx]:
            wf = e._ScheduleEntry__pulse.waveform
            if not isinstance(wf, ChirpTone):
                continue
            t0 = float(e._ScheduleEntry__t0)
            tt = np.linspace(0.0, wf.duration_ns, 400)
            ax_f.plot((t0 + tt) * 1e-3, wf.instantaneous_freq_mhz(tt),
                      lw=1.4, color=color)
            lin = wf.f0_mhz + (wf.f1_mhz - wf.f0_mhz) * tt / wf.duration_ns
            ax_f.plot((t0 + tt) * 1e-3, lin, lw=0.8, ls="--", color=color,
                      alpha=0.5)
            if ch_idx == pc.TRANSPORT_AOD_X:
                move_us.append(wf.duration_ns * 1e-3)
    ax_f.plot([], [], lw=1.4, color="C0", label="AOD X (min-jerk)")
    ax_f.plot([], [], lw=1.4, color="C1", label="AOD Y (min-jerk)")
    ax_f.plot([], [], lw=0.8, ls="--", color="gray", label="linear (old)")
    ax_f.legend(loc="center right", fontsize=7)
    ax_f.set_ylabel("transport\nf (MHz)", fontsize=8, rotation=0,
                    ha="right", va="center")
    ax_f.set_xlabel("t (us)", fontsize=9)
    ax_f.tick_params(labelsize=7)
    if move_us:
        d_big = max(move_us) * V_MAX_UM_US * 8.0 / 15.0        # um
        a_pk_si = 1e6 * 128.0 * V_MAX_UM_US ** 2 / (
            45.0 * np.sqrt(3.0) * d_big)                       # m/s^2
        ax_f.set_title(
            f"{d_big:.0f} um zone hop, v_pk = {V_MAX_UM_US:.0f} m/s -> "
            f"{max(move_us):.0f} us per move (needs a_pk ~ "
            f"{a_pk_si:.1e} m/s^2)", fontsize=8, loc="right")

    # -- zoom panels: the two faster scales ----------------------------------
    # (a) us scale: first dressing-on segment + co-played addressing combs
    ax_us = fig.add_subplot(gs[n_ch + 1, 0])
    pad = 500.0
    for ch_idx, color, lab in ((pc.DRESSING_AOM, "C2", "DRESSING |A|"),
                               (pc.ADDR_RABI, "C4", "ADDR_RABI |A|"),
                               (pc.ADDR_DET, "C5", "ADDR_DET |A|")):
        tz, wz = sample_window(ch_idx, dr_t0 - pad, dr_t1 + pad, dt=1.0)
        ax_us.plot(tz * 1e-3, np.abs(wz), lw=0.9, color=color, label=lab)
    ax_us.set_xlabel("t (us)", fontsize=8)
    ax_us.set_title(f"dressing segment: {(dr_t1 - dr_t0) * 1e-3:.2f} us "
                    "(1 GS/s)", fontsize=8)
    ax_us.legend(fontsize=6)
    ax_us.tick_params(labelsize=7)

    # (b) ns scale: the CZ pulse itself, I/Q resolved
    ax_ns = fig.add_subplot(gs[n_ch + 1, 1])
    tz, wz = sample_window(pc.GATE_AOM, cz_t0 - 150.0, cz_t1 + 150.0, dt=1.0)
    ax_ns.plot(tz - cz_t0, wz.real, lw=0.7, label="I")
    ax_ns.plot(tz - cz_t0, wz.imag, lw=0.7, alpha=0.8, label="Q")
    ax_ns.plot(tz - cz_t0, np.abs(wz), lw=1.1, color="k", alpha=0.6,
               label="|A|")
    ax_ns.set_xlabel("t - t_CZ (ns)", fontsize=8)
    ax_ns.set_title(f"CZ gate: {(cz_t1 - cz_t0):.0f} ns (1 GS/s)", fontsize=8)
    ax_ns.legend(fontsize=6, ncol=3)
    ax_ns.tick_params(labelsize=7)

    fig.suptitle("End-to-end AWG waveforms -- one 2q PSR branch. "
                 "Timescale hierarchy: move ~ 100s of us >> dressing ~ us "
                 ">> gate = 200 ns", fontsize=11)
    out = os.path.join(os.path.dirname(__file__), "awg_waveforms_2q.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"\nSaved waveform figure: {out}")
    print("move durations (us):", [f"{m:.1f}" for m in move_us])


if __name__ == "__main__":
    main()
