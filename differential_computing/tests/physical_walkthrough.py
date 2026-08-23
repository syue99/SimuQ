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


def main():
    fifo = "/Users/syue99/research/RISC-Q/PulseDSL/src/DSL/tmp_pulse_mmio.txt"
    if os.path.exists(fifo):
        os.remove(fifo)

    x = sp.Symbol("x"); T, x_val = 0.5, 0.7
    qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    H = sp.sin(2 * x) * q[0].Z * q[1].Z + sp.sin(2 * x) * q[0].X \
        + sp.sin(2 * x) * q[1].X

    prov = diffQCProvider()
    qs_c = QSystem(); _ = [Qubit(qs_c) for _ in range(2)]
    qs_c.add_evolution(H.set_parameterizedHam({"x": x_val}), T)
    prov.compile(qs_c, "quera", "Aquila", "rydberg2d", tol=0.1, verbose=0)
    n, sol_gvars, boxes, _e, _ = prov.prog
    mapper = TweezerMapper(n_qubits=n, sol_gvars=sol_gvars, boxes=boxes,
                           ramp_time=0.01)

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
    from awg_compile import compile_waveforms, waveform_summary

    t_ns, waves = compile_waveforms(schedule,
                                    n_channels=pc.NUM_PHYSICAL_CHANNELS)
    print("=== AWG waveforms (1 GS/s) ===")
    print(waveform_summary(t_ns, waves, names=pc.CHANNEL_NAMES))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(pc.NUM_PHYSICAL_CHANNELS, 1, sharex=True,
                             figsize=(11, 9))
    for ch_idx, ax in enumerate(axes):
        w = waves[ch_idx]
        ax.plot(t_ns, w.real, lw=0.6, label="I")
        ax.plot(t_ns, w.imag, lw=0.6, alpha=0.7, label="Q")
        ax.plot(t_ns, np.abs(w), lw=1.0, color="k", alpha=0.5, label="|A|")
        ax.set_ylabel(pc.CHANNEL_NAMES[ch_idx], fontsize=8, rotation=0,
                      ha="right", va="center")
        ax.tick_params(labelsize=7)
    axes[0].legend(loc="upper right", fontsize=7, ncol=3)
    axes[-1].set_xlabel("t (ns)")
    fig.suptitle("End-to-end AWG waveforms — one 2q PSR branch, "
                 "6 physical channels", fontsize=11)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "awg_waveforms_2q.png")
    fig.savefig(out, dpi=160)
    print(f"\nSaved waveform figure: {out}")


if __name__ == "__main__":
    main()
