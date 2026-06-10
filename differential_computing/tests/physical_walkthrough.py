"""
physical_walkthrough.py — one PSR branch through the PHYSICAL channel stack.

    compile (2q) -> map_hlist_tree (logical, per-qubit)
    -> physical_channels.to_physical (5 fixed AOM/AOD channels, COMBs)
    -> to_pulsedsl_tree (CombNode -> COMB, PlayNode -> Play) -> RUN
    -> schedule.view()

Shows per-qubit detuning/Rabi consolidated into ADDR_DET / ADDR_RABI tone combs,
dressing on DRESSING_AOM, ZZ on GATE_AOM, transport on TRANSPORT_AOD.

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

    print("\n=== PHYSICAL op-tree (5 AOM/AOD channels) ===")
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


if __name__ == "__main__":
    main()
