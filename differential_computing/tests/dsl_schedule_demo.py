"""
DSL schedule rendering for a single-trial 2-atom walkthrough.

Builds a 2-qubit H(x) = sin(2x)·(Z0Z1 + X0 + X1), runs one PSR trial
through TweezerMapper, then emits the schedule ops via to_pulsedsl_simple
into a PulseDSL Schedule context.

Run standalone:
    conda run -n qec_pg python differential_computing/tests/dsl_schedule_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/Users/syue99/research/RISC-Q/PulseDSL/src/DSL/")

import numpy as np
import sympy as sp

from simuq import QSystem, Qubit
from simuq.braket.diffQC_provider import diffQCProvider, to_pulsedsl_simple
from observable_program_generator import observable_program_generator


def main():
    # ── Build 2-qubit Hamiltonian ────────────────────────────────────────
    x = sp.Symbol("x")
    qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    J = sp.sin(2 * x)
    H_param = J * q[0].Z * q[1].Z + J * q[0].X + J * q[1].X

    T = 0.5
    x_val = 1.0

    # ── Compile ──────────────────────────────────────────────────────────
    prov = diffQCProvider()
    qs_c = QSystem(); q_c = [Qubit(qs_c) for _ in range(2)]
    H_eval = H_param.set_parameterizedHam({"x": x_val})
    qs_c.add_evolution(H_eval, T)
    prov.compile(qs_c, "quera", "Aquila", "rydberg2d", tol=0.1, verbose=0)

    # ── Single trial ─────────────────────────────────────────────────────
    np.random.seed(42)
    programs = observable_program_generator(
        H_param, T, n_sample=1, n_repetition=1, diff_var="x", value=x_val,
    )
    prov.run(programs, None, T=T, backend="hardware", verbose=0)

    # ── Show raw ops ─────────────────────────────────────────────────────
    branch_ops_list, ugrad, n_rep = prov._branch_ops[0]
    ops = branch_ops_list[0]  # first branch

    print("=== Raw schedule ops (branch 0) ===")
    for i, op in enumerate(ops):
        dur_ns = int(op["duration"] * 1000)
        if op["op"] == "aod":
            print(f"  [{i:2d}] AOD   amp={op['amplitude']:.3f}  dur={dur_ns} ns")
        elif op["op"] == "play":
            print(f"  [{i:2d}] PLAY  ch={op['channel']}  amp={op['amplitude']:.6f}  "
                  f"phase={op.get('phase', 0):.4f}  dur={dur_ns} ns")
        elif op["op"] == "delay":
            print(f"  [{i:2d}] DELAY dur={dur_ns} ns")

    # ── Emit via PulseDSL ────────────────────────────────────────────────
    print("\n=== PulseDSL emission ===")
    from PulseDSL_py import Channels, Schedule, PulseLib
    from PulseDSL_py.pulselib import set_platform

    # n_qubits=2: channels 0,1 = detuning; 2,3 = Rabi; 4 = dressing; 5+ = ZZ
    # Plus 1 AOD channel
    n_channels = 8
    ch, reg = Channels(n_channels)
    schedule = Schedule()
    set_platform(PulseLib.Rydberg)

    aod_ch = ch[n_channels - 1]  # last channel for AOD

    to_pulsedsl_simple(ops, ch, aod_ch)

    print(f"Schedule emitted: {len(ops)} ops → PulseDSL")
    print("(schedule.view() available in notebook/interactive context)")


if __name__ == "__main__":
    main()
