"""
dsl_tree_walkthrough.py — Step 4: one PSR branch end-to-end through the new
PulseDSL op-tree scheduler.

Pipeline:
    compile (2q) -> one PSR branch H_list
    -> TweezerMapper.map_hlist_tree  (native op-tree)
    -> to_pulsedsl_tree              (IR -> PulseDSL SEQ/PARA/Play -> RUN)
    -> schedule.view()               (per-channel timeline)

Then assert the timeline matches the position-segmented PARA structure:
  - interaction-zone plays (detuning/rabi/dressing) start together (concurrent),
  - the ZZ play is serialized AFTER them (separated by an AOD ramp),
  - segments run back-to-back (seg2 starts after the kick segment ends).

Run manually (touches the global Schedule singleton + the MMIO FIFO):
    conda run -n qec_pg python differential_computing/tests/dsl_tree_walkthrough.py
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


def build_2q_branch(T=0.5, x_val=0.7, seed=1):
    """Compile a 2q model and return (mapper, one H_list branch)."""
    x = sp.Symbol("x")
    qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    H = sp.sin(2 * x) * q[0].Z * q[1].Z + sp.sin(2 * x) * q[0].X \
        + sp.sin(2 * x) * q[1].X

    prov = diffQCProvider()
    qs_c = QSystem(); _ = [Qubit(qs_c) for _ in range(2)]
    qs_c.add_evolution(H.set_parameterizedHam({"x": x_val}), T)
    prov.compile(qs_c, "quera", "Aquila", "rydberg2d", tol=0.1, verbose=0)
    n_sites, sol_gvars, boxes, _edges, _ = prov.prog
    mapper = TweezerMapper(n_qubits=n_sites, sol_gvars=sol_gvars, boxes=boxes,
                           ramp_time=0.01)

    np.random.seed(seed)
    programs = observable_program_generator(
        H, T, n_sample=1, n_repetition=1, diff_var="x", value=x_val)
    H_list = programs[0][0][0]
    return mapper, H_list, n_sites


def main():
    # Fresh FIFO so RUN's MMIO writes never block on a stale pipe.
    fifo = "/Users/syue99/research/RISC-Q/PulseDSL/src/DSL/tmp_pulse_mmio.txt"
    if os.path.exists(fifo):
        os.remove(fifo)

    mapper, H_list, n = build_2q_branch()
    tree, log, ledger = mapper.map_hlist_tree(H_list, T=0.5)

    print("\n=== Native op-tree (pulse_tree IR) ===")
    print(pt.pretty(tree))

    from PulseDSL_py import Channels, Schedule, PulseLib
    from PulseDSL_py.pulselib import set_platform

    n_channels = 2 * n + 3      # det(n) + rabi(n) + dressing + zz + aod
    ch, reg = Channels(n_channels)
    schedule = Schedule()
    set_platform(PulseLib.Rydberg)
    aod_ch = ch[n_channels - 1]

    print("\n=== Translating to PulseDSL and RUN ===")
    to_pulsedsl_tree(tree, ch, aod_ch, run=True)

    print("\n=== PulseDSL schedule.view() ===")
    schedule.view()

    # ── Timeline assertions ──────────────────────────────────────────────────
    seq = schedule.return_pulse_sequence_by_channel()

    def starts(chan):
        return [e[0]["ns"] for e in seq[str(chan)]]

    def ends(chan):
        return [e[0]["ns"] + e[1]["ns"] for e in seq[str(chan)]]

    det0, rabi0, dress = 0, n, 2 * n         # ch indices: detuning q0, rabi q0, dressing
    zz, aod = 2 * n + 1, 2 * n + 2

    # 1. interaction-zone plays in seg0 start together (concurrency)
    s_det = starts(det0)[0]
    s_rabi = starts(rabi0)[0]
    s_dress = starts(dress)[0]
    assert s_det == s_rabi == s_dress == 0, \
        f"interaction plays not concurrent at t=0: {s_det},{s_rabi},{s_dress}"
    print(f"[OK] seg0 detuning/rabi/dressing all start at t={s_det} ns")

    # 2. ZZ in seg0 is serialized after the interaction plays + an AOD ramp
    e_int = ends(det0)[0]
    s_zz0 = starts(zz)[0]
    assert s_zz0 >= e_int, f"ZZ ({s_zz0}) not after interaction plays ({e_int})"
    print(f"[OK] seg0 ZZ starts at t={s_zz0} ns, after interaction end {e_int} ns")

    # 3. segments run back-to-back: seg2's interaction plays start after the
    #    kick segment (the 2nd ZZ entry) ends.
    s_det_seg2 = starts(det0)[1]
    e_zz_kick = ends(zz)[1]
    assert s_det_seg2 >= e_zz_kick, \
        f"seg2 ({s_det_seg2}) overlaps kick ({e_zz_kick})"
    print(f"[OK] seg2 starts at t={s_det_seg2} ns, after kick end {e_zz_kick} ns")

    # 4. the AOD channel carries the transport moves (>=3: seg0, kick out+back)
    assert len(seq[str(aod)]) >= 3, f"expected AOD moves, got {len(seq[str(aod)])}"
    print(f"[OK] AOD channel has {len(seq[str(aod)])} transport moves")

    print("\nAll walkthrough timeline checks passed.")


if __name__ == "__main__":
    main()
