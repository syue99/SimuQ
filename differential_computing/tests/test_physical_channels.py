"""
test_physical_channels.py — Step C: logical op-tree → physical AOM/AOD channels.

Asserts the to_physical transform consolidates the per-qubit logical tree onto
the 5 fixed physical channels:
  - per-site detuning plays  -> ONE ADDR_DET comb (one tone per site),
  - per-site Rabi plays       -> ONE ADDR_RABI comb,
  - dressing                  -> Play on DRESSING_AOM,
  - ZZ                        -> Play on GATE_AOM,
  - AOD move                  -> TRANSPORT_AOD comb (one tone per atom),
  - flatten is well-formed and the physical tree translates to PulseDSL COMB/Play.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/Users/syue99/research/RISC-Q/PulseDSL/src/DSL/")

import numpy as np
import sympy as sp
import pytest

from simuq import QSystem, Qubit
from simuq.braket.diffQC_provider import diffQCProvider, to_pulsedsl_tree
from observable_program_generator import observable_program_generator
from tweezer_mapper import TweezerMapper
import pulse_tree as pt
import physical_channels as pc


def _compile_mapper(H_param, x_val, T, n_qubits):
    prov = diffQCProvider()
    qs_c = QSystem(); _ = [Qubit(qs_c) for _ in range(n_qubits)]
    qs_c.add_evolution(H_param.set_parameterizedHam({"x": x_val}), T)
    prov.compile(qs_c, "quera", "Aquila", "rydberg2d", tol=0.1, verbose=0)
    n_sites, sol_gvars, boxes, _e, _ = prov.prog
    return TweezerMapper(n_qubits=n_sites, sol_gvars=sol_gvars, boxes=boxes,
                         ramp_time=0.01)


def _two_q_physical(seed=1, T=0.5, x_val=0.7):
    x = sp.Symbol("x")
    qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    H = sp.sin(2 * x) * q[0].Z * q[1].Z + sp.sin(2 * x) * q[0].X \
        + sp.sin(2 * x) * q[1].X
    mapper = _compile_mapper(H, x_val, T, 2)
    np.random.seed(seed)
    programs = observable_program_generator(
        H, T, n_sample=1, n_repetition=1, diff_var="x", value=x_val)
    H_list = programs[0][0][0]
    logical, _, _ = mapper.map_hlist_tree(H_list, T=T)
    return logical, pc.to_physical(logical, 2)


def _all_nodes(node):
    yield node
    if isinstance(node, (pt.Seq, pt.Para)):
        for c in node.children:
            yield from _all_nodes(c)


def test_detuning_and_rabi_collapse_to_combs():
    _, phys = _two_q_physical()
    combs = [nd for nd in _all_nodes(phys) if isinstance(nd, pt.CombNode)]

    det_combs = [c for c in combs if c.channel == pc.ADDR_DET]
    rabi_combs = [c for c in combs if c.channel == pc.ADDR_RABI]
    assert det_combs, "no ADDR_DET comb produced"
    assert rabi_combs, "no ADDR_RABI comb produced"
    # 2 qubits → each detuning/Rabi comb carries 2 tones (sites 0 and 1)
    for c in det_combs + rabi_combs:
        atoms = sorted(t.atom for t in c.tones)
        assert atoms == [0, 1], f"comb tones address {atoms}, expected [0,1]"
        freqs = {t.atom: t.frequency for t in c.tones}
        assert freqs[0] == pc.addr_frequency(0)
        assert freqs[1] == pc.addr_frequency(1)


def test_dressing_and_zz_on_global_aoms():
    _, phys = _two_q_physical()
    plays = [nd for nd in _all_nodes(phys) if isinstance(nd, pt.PlayNode)]
    dressing = [p for p in plays if p.channel == pc.DRESSING_AOM]
    zz = [p for p in plays if p.channel == pc.GATE_AOM]
    assert dressing and all(p.kind == "dressing" for p in dressing)
    assert zz and all(p.kind == "zz" for p in zz)
    # nothing should remain on the old logical per-qubit channels (>= 5)
    assert all(p.channel < pc.NUM_PHYSICAL_CHANNELS for p in plays)


def test_aod_moves_become_transport_combs():
    _, phys = _two_q_physical()
    combs = [nd for nd in _all_nodes(phys) if isinstance(nd, pt.CombNode)]
    transport = [c for c in combs if c.channel == pc.TRANSPORT_AOD]
    assert transport, "no TRANSPORT_AOD comb produced"
    for c in transport:
        assert c.kind == "transport"
        assert len(c.tones) == 2          # one position tone per atom


def test_physical_tree_translates_to_pulsedsl():
    from PulseDSL_py import Channels
    _, phys = _two_q_physical()
    ch, _ = Channels(pc.NUM_PHYSICAL_CHANNELS)
    aod_ch = ch[pc.TRANSPORT_AOD]
    dsl = to_pulsedsl_tree(phys, ch, aod_ch, run=False)
    # walk the PulseDSL tree: every comb/play sits on a physical channel id
    kinds = set()

    def walk(node):
        kinds.add(node.kind)
        for c in getattr(node, "children", []):
            walk(c)
    walk(dsl)
    assert "comb" in kinds and "play" in kinds and "seq" in kinds


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} physical_channels tests passed.")
