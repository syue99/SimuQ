"""
test_pulsedsl_tree.py — Step 3: to_pulsedsl_tree IR → PulseDSL translation.

These tests use run=False so they only *build* the PulseDSL op-tree (no RUN,
no MMIO writes, no Schedule needed) and assert it mirrors the IR one-to-one:
SEQ/PARA nesting, Play channels, μs→ns durations, amplitude/phase pass-through,
and AOD→Play(Sine) / Delay routing onto the AOD channel.

End-to-end RUN + timeline checks live in dsl_tree_walkthrough.py (run manually,
since RUN touches the global Schedule singleton and the MMIO FIFO).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/Users/syue99/research/RISC-Q/PulseDSL/src/DSL/")

import pulse_tree as pt
from simuq.braket.diffQC_provider import to_pulsedsl_tree
from PulseDSL_py import Channels
from PulseDSL_py.core import Shape


def _ir():
    # SEQ( SEQ( PARA(det@ch0, rabi@ch2), AOD, PARA(zz@ch5) ) )
    return pt.Seq([
        pt.Seq([
            pt.Para([pt.PlayNode(0, 7.0, 0.2, kind="detuning"),
                     pt.PlayNode(2, 1.9, 0.2, phase=0.1, kind="rabi")]),
            pt.AodNode([(1000.0, 1000.0), (0.0, 0.0)], 0.01),
            pt.Para([pt.PlayNode(5, -2.5, 0.2, kind="zz")]),
        ]),
    ])


def test_translate_mirrors_ir_structure():
    ch, _ = Channels(7)
    aod_ch = ch[6]
    dsl = to_pulsedsl_tree(_ir(), ch, aod_ch, run=False)

    assert dsl.kind == "seq"
    seg = dsl.children[0]
    assert seg.kind == "seq"
    para0, aod, para1 = seg.children

    assert para0.kind == "para" and len(para0.children) == 2
    assert all(c.kind == "play" for c in para0.children)
    assert para1.kind == "para" and para1.children[0].kind == "play"

    # AOD barrier becomes a Play(Sine) on the AOD channel
    assert aod.kind == "play"
    assert aod.pulse.shape == Shape.Sine
    assert aod.ch is aod_ch


def test_play_fields_and_unit_conversion():
    ch, _ = Channels(7)
    aod_ch = ch[6]
    dsl = to_pulsedsl_tree(_ir(), ch, aod_ch, run=False)
    p_det, p_rabi = dsl.children[0].children[0].children

    # μs → ns (× 1000)
    assert p_det.pulse.duration == 200
    # amplitude / phase pass through; placeholder shape is Constant
    assert p_det.pulse.amplitude == 7.0
    assert p_det.pulse.shape == Shape.Constant
    assert abs(p_rabi.pulse.phase - 0.1) < 1e-12
    # channels routed by index
    assert p_det.ch is ch[0]
    assert p_rabi.ch is ch[2]


def test_delay_routes_to_aod_channel():
    ch, _ = Channels(7)
    aod_ch = ch[6]
    dsl = to_pulsedsl_tree(pt.Seq([pt.DelayNode(0.05)]), ch, aod_ch, run=False)
    d = dsl.children[0]
    assert d.kind == "delay"
    assert d.pulse.duration == 50          # 0.05 μs → 50 ns
    assert d.ch is aod_ch


def test_empty_blocks_translate():
    ch, _ = Channels(7)
    aod_ch = ch[6]
    dsl = to_pulsedsl_tree(pt.Seq([pt.Seq([]), pt.Para([])]), ch, aod_ch,
                           run=False)
    assert dsl.kind == "seq"
    assert dsl.children[0].kind == "seq" and dsl.children[0].children == []
    assert dsl.children[1].kind == "para" and dsl.children[1].children == []


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} pulsedsl_tree tests passed.")
