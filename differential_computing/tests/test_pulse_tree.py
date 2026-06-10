"""
test_pulse_tree.py — unit tests for the DSL-agnostic op-tree IR.

Validates that:
  1. Each leaf's to_op() matches the canonical tweezer_mapper._op_* dict exactly.
  2. flatten() reproduces a flat op list in depth-first, left-to-right order.
  3. Para/Seq nesting does not change the flattened ordering (structure is
     timing-only; it carries no ops of its own).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import pulse_tree as pt
from tweezer_mapper import _op_aod, _op_play, _op_delay


def test_play_node_matches_op_play():
    node = pt.PlayNode(channel=3, amplitude=1.25, duration=0.5, phase=0.7,
                       kind="rabi")
    assert node.to_op() == _op_play(channel_idx=3, amplitude=1.25,
                                    duration=0.5, phase=0.7)
    # kind is metadata only — never leaks into the flat op dict
    assert "kind" not in node.to_op()


def test_delay_node_matches_op_delay():
    node = pt.DelayNode(duration=0.42)
    assert node.to_op() == _op_delay(0.42)


def test_aod_node_matches_op_aod():
    positions = [(0.0, 0.0), (3.0, 4.0)]
    node = pt.AodNode(positions=positions, ramp_time=0.01)
    assert node.to_op() == _op_aod(positions, 0.01)


def test_flatten_order_seq_of_para():
    # SEQ( PARA(p0, p1), aod, PARA(p2) )  -> [p0, p1, aod, p2]  (left-to-right)
    p0 = pt.PlayNode(0, 1.0, 0.5, kind="detuning")
    p1 = pt.PlayNode(2, 0.8, 0.5, kind="rabi")
    aod = pt.AodNode([(1000.0, 1000.0)], 0.01)
    p2 = pt.PlayNode(5, 2.0, 0.5, kind="zz")
    tree = pt.Seq([pt.Para([p0, p1]), aod, pt.Para([p2])])

    flat = pt.flatten(tree)
    assert flat == [p0.to_op(), p1.to_op(), aod.to_op(), p2.to_op()]


def test_flatten_ignores_block_nesting():
    # Two trees with the same leaf order but different Seq/Para nesting
    # must flatten identically — blocks are timing-only.
    leaves = [pt.PlayNode(i, float(i), 0.3, kind="detuning") for i in range(4)]
    nested = pt.Seq([pt.Para([leaves[0], leaves[1]]),
                     pt.Seq([leaves[2], leaves[3]])])
    flat_seq = pt.Seq(list(leaves))
    assert pt.flatten(nested) == pt.flatten(flat_seq)


def test_flatten_empty_and_none():
    assert pt.flatten(None) == []
    assert pt.flatten(pt.Seq([])) == []
    assert pt.flatten(pt.Para([])) == []


def test_builders_add_chaining():
    seq = pt.Seq().add(pt.PlayNode(0, 1.0, 0.5)).add(pt.DelayNode(0.1))
    para = pt.Para().add(pt.PlayNode(1, 2.0, 0.5))
    assert len(seq.children) == 2 and len(para.children) == 1


def test_comb_node_to_op():
    comb = pt.CombNode(
        channel=1,
        tones=[pt.Tone(atom=0, frequency=80.0, amplitude=7.0, phase=0.0),
               pt.Tone(atom=1, frequency=90.0, amplitude=7.0, phase=0.2)],
        duration=0.25,
        kind="detuning",
    )
    op = comb.to_op()
    assert op["op"] == "comb" and op["channel"] == 1 and op["duration"] == 0.25
    assert len(op["tones"]) == 2
    assert op["tones"][0] == {"atom": 0, "frequency": 80.0,
                              "amplitude": 7.0, "phase": 0.0}
    # kind is metadata only — not in the op dict
    assert "kind" not in op


def test_comb_node_flatten_and_pretty():
    comb = pt.CombNode(2, [pt.Tone(0, 80.0, 1.0), pt.Tone(1, 90.0, 1.0)],
                       0.3, kind="rabi")
    tree = pt.Seq([pt.Para([comb]), pt.PlayNode(4, 2.0, 0.3, kind="dressing")])
    flat = pt.flatten(tree)
    assert flat[0]["op"] == "comb" and flat[1]["op"] == "play"
    # pretty() renders a comb without error and mentions its tones
    text = pt.pretty(tree)
    assert "Comb ch2" in text and "a0" in text and "a1" in text


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} pulse_tree tests passed.")
