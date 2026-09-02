"""
test_F_waveform.py — unit tests for the App F emission figure's data path.

The pipeline itself is exercised by build_F_waveform.extract(); these tests
cover the pure functions and the invariants the cached artifact must satisfy,
so they run in a second without touching PulseDSL.

Run:  conda run -n qec_pg python -m pytest differential_computing/tests/test_F_waveform.py -q
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pytest

import build_F_waveform as bw


class _Ins:
    """Stand-in for a machine instruction (only its identity is used)."""

    def __init__(self, name):
        self.name = name


def _boxes(lvars_per_ins):
    """Minimal boxes structure: [( [ (key, ins, h_eval, lvars), ... ], dur )]."""
    ents = [((0, i), _Ins(f"ins{i}"), None, list(lv))
            for i, lv in enumerate(lvars_per_ins)]
    return [(ents, 1.0)]


def _lvars(boxes):
    return [list(lv) for be, _d in boxes for *_rest, lv in be]


# ── _scale_boxes ─────────────────────────────────────────────────────────────

def test_scale_boxes_scales_single_amplitude():
    out = bw._scale_boxes(_boxes([[2.0], [-0.5]]), 0.75)
    assert _lvars(out) == [[1.5], [-0.375]]


def test_scale_boxes_leaves_phase_untouched():
    """Rabi carries (Omega, phi); only Omega scales."""
    out = bw._scale_boxes(_boxes([[2.0, 0.3]]), 0.5)
    amp, phi = _lvars(out)[0]
    assert amp == pytest.approx(1.0)
    assert phi == pytest.approx(0.3)


def test_scale_boxes_negative_scale_goes_into_the_phase():
    """A modulator cannot play a negative Rabi amplitude: sign -> phi + pi."""
    out = bw._scale_boxes(_boxes([[2.0, 0.3]]), -0.5)
    amp, phi = _lvars(out)[0]
    assert amp == pytest.approx(1.0)          # |scale| * amp, stays positive
    assert phi == pytest.approx(0.3 + np.pi)


def test_scale_boxes_negative_scale_keeps_sign_on_single_lvar():
    """Detuning/dressing/zz are signed quantities — no phase to absorb into."""
    out = bw._scale_boxes(_boxes([[2.0]]), -0.5)
    assert _lvars(out) == [[-1.0]]


def test_scale_boxes_does_not_mutate_the_source():
    src = _boxes([[2.0, 0.3]])
    bw._scale_boxes(src, 0.5)
    assert _lvars(src) == [[2.0, 0.3]]


def test_scale_boxes_identity():
    src = _boxes([[2.0], [1.0, 0.25]])
    assert _lvars(bw._scale_boxes(src, 1.0)) == _lvars(src)


# ── cached artifact invariants ───────────────────────────────────────────────

@pytest.mark.skipif(not (os.path.exists(bw.META) and os.path.exists(bw.NPZ)),
                    reason="figure cache not built (run build_F_waveform.py)")
class TestCache:

    @staticmethod
    def _meta():
        with open(bw.META) as f:
            return json.load(f)

    def test_shift_matches_the_drawn_nyquist_mode(self):
        m = self._meta()
        assert m["s"] == pytest.approx((m["nsr_mode"] + 0.5) / (2 * m["K"]))

    def test_scale_is_the_coefficient_ratio(self):
        m = self._meta()
        assert m["scale"] == pytest.approx(m["coeff_shifted"] / m["coeff_source"])

    def test_shifted_branch_is_an_exact_rescale(self):
        """The witness: residual(shifted) == residual(source) * |scale|."""
        m = self._meta()
        assert m["residual_exact"] is True
        assert m["residual_nsr"] == pytest.approx(
            m["residual_source"] * abs(m["scale"]), rel=1e-9)

    def test_nsr_lane_is_one_segment_of_length_T(self):
        m = self._meta()
        assert m["nsr_wall_ns"] == pytest.approx(m["T_us"] * 1e3)

    def test_nsr_lane_emits_no_transport_and_no_gate(self):
        """The structural claim the figure makes."""
        act = self._meta()["lanes"]["nsr"]["active"]
        assert act[0] is False and act[5] is False      # transport AOD x / y
        assert act[4] is False                          # gate AOM
        assert act[2] is True and act[3] is True        # Rabi + dressing on

    def test_psr_lane_does_emit_transport_and_gate(self):
        act = self._meta()["lanes"]["psr"]["active"]
        assert act[0] is True and act[4] is True

    def test_transport_accounting_is_consistent(self):
        """A cache invariant, not a figure claim: the figure no longer asserts
        anything from the per-branch durations (measurement and loading
        dominate on a real machine).  The bound only guards the extraction —
        if interval merging broke, this share would collapse."""
        m = self._meta()
        assert 0.0 < m["psr_transport_ns"] < m["psr_wall_ns"]
        assert m["psr_transport_ns"] / m["psr_wall_ns"] > 0.9

    def test_every_channel_has_a_waveform_row_in_both_lanes(self):
        m = self._meta()
        arrays = np.load(bw.NPZ)
        for tag in ("psr", "nsr"):
            n = len(arrays[f"{tag}_t"])
            for c in m["channels"]:
                assert len(arrays[f"{tag}_ch{c}"]) == n

    def test_silent_rows_really_are_silent(self):
        m = self._meta()
        arrays = np.load(bw.NPZ)
        for tag in ("psr", "nsr"):
            for c_str, _name in m["channels"].items():
                c = int(c_str)
                w = np.abs(arrays[f"{tag}_ch{c}"]).max()
                if not m["lanes"][tag]["active"][c]:
                    assert w <= 1e-12
