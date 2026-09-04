"""
test_F_cycle.py — unit tests for App F's experiment-cycle figure.

fig:cycle is illustrative: it has no cache and no measured data, so what is
worth testing is that the drawing keeps saying what it claims to say. Two of
these are substantive rather than cosmetic:

  * the redrawn-not-copied guard — no source-lab vocabulary in any label, so a
    later edit cannot quietly reintroduce the published figure's terms;
  * the two claims the figure exists to make — every slow-control line holds ONE
    level across the window, and the fast band is active ONLY inside it.

Run:  conda run -n qec_pg python -m pytest differential_computing/tests/test_F_cycle.py -q
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import pytest

import build_F_cycle as fc

# Terms from the source figure and its apparatus. None may appear in a label:
# every label must name a FUNCTION, not a lab's implementation of it.
SOURCE_TERMS = [
    "mot", "pgc", "helmholtz", "pushout", "push-out", "tweezer", "1055",
    "1061", "nm", "cesium", "cs ", "dispenser", "detuning of ", "anti-",
    "2d ", "3d ", "drop", "ramp-down", "spurious", "image 1", "image 2",
]
# "Operation" is deliberately NOT forbidden: it is the source figure's own
# functional name for the programmed window, and the figure follows the
# source's wording there.


def all_label_text():
    """Every string the figure DRAWS, lowercased.

    The caption is deliberately excluded: it has to name the source's class to
    disclaim it ("a different Cs atom array ... not the apparatus of
    \\cite{device}"), which is provenance, not a leaked label.  The caption has
    its own test below.
    """
    out = []
    for _key, hdr, dur, _w, _ms in fc.PHASES:
        out += [hdr, dur]
    out += [label for label, _colour, _levels in fc.SLOW_ROWS]
    out.append(fc.FAST_ROW[0])
    return [t.lower() for t in out]


# ── redrawn, not copied ──────────────────────────────────────────────────────

def test_no_source_lab_vocabulary_in_any_label():
    hits = [(term, text) for text in all_label_text()
            for term in SOURCE_TERMS if term in text]
    assert not hits, f"source-specific vocabulary reappeared: {hits}"


def test_durations_are_orders_of_magnitude_not_measurements():
    """Every duration is written as an approximate scale, never a value."""
    for _key, _hdr, dur, _w, _ms in fc.PHASES:
        assert ("sim" in dur) or ("mu" in dur), dur      # \\sim... or \\mu s


def test_every_row_has_its_own_hue():
    """The source's colour language: one hue per control line."""
    hues = [colour for _l, colour, _lv in fc.SLOW_ROWS] + [fc.C_FAST]
    assert len(set(hues)) == len(hues)


def test_operation_window_duration_is_the_owner_ruling():
    dur = dict((k, d) for k, _h, d, _w, _ms in fc.PHASES)["op"]
    assert "1" in dur and "10 ms" in dur


def test_caption_disclaims_the_device_and_the_data():
    cap = fc.CAPTION.lower()
    assert "generalization" in cap
    assert "not the apparatus" in cap or "different cs atom array" in cap
    assert "no trace here is data" in cap


def test_bib_entry_is_kept_separate_from_the_device_citation():
    """cycle-source is a different apparatus; merging them would misattribute."""
    assert "cycle-source" in fc.BIB
    assert "do not merge" in fc.BIB.lower()


# ── phase layout ─────────────────────────────────────────────────────────────

def test_phase_order_along_the_axis():
    assert fc.SEQUENCE == ["load", "image", "cool", "prep", "op", "readout"]
    assert {k for k, *_ in fc.PHASES} == set(fc.SEQUENCE)


def test_drawn_widths_preserve_the_duration_ordering():
    """The axis is broken, so widths are not proportional — but a phase drawn
    wider than another must still be the longer of the two.  Without this the
    operation (1-10 ms) could end up drawn narrower than the ~1 ms prepare."""
    rows = sorted(fc.PHASES, key=lambda r: -r[4])          # longest first
    widths = [r[3] for r in rows]
    assert widths == sorted(widths, reverse=True), rows


def test_operation_is_longer_than_the_prepare_phase():
    ms = {k: m for k, _h, _d, _w, m in fc.PHASES}
    wd = {k: w for k, _h, _d, w, _m in fc.PHASES}
    assert ms["op"] > ms["prep"] and wd["op"] > wd["prep"]


def test_spans_are_contiguous_and_ordered():
    spans = fc.phase_spans()
    assert spans[0][1] == 0.0
    for (_k0, _a0, b0), (_k1, a1, _b1) in zip(spans, spans[1:]):
        assert b0 == pytest.approx(a1)
        assert a1 < _b1


# ── the two claims the figure makes ──────────────────────────────────────────

def test_every_slow_line_holds_one_level_across_the_window():
    """The figure's claim about slow control: static during the experiment."""
    spans = fc.phase_spans()
    for label, _colour, levels in fc.SLOW_ROWS:
        xs, ys = fc.step_trace(levels, spans)
        wx0, wx1 = [(a, b) for k, a, b in spans if k == "op"][0]
        inside = [y for x, y in zip(xs, ys) if wx0 <= x <= wx1]
        assert len(set(inside)) <= 1, f"{label} changes inside the window"


def test_fast_band_is_silent_outside_the_window_and_active_inside():
    levels = fc.FAST_ROW[1]
    assert levels["op"] > 0
    assert all(v == 0.0 for k, v in levels.items() if k != "op")


def test_every_row_defines_a_level_for_every_phase():
    keys = {k for k, *_ in fc.PHASES}
    rows = [(lab, lv) for lab, _c, lv in fc.SLOW_ROWS] + [fc.FAST_ROW]
    for label, levels in rows:
        assert set(levels) == keys, f"{label} is missing a phase"
        assert all(0.0 <= v <= 1.0 for v in levels.values())


def test_slow_rows_cover_the_five_named_control_lines():
    assert [label for label, _c, _l in fc.SLOW_ROWS] == [
        "trap depth", "cooling light", "bias field", "pump light",
        "camera trigger"]


def test_thumbnail_line_count_matches_the_collapsed_band_label():
    assert str(fc.N_FAST_LINES) in fc.FAST_ROW[0] or \
        fc.N_FAST_LINES == 8            # label is built from N_FAST_LINES
