"""
test_branch_anatomy.py — stage extraction of the branch-anatomy figure.

Validates the cached figure data (figures/branch_anatomy_data.json) and the
stage classifier against the known structure of the compiled 2q PSR branch:
ev ; move ; CZ ; move ; ev (direct moves — no transit legs).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from build_branch_anatomy import DATA_JSON, stage_names


def _data():
    assert os.path.exists(DATA_JSON), "run build_branch_anatomy.py first"
    with open(DATA_JSON) as f:
        return json.load(f)


def test_stage_sequence_matches_branch_structure():
    d = _data()
    names = stage_names(d["bounds_ns"], d["cz"])
    assert names == ["ev(0,τ)", "move →", "CZ", "move ←", "ev(τ,T)"]


def test_bounds_monotone_and_cz_duration():
    d = _data()
    b = d["bounds_ns"]
    assert all(b[i] < b[i + 1] for i in range(len(b) - 1))
    # the CZ window IS the measured fixed gate (gate_amp_and_phase.csv,
    # 697 pts @ 1 ns -> 696 ns); its normalized envelope is stored alongside
    assert abs((d["cz"]["t1"] - d["cz"]["t0"]) - 696.0) < 1.0
    assert abs(d["cz"]["amp"] - 3.14159265) < 1e-6
    assert abs(max(d["cz"]["env"]) - 1.0) < 1e-9
    assert min(d["cz"]["env"]) >= 0.5 - 1e-9


def test_ledger_rows_agree_with_schedule():
    d = _data()
    rows = d["ledger_rows"]
    assert [r["seg"] for r in rows] == [f"seg{i}" for i in range(5)]
    assert [r["stage"] for r in rows] == ["1", "2", "3", "2", "4"]
    # wall clocks tile the schedule boundaries exactly (µs)
    b_us = [b * 1e-3 for b in d["bounds_ns"]]
    for i, r in enumerate(rows):
        assert abs(r["wall"][0] - b_us[i]) < 1e-9
        assert abs(r["wall"][1] - b_us[i + 1]) < 1e-9
    # s appears ONLY in the insertion row's terms/frame columns
    ins = rows[2]
    assert ins["ins"] == "INS" and "insertion" in ins["terms"]
    assert ins["frame"].startswith("Rz(s·")
    for r in rows:
        if r is not ins:
            assert "s·" not in r["terms"] and "s·" not in r["frame"]
    # coverage: 8 App-F channels, addressing AOD idle, gate only in stage 3
    cov = dict((nm, acts) for nm, acts in d["coverage"])
    assert len(cov) == 8
    assert cov["addr-AOD x"] == [False] * 4
    assert cov["gate"] == [False, False, True, False]
    assert cov["dressing"] == [True, False, False, True]
    assert cov["move-AOD x"] == [False, True, False, False]


def test_transport_traces_reach_gate_zone():
    d = _data()
    gz_x = d["meta"]["gate_zone"][0]
    assert d["meta"]["transit_dy"] is None      # atoms always on the AOD
    x_max = max(max(tr["um"]) for tr in d["x_tones"])
    y_max = max(abs(v) for tr in d["y_tones"] for v in tr["um"])
    assert abs(x_max - (gz_x + d["meta"]["R_cz"] / 2)) < 0.5
    assert y_max <= max(abs(p[1]) for p in
                        d["meta"]["interaction_positions"]) + 0.5


if __name__ == "__main__":
    for fn in [test_stage_sequence_matches_branch_structure,
               test_bounds_monotone_and_cz_duration,
               test_transport_traces_reach_gate_zone]:
        fn()
        print(f"PASS {fn.__name__}")
