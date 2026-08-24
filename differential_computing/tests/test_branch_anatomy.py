"""
test_branch_anatomy.py — stage extraction of the branch-anatomy figure.

Validates the cached figure data (figures/branch_anatomy_data.json) and the
stage classifier against the known structure of the compiled 2q PSR branch:
ev ; lift ; move ; drop ; CZ ; lift ; move ; drop ; ev.
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
    assert names == ["ev(0,τ)", "lift", "move →", "drop", "CZ",
                     "lift", "move ←", "drop", "ev(τ,T)"]


def test_bounds_monotone_and_cz_duration():
    d = _data()
    b = d["bounds_ns"]
    assert all(b[i] < b[i + 1] for i in range(len(b) - 1))
    assert abs((d["cz"]["t1"] - d["cz"]["t0"]) - 200.0) < 1.0
    assert abs(d["cz"]["amp"] - 3.14159265) < 1e-6


def test_transport_traces_reach_gate_zone_and_lane():
    d = _data()
    gz_x = d["meta"]["gate_zone"][0]
    dy = d["meta"]["transit_dy"]
    x_max = max(max(tr["um"]) for tr in d["x_tones"])
    y_max = max(max(tr["um"]) for tr in d["y_tones"])
    assert abs(x_max - (gz_x + d["meta"]["R_cz"] / 2)) < 0.5
    assert abs(y_max - (max(p[1] for p in
               d["meta"]["interaction_positions"]) + dy)) < 0.5


if __name__ == "__main__":
    for fn in [test_stage_sequence_matches_branch_structure,
               test_bounds_monotone_and_cz_duration,
               test_transport_traces_reach_gate_zone_and_lane]:
        fn()
        print(f"PASS {fn.__name__}")
