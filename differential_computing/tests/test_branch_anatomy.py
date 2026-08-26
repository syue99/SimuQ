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
