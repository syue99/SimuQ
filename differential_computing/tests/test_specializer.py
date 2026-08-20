"""
Tests for the target-aware specialization layer (simuq/specializer.py) and the
specialize=True provider path: plan geometry, warm-start exactness, pruning,
compiled-Hamiltonian round trip vs both the target and the vanilla path, n>=10
ZZ-name parsing, and the hardware ledger.

Run:  conda run -n qec_pg python -m pytest differential_computing/tests/test_specializer.py -q
"""

import os
import sys

import numpy as np
import pytest
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simuq import QSystem, Qubit
from simuq import specializer
from simuq.aais import rydberg2d
from simuq.braket.diffQC_provider import diffQCProvider
from observable_program_generator import observable_program_generator
from tweezer_mapper import classify_instruction

T = 1.0
X_VAL = 0.8


def tfim_qs(n, x_val=X_VAL, T_=T):
    x = sp.Symbol("x")
    qs = QSystem()
    q = [Qubit(qs) for _ in range(n)]
    H = x * sum((q[i].Z * q[i + 1].Z for i in range(n - 1)), 0 * q[0].Z)
    for i in range(n):
        H = H + q[i].X
    qs.add_evolution(H.set_parameterizedHam({"x": x_val}), T_)
    return qs, H


def ham_dict(h):
    d = {}
    for prod, c in h.ham:
        k = prod.to_tuple()
        if k:
            d[k] = d.get(k, 0.0) + float(c)
    return d


def compiled_ham(prov):
    comp = {}
    _n, _gv, boxes, _e, _tr = prov.prog
    for entries, _dur in boxes:
        for (_, ins, h_eval, _lv) in entries:
            for prod, c in h_eval.ham:
                k = prod.to_tuple()
                if k:
                    comp[k] = comp.get(k, 0.0) + float(c)
    return comp, boxes


def max_diff(a, b):
    return max(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


# ── plan / geometry ──────────────────────────────────────────────────────────

def test_plan_chain_geometry():
    n = 6
    qs, _ = tfim_qs(n)
    plan = specializer.make_plan(qs, C_6=rydberg2d.C_6)
    assert plan.n == n
    assert plan.links == [(i, i + 1) for i in range(n - 1)]
    # shells=1 keeps exactly the nearest-neighbor pairs
    assert sorted(plan.dressing_pairs) == plan.links
    # collinear, uniform spacing R, qubit 0 at origin
    assert plan.positions[0] == (0.0, 0.0)
    for (x0, y0), (x1, y1) in zip(plan.positions, plan.positions[1:]):
        assert y0 == y1 == 0.0
        assert abs(abs(x1 - x0) - plan.R) < 1e-9
    # spacing chosen so the dressing amplitude warm-starts at |o| = 1
    assert abs(rydberg2d.C_6 / (4 * plan.R ** 6) - abs(plan.theta)) < 1e-9
    assert plan.dressing_init == pytest.approx(np.sign(plan.theta))
    # dropped tail is small and reported
    assert 0 < plan.dropped_zz_l1 < 0.1 * abs(plan.theta) * len(plan.links)


def test_plan_rejects_branching_graph():
    x = sp.Symbol("x")
    qs = QSystem()
    q = [Qubit(qs) for _ in range(4)]
    # star graph: site 0 coupled to 1, 2, 3 — degree 3
    H = x * (q[0].Z * q[1].Z + q[0].Z * q[2].Z + q[0].Z * q[3].Z)
    qs.add_evolution(H.set_parameterizedHam({"x": 1.0}), T)
    with pytest.raises(NotImplementedError):
        specializer.make_plan(qs, C_6=rydberg2d.C_6)


def test_warm_start_is_exact_solution():
    """The analytic init must satisfy the target term-by-term: dressing carries
    ZZ, detunings cancel the dressing's single-Z side effect, Rabi carries X."""
    n = 5
    qs, _ = tfim_qs(n)
    plan = specializer.make_plan(qs, C_6=rydberg2d.C_6)
    theta = plan.theta
    # interior sites see 2 bonds, edge sites 1
    assert plan.detuning_init[0] == pytest.approx(2 * theta, rel=1e-9)
    assert plan.detuning_init[2] == pytest.approx(4 * theta, rel=1e-9)
    for i in range(n):
        assert plan.rabi_init[i] == (pytest.approx(2.0), pytest.approx(0.0))


# ── compiled Hamiltonian round trips ─────────────────────────────────────────

@pytest.mark.parametrize("n", [4, 7])
def test_specialized_compile_matches_target(n):
    qs, _ = tfim_qs(n)
    prov = diffQCProvider()
    prov.compile(qs, "quera", "Aquila", "rydberg2d", tol=0.1, specialize=True)
    comp, boxes = compiled_ham(prov)
    assert len(boxes) == 1
    assert max_diff(comp, ham_dict(qs.evos[0][0])) < 1e-9
    # ZZ derived lines are pruned from the evolution segment; the coupling
    # rides on the dressing line
    kinds = {classify_instruction(ins)[0]
             for entries, _ in boxes for (_, ins, _, _) in entries}
    assert kinds == {"detuning", "rabi", "dressing"}


def test_specialized_agrees_with_vanilla():
    """Both paths compile the same AAIS; their compiled Hamiltonians must agree
    (each within its solve tolerance of the shared target)."""
    n = 5
    qs_s, _ = tfim_qs(n)
    prov_s = diffQCProvider()
    prov_s.compile(qs_s, "quera", "Aquila", "rydberg2d", tol=0.1, specialize=True)
    comp_s, _ = compiled_ham(prov_s)

    qs_v, _ = tfim_qs(n)
    prov_v = diffQCProvider()
    prov_v.compile(qs_v, "quera", "Aquila", "rydberg2d", tol=0.1, specialize=False)
    comp_v, _ = compiled_ham(prov_v)

    targ = ham_dict(qs_s.evos[0][0])
    assert max_diff(comp_s, targ) < 1e-9        # specialized: exact witness
    assert max_diff(comp_v, targ) < 0.1         # vanilla: within solver tol
    assert max_diff(comp_s, comp_v) < 0.1


# ── n >= 10 name format ──────────────────────────────────────────────────────

def test_zz_name_parses_two_digit_indices():
    class FakeIns:
        def __init__(self, name):
            self.name = name

    assert classify_instruction(FakeIns("c10_11_zz")) == ("zz", 10, 11)
    assert classify_instruction(FakeIns("c3_7_zz")) == ("zz", 3, 7)
    assert classify_instruction(FakeIns("c12_zz")) == ("zz", 1, 2)  # legacy

    n = 12
    qs, _ = tfim_qs(n)
    plan = specializer.make_plan(qs, C_6=rydberg2d.C_6)
    mach = rydberg2d.generate_qmachine(
        n, inits=plan.positions, fix_positions=True,
        links=plan.links, dressing_pairs=plan.dressing_pairs)
    zz_pairs = set()
    for line in mach.lines:
        for ins in line.inss:
            cls = classify_instruction(ins)
            if cls[0] == "zz":
                zz_pairs.add((cls[1], cls[2]))
    assert (10, 11) in zz_pairs
    assert zz_pairs == set(plan.links)


# ── hardware path: ledger + round-trip verify ────────────────────────────────

def test_hardware_ledger_and_verify_roundtrip():
    n = 4
    qs, H = tfim_qs(n)
    prov = diffQCProvider()
    prov.compile(qs, "quera", "Aquila", "rydberg2d", tol=0.1, specialize=True)

    np.random.seed(1)
    progs = observable_program_generator(H, T, n_sample=2, n_repetition=1,
                                         diff_var="x", value=X_VAL)
    prov.run(progs, None, T, backend="hardware")
    led = prov.get_pulse_ledger(program_idx=0, branch_idx=0)
    assert led is not None and len(led.entries) > 0

    from qutip_sequential import QuTiPSequentialRunner
    obs = QuTiPSequentialRunner(n).zz_observable(0, 1)
    res = prov.verify(progs, obs, T)
    # reconstruction from ledger meta-data must reproduce the direct gradient
    assert res["error"] < 1e-6, res
    worst = max(nd["norm_diff"] for nd in res["norm_diffs"])
    assert worst < 1e-6, f"worst segment norm diff {worst}"


def test_hardware_map_n20():
    n = 20
    qs, H = tfim_qs(n)
    prov = diffQCProvider()
    prov.compile(qs, "quera", "Aquila", "rydberg2d", tol=0.1, specialize=True)
    np.random.seed(2)
    progs = observable_program_generator(H, T, n_sample=1, n_repetition=1,
                                         diff_var="x", value=X_VAL)
    assert len(progs) == n - 1
    prov.run(progs[:1], None, T, backend="hardware")
    led = prov.get_pulse_ledger(program_idx=0, branch_idx=0)
    kinds = [e.channel_kind for e in led.entries if e.channel_kind]
    assert "dressing" in kinds and "kick" in kinds
    # every position snapshot covers all atoms
    assert all(len(e.positions) == n for e in led.entries)
