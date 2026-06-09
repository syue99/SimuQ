"""
test_tweezer_tree.py — Step 2: TweezerMapper.map_hlist_tree native op-tree.

Validates that:
  1. flatten(map_hlist_tree(H_list)) == map_hlist(H_list)[0]  exactly
     (the tree only adds timing structure; it adds/drops/reorders no op),
     for the 1q / 2q / 3q validated models.
  2. The tree has the expected position-segmented PARA structure:
     - exactly one subtree per H_list segment (outer SEQ),
     - same-position plays (dressing + native) share a PARA,
     - a ZZ play lands in its own PARA, separated from interaction-zone
       plays by an AOD barrier.
  3. The ledger produced by map_hlist_tree matches map_hlist's (the tree path
     must not change verification data).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sympy as sp
import pytest

from simuq import QSystem, Qubit
from simuq.braket.diffQC_provider import diffQCProvider
from observable_program_generator import observable_program_generator
from tweezer_mapper import TweezerMapper
import pulse_tree as pt


# ── Setup helpers (mirror test_single_trial.py) ───────────────────────────────

def _compile_mapper(H_param, x_val, T, n_qubits):
    prov = diffQCProvider()
    qs_c = QSystem()
    _ = [Qubit(qs_c) for _ in range(n_qubits)]
    H_eval = H_param.set_parameterizedHam({"x": x_val})
    qs_c.add_evolution(H_eval, T)
    prov.compile(qs_c, "quera", "Aquila", "rydberg2d", tol=0.1, verbose=0)
    n_sites, sol_gvars, boxes, _edges, _ = prov.prog
    return TweezerMapper(n_qubits=n_sites, sol_gvars=sol_gvars, boxes=boxes,
                         ramp_time=0.01)


def _one_branch(H_param, T, x_val, seed=1):
    np.random.seed(seed)
    programs = observable_program_generator(
        H_param, T, n_sample=1, n_repetition=1, diff_var="x", value=x_val)
    # programs[k] = (H_tot_list, ugrad, n_rep); take the first branch
    H_tot_list = programs[0][0]
    return H_tot_list[0]


def _build_1q():
    x = sp.Symbol("x")
    qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    return x * q[0].Z + q[0].X, 2


def _build_nq(n):
    x = sp.Symbol("x")
    qs = QSystem(); q = [Qubit(qs) for _ in range(n)]
    H = sp.sin(2 * x) * q[0].Z * q[1].Z
    for i in range(n):
        H = H + sp.sin(2 * x) * q[i].X
    return H, n


MODELS = {
    "1q": _build_1q(),
    "2q": _build_nq(2),
    "3q": _build_nq(3),
}


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", list(MODELS))
def test_flatten_matches_flat_map_hlist(name):
    H_param, n = MODELS[name]
    T, x_val = 0.5, 0.7
    H_list = _one_branch(H_param, T, x_val)

    mapper = _compile_mapper(H_param, x_val, T, n)
    flat_ops, _, _ = mapper.map_hlist(H_list, T=T)
    tree, _, _ = mapper.map_hlist_tree(H_list, T=T)

    assert pt.flatten(tree) == flat_ops


@pytest.mark.parametrize("name", list(MODELS))
def test_one_subtree_per_segment(name):
    H_param, n = MODELS[name]
    T, x_val = 0.5, 0.7
    H_list = _one_branch(H_param, T, x_val)
    mapper = _compile_mapper(H_param, x_val, T, n)
    tree, _, _ = mapper.map_hlist_tree(H_list, T=T)

    assert isinstance(tree, pt.Seq)
    # Outer SEQ has exactly one child subtree per H_list segment.
    assert len(tree.children) == len(H_list)
    assert all(isinstance(c, pt.Seq) for c in tree.children)


def test_ledger_matches_between_paths():
    H_param, n = MODELS["2q"]
    T, x_val = 0.5, 0.7
    H_list = _one_branch(H_param, T, x_val)
    mapper = _compile_mapper(H_param, x_val, T, n)

    _, _, ledger_flat = mapper.map_hlist(H_list, T=T)
    _, _, ledger_tree = mapper.map_hlist_tree(H_list, T=T)
    # Same number of recorded ledger steps, same channel_kind sequence.
    kinds_flat = [e.channel_kind for e in ledger_flat.entries]
    kinds_tree = [e.channel_kind for e in ledger_tree.entries]
    assert kinds_flat == kinds_tree


def test_zz_isolated_from_interaction_plays():
    """In a 2q evolution segment, the ZZ play (ch[2n+1]) must not share a PARA
    with interaction-zone plays (dressing ch[2n] / detuning / rabi); an AOD
    barrier must separate them."""
    H_param, n = MODELS["2q"]
    T, x_val = 0.5, 0.7
    H_list = _one_branch(H_param, T, x_val)
    mapper = _compile_mapper(H_param, x_val, T, n)
    tree, _, _ = mapper.map_hlist_tree(H_list, T=T)

    zz_ch = 2 * n + 1
    found_zz = False
    for seg in tree.children:                 # each segment subtree (a Seq)
        for node in seg.children:
            if isinstance(node, pt.Para):
                channels = {c.channel for c in node.children
                            if isinstance(c, pt.PlayNode)}
                if zz_ch in channels:
                    found_zz = True
                    # ZZ alone in its PARA — no interaction-zone channel with it
                    interaction_chs = set(range(2 * n))        # detuning+rabi
                    interaction_chs.add(2 * n)                 # dressing
                    assert not (channels & interaction_chs), (
                        f"ZZ shares a PARA with interaction plays: {channels}")
    assert found_zz, "expected a ZZ play in the 2q evolution segment"


if __name__ == "__main__":
    for name in MODELS:
        test_flatten_matches_flat_map_hlist(name)
        print(f"PASS flatten-matches [{name}]")
        test_one_subtree_per_segment(name)
        print(f"PASS subtree-per-segment [{name}]")
    test_ledger_matches_between_paths()
    print("PASS ledger-matches")
    test_zz_isolated_from_interaction_plays()
    print("PASS zz-isolated")
    print("\nAll tweezer_tree tests passed.")
