"""
test_F_regimes.py — unit tests for App G.4's three-family (p, q) sweep.

Covers the parts that are new in build_F_regimes (the extended tangent
alphabet, the family draws, the eq:margin selector) and the invariants the
cached plane must satisfy — above all that panels (a)/(b) still reproduce
Figure 10, which is what licenses comparing the three panels at all.

Run:  conda run -n qec_pg python -m pytest differential_computing/tests/test_F_regimes.py -q
"""

import itertools
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pytest

import build_F_regimes as fr
import build_F_select as bs
import selector_check as sc


# ── extended alphabet ────────────────────────────────────────────────────────

def test_device_terms_come_first_and_are_unchanged():
    """Panels (a)/(b) index the same operators as Fig 10 — same indices."""
    ops, labels, _edges = fr.build_extended_alphabet()
    base_ops, base_labels = bs.build_alphabet()
    assert labels[:len(base_labels)] == base_labels
    assert len(base_labels) == 35                      # 7 X + 7 Z + 21 ZZ
    assert len(ops) == 35 + 21 + 21                    # + XX + YY per edge


def test_no_Y_in_the_device_signature():
    """G.1's alphabet question: the device pool has no single-site Y."""
    _ops, labels = bs.build_alphabet()
    assert {l[0] for l in labels} == {"X", "Z", "ZZ"}


def test_edges_hold_the_heisenberg_triple():
    ops, labels, edges = fr.build_extended_alphabet()
    assert len(edges) == 21
    for (i, j), (ixx, iyy, izz) in edges.items():
        assert labels[ixx] == ("XX", i, j)
        assert labels[iyy] == ("YY", i, j)
        assert labels[izz] == ("ZZ", i, j)


# ── Pauli bit-vectors ────────────────────────────────────────────────────────

def test_label_to_xz_matches_selector_check_on_device_terms():
    _ops, labels, _e = fr.build_extended_alphabet()
    for l in labels:
        if l[0] in ("X", "Z", "ZZ"):
            assert fr.label_to_xz(l) == sc.label_to_xz(l)


def test_yy_sets_both_bit_planes():
    """Y = iXZ, so a Y site carries an x AND a z bit."""
    x, z = fr.label_to_xz(("YY", 0, 3))
    assert x == z == (1 << 0) | (1 << 3)


def test_same_bond_heisenberg_terms_commute():
    """XX, YY, ZZ on ONE bond pairwise commute — the q<=3 singleton case."""
    trio = [fr.label_to_xz(("XX", 1, 4)), fr.label_to_xz(("YY", 1, 4)),
            fr.label_to_xz(("ZZ", 1, 4))]
    for a, b in itertools.combinations(trio, 2):
        assert not sc.anticommute(a, b)


def test_bonds_sharing_a_qubit_anticommute():
    """Why the clique cover still bites past one bond (docstring's claim)."""
    assert sc.anticommute(fr.label_to_xz(("XX", 0, 1)),
                          fr.label_to_xz(("YY", 1, 2)))


def test_disjoint_bonds_commute():
    assert not sc.anticommute(fr.label_to_xz(("XX", 0, 1)),
                              fr.label_to_xz(("YY", 2, 3)))


# ── family draws ─────────────────────────────────────────────────────────────

def test_general_and_aligned_delegate_verbatim_to_fig10():
    _ops, labels, edges = fr.build_extended_alphabet()
    base = [l for l in labels if l[0] in ("X", "Z", "ZZ")]
    for family in ("general", "aligned"):
        mine = fr.draw_params(family, 5, 3, labels, edges,
                              np.random.default_rng(7))
        theirs = bs.draw_params(family, 5, 3, base, np.random.default_rng(7))
        assert mine == theirs


def test_aligned_draws_only_zz():
    _ops, labels, edges = fr.build_extended_alphabet()
    params = fr.draw_params("aligned", 8, 2, labels, edges,
                            np.random.default_rng(1))
    for d in params:
        assert all(labels[j][0] == "ZZ" for j in d)


def test_heisenberg_fills_whole_bonds_and_respects_q():
    _ops, labels, edges = fr.build_extended_alphabet()
    for q in (1, 3, 6, 12, 35):
        params = fr.draw_params("heisenberg", q, 2, labels, edges,
                                np.random.default_rng(3))
        for d in params:
            assert len(d) == q
            assert all(labels[j][0] in ("XX", "YY", "ZZ") for j in d)
        # a q that is a whole number of bonds must cover exactly q/3 bonds
        if q % 3 == 0:
            for d in params:
                bonds = {labels[j][1:] for j in d}
                assert len(bonds) == q // 3


# ── the eq:margin selector ───────────────────────────────────────────────────

def test_margin_is_capped_at_one_and_pinned_to_sqrt_q():
    """gamma(q) = min(1, 1.86/sqrt(q)); the cap binds only while q <= 1.86^2."""
    m = fr.margin_column([1, 3, 4, 9, 36])[:, 0]
    assert m[0] == pytest.approx(0.0)                  # 1.86/1   = 1.86 -> capped
    assert m[1] == pytest.approx(0.0)                  # 1.86/1.7 = 1.07 -> capped
    assert m[2] == pytest.approx(np.log10(fr.GAMMA0 / 2.0))    # 0.93, not capped
    assert m[3] == pytest.approx(np.log10(fr.GAMMA0 / 3.0))
    assert m[4] == pytest.approx(np.log10(fr.GAMMA0 / 6.0))


def test_margin_never_penalises_psr():
    """gamma <= 1, so the margin can only move cells toward NSR."""
    assert (fr.margin_column(list(range(1, 36))) <= 1e-12).all()


# ── cached plane ─────────────────────────────────────────────────────────────

@pytest.mark.skipif(not os.path.exists(fr.CACHE),
                    reason="sweep cache not built (run build_F_regimes.py)")
class TestCache:

    @staticmethod
    def _data():
        return json.load(open(fr.CACHE))

    def test_grid_is_fig10s(self):
        m = self._data()["meta"]
        assert m["PS"] == list(range(1, 11))
        assert m["KS"] == list(range(1, 36))

    def test_panel_a_reproduces_fig10_bit_exactly(self):
        repro = fr.check_reproduces_fig10(self._data())
        assert repro is not None and repro < 1e-12

    def test_certificate_chain_holds_everywhere(self):
        """L1 >= AC >= true, so AC/L1 <= 1 in every cell."""
        d = self._data()
        for fam, _t in fr.FAMILIES:
            chi = np.array(d[fam]["ac_over_l1"])
            assert (chi <= 1.0 + 1e-9).all() and (chi > 0).all()

    def test_aligned_grouping_never_helps(self):
        """ZZ terms all commute -> singleton cliques -> AC == L1 at every q."""
        chi = np.array(self._data()["aligned"]["ac_over_l1"])
        assert np.allclose(chi, 1.0, atol=1e-9)

    def test_heisenberg_grouping_helps_only_past_one_bond(self):
        d = self._data()
        chi = np.array(d["heisenberg"]["ac_over_l1"])
        ks = d["meta"]["KS"]
        for i, q in enumerate(ks):
            if q <= 3:                                  # one bond: commuting
                assert np.allclose(chi[i], 1.0, atol=1e-9)
        assert chi[ks.index(35)].mean() < 0.7           # measured 0.60

    def test_families_land_in_different_regimes(self):
        """The figure's point: the same plane, three different answers."""
        d = self._data()
        share = {f: float((np.array(d[f]["Z"]) < 0).mean())
                 for f, _t in fr.FAMILIES}
        assert share["aligned"] < share["general"] < share["heisenberg"]
        assert share["heisenberg"] > 0.85

    def test_margin_improves_agreement_on_every_family(self):
        for r in fr.selector_table(self._data()):
            assert r["agree_AC+margin"] >= r["agree_AC"]
            assert r["agree_AC"] >= r["agree_L1"]
