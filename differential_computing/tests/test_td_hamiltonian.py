"""
Unit tests for differential_computing.td_hamiltonian.

Covers factor_td_hamiltonian, check_dressing_collision, trotterize_td, and
build_channel_envelopes.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import sympy as sp

from simuq import QSystem, Qubit
from simuq.hamiltonian import TIHamiltonian

from td_hamiltonian import (
    build_channel_envelopes,
    check_dressing_collision,
    factor_td_hamiltonian,
    trotterize_td,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _qubit_system(n):
    qs = QSystem()
    q = [Qubit(qs) for _ in range(n)]
    return qs, q


def _find_group(groups, env_target):
    """Return the TIHamiltonian whose envelope equals env_target (sympy)."""
    for env, ti in groups:
        if sp.simplify(env - env_target) == 0:
            return ti
    raise AssertionError(f"No group with envelope {env_target}; got {[str(e) for e,_ in groups]}")


def _ham_dict(ti: TIHamiltonian):
    """Return {h_tuple: float_coef}."""
    return {prod.to_tuple(): float(coef) for prod, coef in ti.ham}


# ═════════════════════════════════════════════════════════════════════════════
# factor_td_hamiltonian
# ═════════════════════════════════════════════════════════════════════════════

class TestFactorTDHamiltonian:

    def test_case1_single_qubit_two_envelopes(self):
        """sin(t)·Z0 + cos(t)·X0 → 2 groups, no collision."""
        t = sp.Symbol("t")
        _qs, q = _qubit_system(1)
        H = sp.sin(t) * q[0].Z + sp.cos(t) * q[0].X

        groups = factor_td_hamiltonian(H, t, param_dict={})

        assert len(groups) == 2
        sin_ti = _find_group(groups, sp.sin(t))
        cos_ti = _find_group(groups, sp.cos(t))

        sin_map = _ham_dict(sin_ti)
        cos_map = _ham_dict(cos_ti)

        # sin group has Z on qubit 0, weight 1
        ((h_sin, c_sin),) = [(k, v) for k, v in sin_map.items() if abs(v) > 0]
        assert dict(h_sin)[0] == "Z"
        assert abs(c_sin - 1.0) < 1e-12

        ((h_cos, c_cos),) = [(k, v) for k, v in cos_map.items() if abs(v) > 0]
        assert dict(h_cos)[0] == "X"
        assert abs(c_cos - 1.0) < 1e-12

    def test_case2_disjoint_two_body(self):
        """sin(t)·(Z0Z1 + X0) + cos(t)·Z2Z3 → 2 groups, disjoint active sets."""
        t = sp.Symbol("t")
        _qs, q = _qubit_system(4)
        H = sp.sin(t) * (q[0].Z * q[1].Z + q[0].X) + sp.cos(t) * q[2].Z * q[3].Z

        groups = factor_td_hamiltonian(H, t, param_dict={})

        assert len(groups) == 2
        sin_ti = _find_group(groups, sp.sin(t))
        cos_ti = _find_group(groups, sp.cos(t))

        # sin group touches qubits {0, 1}
        sin_prods = [prod.to_tuple() for prod, _ in sin_ti.ham]
        sin_sites = {i for tup in sin_prods for i, op in tup if op != ""}
        assert sin_sites == {0, 1}

        # cos group touches qubits {2, 3}
        cos_prods = [prod.to_tuple() for prod, _ in cos_ti.ham]
        cos_sites = {i for tup in cos_prods for i, op in tup if op != ""}
        assert cos_sites == {2, 3}

    def test_case3_shared_atom_collision(self):
        """sin(t)·Z0Z1 + cos(t)·Z1Z2 → qubit 1 shared; collision flag test covered separately."""
        t = sp.Symbol("t")
        _qs, q = _qubit_system(3)
        H = sp.sin(t) * q[0].Z * q[1].Z + sp.cos(t) * q[1].Z * q[2].Z

        groups = factor_td_hamiltonian(H, t, param_dict={})

        assert len(groups) == 2
        sin_ti = _find_group(groups, sp.sin(t))
        cos_ti = _find_group(groups, sp.cos(t))

        sin_sites = {i for prod, _ in sin_ti.ham for i in prod.keys() if prod[i] != ""}
        cos_sites = {i for prod, _ in cos_ti.ham for i in prod.keys() if prod[i] != ""}
        assert sin_sites == {0, 1}
        assert cos_sites == {1, 2}

    def test_param_substitution_before_factor(self):
        """H = sin(x·t)·Z0 with x=1.0 → envelope sin(1.0·t) becomes the grouping key."""
        t, x = sp.symbols("t x")
        _qs, q = _qubit_system(1)
        H = sp.sin(x * t) * q[0].Z

        groups = factor_td_hamiltonian(H, t, param_dict={x: 1.0})
        assert len(groups) == 1
        env, ti = groups[0]
        # sin(1.0*t) should be present; substitute t=0 → 0 and t=π/2 → ~1.0
        f = sp.lambdify(t, env, "numpy")
        assert abs(float(f(0.0))) < 1e-9
        assert abs(float(f(np.pi / 2.0)) - 1.0) < 1e-9

    def test_constant_envelope_kept(self):
        """A constant coefficient groups under envelope=1 (genuine TI term)."""
        t = sp.Symbol("t")
        _qs, q = _qubit_system(1)
        H = 3.0 * q[0].Z + sp.sin(t) * q[0].X

        groups = factor_td_hamiltonian(H, t, param_dict={})
        assert len(groups) == 2
        ti_const = _find_group(groups, sp.Integer(1))
        sin_ti = _find_group(groups, sp.sin(t))

        const_map = _ham_dict(ti_const)
        ((_, c),) = const_map.items()
        assert abs(c - 3.0) < 1e-12

        sin_map = _ham_dict(sin_ti)
        ((_, c2),) = sin_map.items()
        assert abs(c2 - 1.0) < 1e-12

    def test_mixed_sum_coefficient_splits(self):
        """coef 2·sin(t)+3·cos(t) on a single term splits into 2 groups."""
        t = sp.Symbol("t")
        _qs, q = _qubit_system(1)
        # Build via addition so the coefs land on the same productHamiltonian.
        H = 2 * sp.sin(t) * q[0].Z + 3 * sp.cos(t) * q[0].Z

        groups = factor_td_hamiltonian(H, t, param_dict={})

        assert len(groups) == 2
        sin_ti = _find_group(groups, sp.sin(t))
        cos_ti = _find_group(groups, sp.cos(t))

        sin_map = _ham_dict(sin_ti)
        cos_map = _ham_dict(cos_ti)
        ((_, cs),) = sin_map.items()
        ((_, cc),) = cos_map.items()
        assert abs(cs - 2.0) < 1e-12
        assert abs(cc - 3.0) < 1e-12


# ═════════════════════════════════════════════════════════════════════════════
# check_dressing_collision
# ═════════════════════════════════════════════════════════════════════════════

class TestCheckDressingCollision:

    def test_disjoint_two_body(self):
        t = sp.Symbol("t")
        _qs, q = _qubit_system(4)
        H = sp.sin(t) * q[0].Z * q[1].Z + sp.cos(t) * q[2].Z * q[3].Z
        groups = factor_td_hamiltonian(H, t, param_dict={})
        assert check_dressing_collision(groups) is False

    def test_shared_atom_collides(self):
        t = sp.Symbol("t")
        _qs, q = _qubit_system(3)
        H = sp.sin(t) * q[0].Z * q[1].Z + sp.cos(t) * q[1].Z * q[2].Z
        groups = factor_td_hamiltonian(H, t, param_dict={})
        assert check_dressing_collision(groups) is True

    def test_single_body_sharing_is_not_a_collision(self):
        """sin(t)·Z0 + cos(t)·X0: same qubit, but only 1-body terms → no collision."""
        t = sp.Symbol("t")
        _qs, q = _qubit_system(1)
        H = sp.sin(t) * q[0].Z + sp.cos(t) * q[0].X
        groups = factor_td_hamiltonian(H, t, param_dict={})
        assert check_dressing_collision(groups) is False

    def test_mixed_two_body_and_single_body_disjoint(self):
        """sin(t)·Z0Z1 + cos(t)·X2: 2-body on {0,1}, 1-body on {2} → no collision."""
        t = sp.Symbol("t")
        _qs, q = _qubit_system(3)
        H = sp.sin(t) * q[0].Z * q[1].Z + sp.cos(t) * q[2].X
        groups = factor_td_hamiltonian(H, t, param_dict={})
        assert check_dressing_collision(groups) is False


# ═════════════════════════════════════════════════════════════════════════════
# trotterize_td
# ═════════════════════════════════════════════════════════════════════════════

class TestTrotterizeTD:

    def test_n_steps_and_dt(self):
        t = sp.Symbol("t")
        _qs, q = _qubit_system(1)
        H = sp.sin(t) * q[0].Z
        T, n = 2.0, 8

        segs = trotterize_td(H, t, T, n, param_dict={})
        assert len(segs) == n
        for _, dt in segs:
            assert abs(dt - T / n) < 1e-12

    def test_midpoint_values(self):
        """At each midpoint τ_i, coefficient equals sin(τ_i)."""
        t = sp.Symbol("t")
        _qs, q = _qubit_system(1)
        H = sp.sin(t) * q[0].Z
        T, n = np.pi, 4

        segs = trotterize_td(H, t, T, n, param_dict={})
        dt = T / n
        for i, (ti_seg, _) in enumerate(segs):
            tau = (i + 0.5) * dt
            expected = np.sin(tau)
            if abs(expected) < 1e-12:
                assert len(ti_seg.ham) == 0
            else:
                ((_prod, coef),) = ti_seg.ham
                assert abs(float(coef) - expected) < 1e-9

    def test_param_substitution_required(self):
        """Unresolved parameter symbols should raise."""
        t, x = sp.symbols("t x")
        _qs, q = _qubit_system(1)
        H = sp.sin(x * t) * q[0].Z
        with pytest.raises(ValueError):
            trotterize_td(H, t, 1.0, 4, param_dict={})


# ═════════════════════════════════════════════════════════════════════════════
# build_channel_envelopes
# ═════════════════════════════════════════════════════════════════════════════

class TestBuildChannelEnvelopes:

    def test_two_disjoint_channels(self):
        t = sp.Symbol("t")
        T, fs = 1.0, 100.0
        groups = [
            (sp.sin(2 * sp.pi * t), {0: 1.0}),
            (sp.cos(2 * sp.pi * t), {1: 2.0}),
        ]
        wf = build_channel_envelopes(groups, t, T, fs)

        n = int(T * fs)
        ts = np.linspace(0, T, n, endpoint=False)
        assert set(wf.keys()) == {0, 1}
        assert np.allclose(wf[0], np.sin(2 * np.pi * ts))
        assert np.allclose(wf[1], 2.0 * np.cos(2 * np.pi * ts))

    def test_shared_channel_sums(self):
        """Two groups targeting the same channel → their samples add."""
        t = sp.Symbol("t")
        T, fs = 1.0, 100.0
        groups = [
            (sp.sin(2 * sp.pi * t), {0: 1.0}),
            (sp.cos(2 * sp.pi * t), {0: 1.0}),
        ]
        wf = build_channel_envelopes(groups, t, T, fs)
        n = int(T * fs)
        ts = np.linspace(0, T, n, endpoint=False)
        expected = np.sin(2 * np.pi * ts) + np.cos(2 * np.pi * ts)
        assert np.allclose(wf[0], expected)

    def test_constant_envelope_sampled(self):
        """envelope=1 should give a flat waveform equal to the weight."""
        t = sp.Symbol("t")
        T, fs = 1.0, 50.0
        groups = [(sp.Integer(1), {3: 0.7})]
        wf = build_channel_envelopes(groups, t, T, fs)
        n = int(T * fs)
        assert wf[3].shape == (n,)
        assert np.allclose(wf[3], 0.7)
