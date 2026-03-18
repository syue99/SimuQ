"""
Unit tests for the transport layer: aod_channel and tweezer_mapper.

All tests run without a live PulseDSL / MMIO session — the mapper returns
plain dict ops, so hardware dependencies are zero.

Coverage
--------
- encode_positions / make_aod_pulse
- classify_instruction (all rydberg2d instruction types)
- rest_positions / _pair_target_positions
- _dressing_ops / _cz_ops / _native_ops (op type, logging)
- map_evaluated_H (priority: dressing suppresses ZZ)
- map_hlist (3-segment branch, time cursor, TransportLog)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from unittest.mock import MagicMock

from simuq import QSystem, Qubit
from simuq.hamiltonian import TIHamiltonian, productHamiltonian

from aod_channel import encode_positions, make_aod_pulse, AOD_FREQ_MHZ
from tweezer_mapper import (
    TransportLog, TweezerMapper,
    classify_instruction,
    _op_aod, _op_play, _op_delay,
    C_6,
)


# ── Fake instruction helper ───────────────────────────────────────────────────

def _fake_ins(name, nativeness="native"):
    ins = MagicMock()
    ins.name = name
    ins.nativeness = nativeness
    return ins


# ── Fake boxes helper ─────────────────────────────────────────────────────────

def _make_box(ins_specs, duration=1.0):
    """
    ins_specs: list of (name, nativeness, ins_lvars)
    Returns a box in the with_sys_ham format:
        ([((li, ji), ins, h_eval, ins_lvars), ...], duration)
    """
    entries = []
    for idx, (name, nat, lvars) in enumerate(ins_specs):
        ins = _fake_ins(name, nat)
        entries.append(((0, idx), ins, MagicMock(), lvars))
    return (entries, duration)


# ── Fake Hj helper ────────────────────────────────────────────────────────────

def _make_Hj(site_ops):
    """
    site_ops: dict {site_int: op_str}  e.g. {0: 'Z'} or {0: 'Z', 1: 'Z'}
    Returns a TIHamiltonian with one product term, coefficient 1.
    """
    qs = QSystem()
    q  = [Qubit(qs) for _ in range(max(site_ops.keys()) + 1)]
    prod = productHamiltonian()
    prod.d = dict(site_ops)
    return TIHamiltonian(qs.sites_type, qs.sites_name, [(prod, 1)])


# ── aod_channel ───────────────────────────────────────────────────────────────

class TestAODChannel:

    def test_encode_empty(self):
        assert encode_positions([]) == 0.0

    def test_encode_origin(self):
        assert encode_positions([(0.0, 0.0)]) == pytest.approx(0.0)

    def test_encode_single(self):
        assert encode_positions([(3.0, 4.0)]) == pytest.approx(5.0)

    def test_encode_max(self):
        # max(5, 13) = 13
        assert encode_positions([(3.0, 4.0), (5.0, 12.0)]) == pytest.approx(13.0)

    def test_make_aod_pulse_type(self):
        p = make_aod_pulse([(1.0, 0.0)], ramp_time=10.0)
        assert p["type"] == "aod"

    def test_make_aod_pulse_freq(self):
        p = make_aod_pulse([(1.0, 0.0)], ramp_time=10.0)
        assert p["freq_MHz"] == AOD_FREQ_MHZ

    def test_make_aod_pulse_duration(self):
        p = make_aod_pulse([(1.0, 0.0)], ramp_time=42.0)
        assert p["duration"] == pytest.approx(42.0)

    def test_make_aod_pulse_positions_recorded(self):
        pos = [(1.0, 2.0), (3.0, 4.0)]
        p   = make_aod_pulse(pos, ramp_time=5.0)
        assert p["positions"] == pos


# ── classify_instruction ──────────────────────────────────────────────────────

class TestClassifyInstruction:

    def test_detuning(self):
        assert classify_instruction(_fake_ins("Detuning of site 0")) == ('detuning', 0)
        assert classify_instruction(_fake_ins("Detuning of site 2")) == ('detuning', 2)

    def test_rabi(self):
        assert classify_instruction(_fake_ins("Rabi of site 1")) == ('rabi', 1)

    def test_dressing(self):
        assert classify_instruction(_fake_ins("dressing gloabl potential")) == ('dressing',)

    def test_zz(self):
        assert classify_instruction(_fake_ins("c01_zz", "derived")) == ('zz', 0, 1)
        assert classify_instruction(_fake_ins("c12_zz", "derived")) == ('zz', 1, 2)

    def test_unknown(self):
        assert classify_instruction(_fake_ins("mystery")) == ('unknown',)


# ── TweezerMapper geometry ────────────────────────────────────────────────────

def _mapper_3q(ramp_time=0.01):
    """3-qubit mapper with atoms at (0,0), (2,0), (1, 1.73) (equilateral triangle)."""
    sol_gvars = [2.0, 0.0,   1.0, 1.73]   # atoms 1, 2; atom 0 at origin
    return TweezerMapper(n_qubits=3, sol_gvars=sol_gvars, boxes=[], ramp_time=ramp_time)


class TestTweezerMapperGeometry:

    def test_rest_positions_atom0_at_origin(self):
        m = _mapper_3q()
        pos = m.rest_positions()
        assert pos[0] == (0.0, 0.0)

    def test_rest_positions_correct(self):
        m = _mapper_3q()
        pos = m.rest_positions()
        assert pos[1] == pytest.approx((2.0, 0.0))
        assert pos[2] == pytest.approx((1.0, 1.73))

    def test_pair_target_distance_achieved(self):
        m   = _mapper_3q()
        R   = 1.5
        pos = m._pair_target_positions(0, 1, R)
        dist = np.sqrt((pos[0][0] - pos[1][0])**2 + (pos[0][1] - pos[1][1])**2)
        assert dist == pytest.approx(R, rel=1e-6)

    def test_pair_target_other_atoms_unchanged(self):
        m   = _mapper_3q()
        pos = m._pair_target_positions(0, 1, 1.5)
        # atom 2 must not move
        assert pos[2] == pytest.approx(m.rest_positions()[2])

    def test_pair_target_symmetric_about_midpoint(self):
        m   = _mapper_3q()
        R   = 1.2
        pos = m._pair_target_positions(0, 1, R)
        rest = m.rest_positions()
        mid_rest = ((rest[0][0] + rest[1][0]) / 2,
                    (rest[0][1] + rest[1][1]) / 2)
        mid_new  = ((pos[0][0] + pos[1][0]) / 2,
                    (pos[0][1] + pos[1][1]) / 2)
        assert mid_new == pytest.approx(mid_rest, rel=1e-6)


# ── _dressing_ops ─────────────────────────────────────────────────────────────

class TestDressingOps:

    def setup_method(self):
        self.m = _mapper_3q()

    def test_emits_three_ops(self):
        ops = self.m._dressing_ops(o_coef=0.5, duration=1.0, t_cursor=0.0)
        assert len(ops) == 3

    def test_first_and_last_are_aod(self):
        ops = self.m._dressing_ops(o_coef=0.5, duration=1.0, t_cursor=0.0)
        assert ops[0]["op"] == "aod"
        assert ops[2]["op"] == "aod"

    def test_middle_is_delay(self):
        ops = self.m._dressing_ops(o_coef=0.5, duration=1.5, t_cursor=0.0)
        assert ops[1]["op"] == "delay"
        assert ops[1]["duration"] == pytest.approx(1.5)

    def test_logs_dressing_move(self):
        self.m._dressing_ops(o_coef=0.7, duration=1.0, t_cursor=2.5)
        assert len(self.m.log.dressing_moves) == 1
        assert self.m.log.dressing_moves[0].o_coef == pytest.approx(0.7)
        assert self.m.log.dressing_moves[0].t_start == pytest.approx(2.5)

    def test_logs_positions(self):
        self.m._dressing_ops(o_coef=0.5, duration=1.0, t_cursor=0.0)
        logged_pos = self.m.log.dressing_moves[0].positions
        assert logged_pos == self.m.rest_positions()


# ── _cz_ops ───────────────────────────────────────────────────────────────────

class TestCZOps:

    def setup_method(self):
        self.m = _mapper_3q()

    def test_zero_J_returns_empty(self):
        ops = self.m._cz_ops(0, 1, J=0.0, duration=1.0, t_cursor=0.0)
        assert ops == []

    def test_emits_three_ops(self):
        ops = self.m._cz_ops(0, 1, J=1.0, duration=1.0, t_cursor=0.0)
        assert len(ops) == 3

    def test_first_is_aod_to_target(self):
        J   = 2.0
        ops = self.m._cz_ops(0, 1, J=J, duration=1.0, t_cursor=0.0)
        R_expected = (C_6 / J) ** (1.0 / 6.0)
        target_positions = ops[0]["positions"]
        dist = np.sqrt(
            (target_positions[0][0] - target_positions[1][0])**2 +
            (target_positions[0][1] - target_positions[1][1])**2
        )
        assert dist == pytest.approx(R_expected, rel=1e-5)

    def test_last_is_aod_to_rest(self):
        ops  = self.m._cz_ops(0, 1, J=1.0, duration=1.0, t_cursor=0.0)
        rest = self.m.rest_positions()
        assert ops[2]["positions"] == rest

    def test_middle_is_delay(self):
        ops = self.m._cz_ops(0, 1, J=1.0, duration=2.3, t_cursor=0.0)
        assert ops[1]["op"] == "delay"
        assert ops[1]["duration"] == pytest.approx(2.3)

    def test_logs_cz_move(self):
        self.m._cz_ops(1, 2, J=0.5, duration=1.0, t_cursor=3.0)
        assert len(self.m.log.cz_moves) == 1
        ev = self.m.log.cz_moves[0]
        assert ev.pair    == (1, 2)
        assert ev.J       == pytest.approx(0.5)
        assert ev.t_start == pytest.approx(3.0)
        R_expected = (C_6 / 0.5) ** (1.0 / 6.0)
        assert ev.R_target == pytest.approx(R_expected, rel=1e-5)


# ── _native_ops ───────────────────────────────────────────────────────────────

class TestNativeOps:

    def setup_method(self):
        self.m = _mapper_3q()

    def test_detuning_channel_index(self):
        ops = self.m._native_ops(('detuning', 2), [0.8], duration=1.0)
        assert len(ops) == 1
        assert ops[0]["op"]      == "play"
        assert ops[0]["channel"] == 2

    def test_detuning_amplitude(self):
        ops = self.m._native_ops(('detuning', 0), [1.5], duration=1.0)
        assert ops[0]["amplitude"] == pytest.approx(1.5)

    def test_rabi_channel_index(self):
        # n=3, so rabi site 1 → ch[3+1] = ch[4]
        ops = self.m._native_ops(('rabi', 1), [0.5, 0.3], duration=1.0)
        assert ops[0]["channel"] == 4

    def test_rabi_phase(self):
        ops = self.m._native_ops(('rabi', 0), [1.0, 0.7], duration=1.0)
        assert ops[0]["phase"] == pytest.approx(0.7)

    def test_unknown_returns_empty(self):
        ops = self.m._native_ops(('unknown',), [], duration=1.0)
        assert ops == []


# ── map_evaluated_H ───────────────────────────────────────────────────────────

class TestMapEvaluatedH:

    def _mapper_with_boxes(self, specs, duration=1.0):
        """specs: list of (name, nativeness, ins_lvars) per instruction."""
        boxes = [_make_box(specs, duration)]
        return TweezerMapper(n_qubits=3, sol_gvars=[2.0, 0.0, 1.0, 1.73],
                             boxes=boxes, ramp_time=0.01)

    def test_dressing_box_emits_aod(self):
        m   = self._mapper_with_boxes([("dressing gloabl potential", "native", [0.5])])
        ops = m.map_evaluated_H(duration=1.0, t_cursor=0.0)
        aod_ops = [o for o in ops if o["op"] == "aod"]
        assert len(aod_ops) >= 2

    def test_dressing_suppresses_zz(self):
        """If dressing is active, ZZ entries in the same box must be skipped."""
        m = self._mapper_with_boxes([
            ("dressing gloabl potential", "native",  [0.5]),
            ("c01_zz",                   "derived",  [1.0]),
        ])
        ops = m.map_evaluated_H(duration=1.0, t_cursor=0.0)
        # Only dressing AOD ops, no CZ target positions (which differ from rest)
        # The CZ _cz_ops would produce positions ≠ rest; dressing produces rest positions.
        cz_logs = m.log.cz_moves
        assert len(cz_logs) == 0

    def test_zz_only_box_emits_cz(self):
        m   = self._mapper_with_boxes([("c01_zz", "derived", [2.0])])
        ops = m.map_evaluated_H(duration=1.0, t_cursor=0.0)
        aod_ops = [o for o in ops if o["op"] == "aod"]
        assert len(aod_ops) >= 2
        assert len(m.log.cz_moves) == 1

    def test_native_only_no_aod(self):
        m   = self._mapper_with_boxes([("Detuning of site 0", "native", [0.5])])
        ops = m.map_evaluated_H(duration=1.0, t_cursor=0.0)
        aod_ops = [o for o in ops if o["op"] == "aod"]
        assert len(aod_ops) == 0

    def test_native_play_duration(self):
        m   = self._mapper_with_boxes([("Rabi of site 1", "native", [0.4, 0.1])])
        ops = m.map_evaluated_H(duration=2.5, t_cursor=0.0)
        play_ops = [o for o in ops if o["op"] == "play"]
        assert any(o["duration"] == pytest.approx(2.5) for o in play_ops)


# ── map_hlist ─────────────────────────────────────────────────────────────────

class TestMapHlist:

    def _build_hlist(self, tau=0.3, kick=np.pi / 4, T=1.0):
        """Build a 3-segment H_list with fake TIHamiltonians."""
        # Use a simple 1-qubit Z Hamiltonian as evaluated_H placeholder
        qs = QSystem()
        q  = [Qubit(qs)]
        H_eval = 1.0 * q[0].Z   # simplest non-trivial TIHam
        Hj     = _make_Hj({0: 'Z'})
        return [
            [H_eval, tau],
            [Hj,     kick],
            [H_eval, T - tau],
        ]

    def test_returns_ops_and_log(self):
        m = _mapper_3q()
        ops, log = m.map_hlist(self._build_hlist())
        assert isinstance(ops, list)
        assert isinstance(log, TransportLog)

    def test_log_is_fresh_per_call(self):
        m    = _mapper_3q()
        _,  log1 = m.map_hlist(self._build_hlist())
        _, log2  = m.map_hlist(self._build_hlist())
        # Each call resets the log — counts should match, not accumulate
        assert len(log1.cz_moves) == len(log2.cz_moves)

