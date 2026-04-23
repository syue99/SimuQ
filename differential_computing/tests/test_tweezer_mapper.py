"""
Unit tests for the transport layer: aod_channel and tweezer_mapper.

All tests run without a live PulseDSL / MMIO session — the mapper returns
plain dict ops, so hardware dependencies are zero.

Coverage
--------
- encode_positions / make_aod_pulse
- classify_instruction (all rydberg2d instruction types)
- interaction_positions / gate_positions
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
from pulse_ledger import PulseLedger, GATE_ZONE, idle_position


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
        h_eval = TIHamiltonian.identity(
            ["qubit"] * 3, [f"q{i}" for i in range(3)])
        entries.append(((0, idx), ins, h_eval, lvars))
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

    def test_gate_positions_pair_distance(self):
        m   = _mapper_3q()
        R   = 1.5
        pos = m.gate_positions(0, 1, R)
        dist = np.sqrt((pos[0][0] - pos[1][0])**2 + (pos[0][1] - pos[1][1])**2)
        assert dist == pytest.approx(R, rel=1e-6)

    def test_gate_positions_others_at_idle(self):
        m   = _mapper_3q()
        pos = m.gate_positions(0, 1, 1.5)
        # atom 2 should be at its idle position
        assert pos[2] == pytest.approx(idle_position(2))

    def test_gate_positions_pair_centered_at_gate_zone(self):
        m   = _mapper_3q()
        R   = 1.2
        pos = m.gate_positions(0, 1, R)
        mid = ((pos[0][0] + pos[1][0]) / 2,
               (pos[0][1] + pos[1][1]) / 2)
        assert mid == pytest.approx(GATE_ZONE, rel=1e-6)


# ── _dressing_ops ─────────────────────────────────────────────────────────────

class TestDressingOps:

    def setup_method(self):
        self.m = _mapper_3q()
        # _dressing_ops now uses self.ledger, so initialise it
        self.m.ledger = PulseLedger(self.m.n)

    def test_skips_aod_when_already_at_interaction(self):
        """Atoms start at interaction zone — no AOD needed."""
        ops = self.m._dressing_ops(o_coef=0.5, duration=1.0, t_cursor=0.0)
        assert len(ops) == 1  # play on dressing channel only, no aod
        assert ops[0]["op"] == "play"
        assert ops[0]["channel"] == 2 * self.m.n  # dressing channel

    def test_emits_aod_when_not_at_interaction(self):
        """Atoms NOT at interaction zone — AOD move required."""
        self.m.current_positions = [(999.0, 999.0)] * self.m.n
        ops = self.m._dressing_ops(o_coef=0.5, duration=1.0, t_cursor=0.0)
        assert len(ops) == 2  # aod + play
        assert ops[0]["op"] == "aod"
        assert ops[1]["op"] == "play"

    def test_dressing_play_duration(self):
        ops = self.m._dressing_ops(o_coef=0.5, duration=1.5, t_cursor=0.0)
        play_ops = [o for o in ops if o["op"] == "play"]
        assert play_ops[0]["duration"] == pytest.approx(1.5)

    def test_logs_dressing_move(self):
        self.m._dressing_ops(o_coef=0.7, duration=1.0, t_cursor=2.5)
        assert len(self.m.log.dressing_moves) == 1
        assert self.m.log.dressing_moves[0].o_coef == pytest.approx(0.7)
        assert self.m.log.dressing_moves[0].t_start == pytest.approx(2.5)

    def test_logs_positions(self):
        self.m._dressing_ops(o_coef=0.5, duration=1.0, t_cursor=0.0)
        logged_pos = self.m.log.dressing_moves[0].positions
        assert logged_pos == self.m.rest_positions()

    def test_ledger_records_play_only_when_no_move(self):
        """Already at interaction → ledger has play only (no aod entry)."""
        self.m._dressing_ops(o_coef=0.5, duration=1.0, t_cursor=0.0)
        assert len(self.m.ledger.entries) == 1
        assert self.m.ledger.entries[0].op_type == "play"
        assert self.m.ledger.entries[0].channel_kind == "dressing"

    def test_ledger_records_aod_and_play_when_moved(self):
        """Not at interaction → ledger has aod + play."""
        self.m.current_positions = [(999.0, 999.0)] * self.m.n
        self.m._dressing_ops(o_coef=0.5, duration=1.0, t_cursor=0.0)
        assert len(self.m.ledger.entries) == 2
        assert self.m.ledger.entries[0].op_type == "aod"
        assert self.m.ledger.entries[1].op_type == "play"

    def test_updates_zones_to_interaction(self):
        # Start from some non-interaction state
        self.m.current_zones = ["idle"] * self.m.n
        self.m._dressing_ops(o_coef=0.5, duration=1.0, t_cursor=0.0)
        assert self.m.current_zones == ["interaction"] * self.m.n


# ── _cz_ops ───────────────────────────────────────────────────────────────────

class TestCZOps:

    def setup_method(self):
        self.m = _mapper_3q()
        self.m.ledger = PulseLedger(self.m.n)

    def test_zero_J_returns_empty(self):
        ops = self.m._cz_ops(0, 1, J=0.0, duration=1.0, t_cursor=0.0)
        assert ops == []

    def test_emits_two_ops(self):
        ops = self.m._cz_ops(0, 1, J=1.0, duration=1.0, t_cursor=0.0)
        # aod + delay (no trailing aod — atoms stay at gate zone)
        assert len(ops) == 2

    def test_first_is_aod_to_gate_zone(self):
        J   = 2.0
        ops = self.m._cz_ops(0, 1, J=J, duration=1.0, t_cursor=0.0)
        R_expected = (C_6 / J) ** (1.0 / 6.0)
        target_positions = ops[0]["positions"]
        dist = np.sqrt(
            (target_positions[0][0] - target_positions[1][0])**2 +
            (target_positions[0][1] - target_positions[1][1])**2
        )
        assert dist == pytest.approx(R_expected, rel=1e-5)

    def test_second_is_play_on_zz_channel(self):
        ops = self.m._cz_ops(0, 1, J=1.0, duration=2.3, t_cursor=0.0)
        assert ops[1]["op"] == "play"
        assert ops[1]["channel"] == 2 * self.m.n + 1  # ZZ channel
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

    def test_zones_set_to_gate_and_idle(self):
        self.m._cz_ops(0, 1, J=1.0, duration=1.0, t_cursor=0.0)
        assert self.m.current_zones[0] == "gate"
        assert self.m.current_zones[1] == "gate"
        assert self.m.current_zones[2] == "idle"

    def test_pair_at_gate_zone_center(self):
        """Pair should be centered around GATE_ZONE."""
        from pulse_ledger import GATE_ZONE
        J = 2.0
        self.m._cz_ops(0, 1, J=J, duration=1.0, t_cursor=0.0)
        p0 = self.m.current_positions[0]
        p1 = self.m.current_positions[1]
        mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
        assert mid == pytest.approx(GATE_ZONE, rel=1e-6)


# ── _native_ops ───────────────────────────────────────────────────────────────

class TestNativeOps:

    def setup_method(self):
        self.m = _mapper_3q()
        self.m.ledger = PulseLedger(self.m.n)

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

    def test_retrieval_from_gate_zone(self):
        """If qubit is at gate zone, single-qubit op should return it first."""
        # Simulate qubit 0 being at gate zone from a prior ZZ
        self.m.current_zones[0] = "gate"
        self.m.current_positions[0] = (1000.0, 1000.0)
        ops = self.m._native_ops(('detuning', 0), [0.5], duration=1.0)
        # Should emit: AOD return + play = 2 ops
        assert len(ops) == 2
        assert ops[0]["op"] == "aod"
        assert ops[1]["op"] == "play"
        # After retrieval, zone should be interaction
        assert self.m.current_zones[0] == "interaction"

    def test_no_retrieval_when_at_interaction(self):
        """No AOD move needed if qubit already at interaction zone."""
        ops = self.m._native_ops(('detuning', 0), [0.5], duration=1.0)
        assert len(ops) == 1
        assert ops[0]["op"] == "play"

    def test_ledger_records_play(self):
        self.m._native_ops(('rabi', 1), [0.5, 0.3], duration=1.0)
        play_entries = self.m.ledger.play_entries()
        assert len(play_entries) == 1
        assert play_entries[0].channel_kind == "rabi"
        assert play_entries[0].target_qubits == [1]
        assert play_entries[0].amplitude == pytest.approx(0.5)
        assert play_entries[0].phase == pytest.approx(0.3)


# ── map_evaluated_H ───────────────────────────────────────────────────────────

class TestMapEvaluatedH:

    def _mapper_with_boxes(self, specs, duration=1.0):
        """specs: list of (name, nativeness, ins_lvars) per instruction."""
        boxes = [_make_box(specs, duration)]
        m = TweezerMapper(n_qubits=3, sol_gvars=[2.0, 0.0, 1.0, 1.73],
                          boxes=boxes, ramp_time=0.01)
        m.ledger = PulseLedger(m.n)
        return m

    def _fake_H(self):
        """A dummy TIHamiltonian placeholder for evaluated_H arg."""
        return MagicMock()

    def test_dressing_box_skips_aod_when_already_at_interaction(self):
        """Atoms start at interaction → no AOD, just delay + ledger play."""
        m   = self._mapper_with_boxes([("dressing gloabl potential", "native", [0.5])])
        ops = m.map_evaluated_H(duration=1.0, t_cursor=0.0)
        aod_ops = [o for o in ops if o["op"] == "aod"]
        assert len(aod_ops) == 0  # already at interaction zone

    def test_dressing_box_emits_aod_when_moved(self):
        """Atoms NOT at interaction → AOD emitted."""
        m   = self._mapper_with_boxes([("dressing gloabl potential", "native", [0.5])])
        m.current_positions = [(999.0, 999.0)] * m.n
        ops = m.map_evaluated_H(duration=1.0, t_cursor=0.0)
        aod_ops = [o for o in ops if o["op"] == "aod"]
        assert len(aod_ops) >= 1

    def test_dressing_and_zz_both_emitted(self):
        """Both dressing and ZZ are emitted — solver uses both together."""
        m = self._mapper_with_boxes([
            ("dressing gloabl potential", "native",  [0.5]),
            ("c01_zz",                   "derived",  [1.0]),
        ])
        ops = m.map_evaluated_H(duration=1.0, t_cursor=0.0)
        assert len(m.log.dressing_moves) > 0, "Dressing should be emitted"
        assert len(m.log.cz_moves) > 0, "ZZ should also be emitted"

    def test_zz_only_box_emits_cz(self):
        m   = self._mapper_with_boxes([("c01_zz", "derived", [2.0])])
        ops = m.map_evaluated_H(duration=1.0, t_cursor=0.0)
        aod_ops = [o for o in ops if o["op"] == "aod"]
        assert len(aod_ops) >= 1
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

    def test_returns_ops_log_and_ledger(self):
        m = _mapper_3q()
        ops, log, ledger = m.map_hlist(self._build_hlist())
        assert isinstance(ops, list)
        assert isinstance(log, TransportLog)
        assert isinstance(ledger, PulseLedger)

    def test_log_is_fresh_per_call(self):
        m = _mapper_3q()
        _, log1, _ = m.map_hlist(self._build_hlist())
        _, log2, _ = m.map_hlist(self._build_hlist())
        # Each call resets the log — counts should match, not accumulate
        assert len(log1.cz_moves) == len(log2.cz_moves)

    def test_ledger_is_fresh_per_call(self):
        m = _mapper_3q()
        _, _, ledger1 = m.map_hlist(self._build_hlist())
        _, _, ledger2 = m.map_hlist(self._build_hlist())
        assert len(ledger1.entries) == len(ledger2.entries)

