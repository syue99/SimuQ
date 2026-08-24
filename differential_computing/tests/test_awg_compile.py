"""
test_awg_compile.py — AWG waveform synthesis (the end-to-end waveform step).

Covers, bottom-up:
  1. ConstantTone / ChirpTone against closed forms (analytic reference).
  2. tone_waveform dispatch (constant vs chirp from Tone.frequency_end).
  3. compile_waveforms summation/indexing on a hand-built Sched (co-temporal
     COMB entries add; gaps stay zero) — no RUN, no MMIO writes.
  4. Transport chirps: AodNode.positions_from → crossed X/Y combs whose
     frequency endpoints match the coord_to_freq calibration; parked atoms
     hold a constant tone.
  5. Mapper threading: every aod op emitted by map_hlist carries the
     start-of-move positions.
  6. End-to-end 2q: compile → map_hlist_tree → to_physical → RUN →
     compile_waveforms produces per-channel sample arrays whose span matches
     the schedule and whose channel activity matches the physical layout.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/Users/syue99/research/RISC-Q/PulseDSL/src/DSL/")

import numpy as np
import sympy as sp

import pulse_tree as pt
import physical_channels as pc
from awg_compile import (ConstantTone, ChirpTone, tone_waveform,
                         compile_waveforms)


# ── 1. analytic waveform checks ───────────────────────────────────────────────

def test_constant_tone_matches_closed_form():
    tone = ConstantTone(amplitude=2.5, phase=0.3, freq_mhz=80.0)
    t = np.linspace(0.0, 100.0, 501)
    expect = 2.5 * np.exp(1j * (2 * np.pi * 80.0 * 1e-3 * t + 0.3))
    assert np.allclose(tone(t), expect)
    # zero frequency = plain constant envelope
    flat = ConstantTone(1.7, phase=-0.2)
    assert np.allclose(flat(t), 1.7 * np.exp(-0.2j))


def test_chirp_tone_endpoints_and_phase_continuity():
    T = 1000.0
    chirp = ChirpTone(amplitude=1.0, f0_mhz=90.0, f1_mhz=110.0, duration_ns=T)
    # minimum-jerk profile by default; endpoints + midpoint (time-reversal
    # symmetry keeps f(T/2) at the mean frequency)
    assert chirp.profile == "minjerk"
    assert np.isclose(chirp.instantaneous_freq_mhz(0.0), 90.0)
    assert np.isclose(chirp.instantaneous_freq_mhz(T / 2), 100.0)
    assert np.isclose(chirp.instantaneous_freq_mhz(T), 110.0)
    # numerical instantaneous frequency (phase derivative) matches
    t = np.linspace(0.0, T, 20001)
    phase = np.unwrap(np.angle(chirp(t)))
    f_num = np.gradient(phase, t) / (2 * np.pi * 1e-3)   # MHz
    f_ref = chirp.instantaneous_freq_mhz(t)
    assert np.allclose(f_num[10:-10], f_ref[10:-10], atol=1e-3)
    # starts at the declared phase, |A| constant
    assert np.isclose(np.angle(chirp(0.0)), 0.0, atol=1e-12)
    assert np.allclose(np.abs(chirp(t)), 1.0)


def test_chirp_tone_minjerk_boundary_conditions():
    # Cicali et al. PRApplied 24, 024070 (2025) Eq. (6): position (here
    # frequency) has zero velocity AND zero acceleration at both endpoints.
    T = 1000.0
    chirp = ChirpTone(1.0, 90.0, 110.0, duration_ns=T)
    f = chirp.instantaneous_freq_mhz(np.linspace(0.0, T, 100001))
    dt = T / 100000
    v = np.gradient(f, dt)                 # MHz/ns
    a = np.gradient(v, dt)
    v_peak, a_peak = np.abs(v).max(), np.abs(a).max()
    assert abs(v[0]) < 1e-3 * v_peak and abs(v[-1]) < 1e-3 * v_peak
    assert abs(a[1]) < 1e-2 * a_peak and abs(a[-2]) < 1e-2 * a_peak
    # peak sweep rate of the min-jerk profile is 15/8 x the linear rate
    lin_rate = (110.0 - 90.0) / T
    assert np.isclose(v_peak, 15.0 / 8.0 * lin_rate, rtol=1e-3)
    # exact closed-form frequency at the quarter point: s=1/4 ->
    # 10/64 - 15/256 + 6/1024 = 0.103515625
    assert np.isclose(chirp.instantaneous_freq_mhz(T / 4),
                      90.0 + 20.0 * 0.103515625)


def test_chirp_tone_linear_profile_kept_for_comparison():
    T = 500.0
    lin = ChirpTone(1.0, 80.0, 95.0, duration_ns=T, profile="linear")
    t = np.linspace(0.0, T, 2001)
    assert np.allclose(lin.instantaneous_freq_mhz(t), 80.0 + 15.0 * t / T)
    # old closed form for the linear phase
    slope = 15.0 / T
    expect = np.exp(1j * 2 * np.pi * 1e-3 * (80.0 * t + 0.5 * slope * t * t))
    assert np.allclose(lin(t), expect)
    # both profiles accrue the same total phase (same mean frequency)
    mj = ChirpTone(1.0, 80.0, 95.0, duration_ns=T)
    assert np.isclose(np.angle(mj(T) / lin(T)), 0.0, atol=1e-9)


def test_tone_waveform_dispatch():
    const = tone_waveform(pt.Tone(0, 80.0, 1.0, 0.1), 500)
    assert isinstance(const, ConstantTone)
    same = tone_waveform(pt.Tone(0, 80.0, 1.0, 0.0, frequency_end=80.0), 500)
    assert isinstance(same, ConstantTone)
    chirp = tone_waveform(pt.Tone(0, 80.0, 1.0, 0.0, frequency_end=95.0), 500)
    assert isinstance(chirp, ChirpTone)
    assert chirp.f0_mhz == 80.0 and chirp.f1_mhz == 95.0
    assert chirp.duration_ns == 500


# ── 3. sampler summation on a hand-built schedule ─────────────────────────────

class _FakeEntry:
    def __init__(self, pulse, t0, t1):
        self._ScheduleEntry__pulse = pulse
        self._ScheduleEntry__t0 = t0
        self._ScheduleEntry__t1 = t1


class _FakePulse:
    def __init__(self, waveform=None, amplitude=0.0, phase=0.0, frequency=0.0):
        self.waveform = waveform
        self.amplitude = amplitude
        self.phase = phase
        self.frequency = frequency


class _FakeSched:
    def __init__(self, rows):
        self._Sched__schedule = rows


def test_compile_waveforms_sums_cotemporal_entries():
    # channel 0: two co-temporal tones (a COMB) on [0, 200), then silence,
    # then one tone on [300, 400)
    tone_a = ConstantTone(1.0, 0.0, 80.0)
    tone_b = ConstantTone(0.5, 0.4, 90.0)
    tone_c = ConstantTone(2.0, 0.0, 0.0)
    rows = [[
        _FakeEntry(_FakePulse(tone_a), 0, 200),
        _FakeEntry(_FakePulse(tone_b), 0, 200),
        _FakeEntry(_FakePulse(tone_c), 300, 400),
    ]]
    t, waves = compile_waveforms(_FakeSched(rows), n_channels=1, dt_ns=1.0)
    w = waves[0]
    assert len(t) == 400 and len(w) == 400
    # comb window = sum of the two tones (t relative to entry start = t here)
    expect = tone_a(t[:200]) + tone_b(t[:200])
    assert np.allclose(w[:200], expect)
    # gap stays exactly zero
    assert np.all(w[200:300] == 0)
    # second entry evaluated relative to ITS start
    assert np.allclose(w[300:400], tone_c(t[300:400] - 300.0))


def test_compile_waveforms_fallback_and_empty():
    # entry without waveform → constant fallback from pulse fields
    rows = [[_FakeEntry(_FakePulse(None, amplitude=1.5, phase=0.2), 0, 50)],
            []]   # silent channel present in output
    t, waves = compile_waveforms(_FakeSched(rows), dt_ns=1.0)
    assert np.allclose(waves[0][:50], 1.5 * np.exp(0.2j))
    assert np.all(waves[1] == 0) and len(waves[1]) == len(t)


# ── 4. transport chirps from the physical layer ───────────────────────────────

def test_transport_combs_encode_move_as_chirps():
    start = [(0.0, 0.0), (10.0, 0.0)]
    target = [(995.0, 1000.0), (10.0, 0.0)]     # atom 0 moves, atom 1 parked
    aod = pt.AodNode(positions=target, ramp_time=0.01, positions_from=start)
    para = pc._transport_combs(aod)
    combs = {c.channel: c for c in para.children}
    xc, yc = combs[pc.TRANSPORT_AOD_X], combs[pc.TRANSPORT_AOD_Y]
    assert xc.kind == "transport" and yc.kind == "transport"
    # atom 0: chirp from f(start) to f(target) on both axes
    assert np.isclose(xc.tones[0].frequency, pc.coord_to_freq(0.0))
    assert np.isclose(xc.tones[0].frequency_end, pc.coord_to_freq(995.0))
    assert np.isclose(yc.tones[0].frequency_end, pc.coord_to_freq(1000.0))
    # atom 1 parked: constant hold tone at its position
    assert np.isclose(xc.tones[1].frequency, pc.coord_to_freq(10.0))
    assert xc.tones[1].frequency_end is None
    assert yc.tones[1].frequency_end is None


def test_transport_combs_without_start_positions_hold_targets():
    aod = pt.AodNode(positions=[(5.0, -3.0)], ramp_time=0.01)
    para = pc._transport_combs(aod)
    for comb in para.children:
        assert comb.tones[0].frequency_end is None
    combs = {c.channel: c for c in para.children}
    assert np.isclose(combs[pc.TRANSPORT_AOD_X].tones[0].frequency,
                      pc.coord_to_freq(5.0))
    assert np.isclose(combs[pc.TRANSPORT_AOD_Y].tones[0].frequency,
                      pc.coord_to_freq(-3.0))


# ── 5/6. mapper threading + end-to-end 2q ─────────────────────────────────────

def _compile_2q():
    from simuq import QSystem, Qubit
    from simuq.braket.diffQC_provider import diffQCProvider
    from observable_program_generator import observable_program_generator
    from tweezer_mapper import TweezerMapper

    x = sp.Symbol("x")
    T, x_val = 0.5, 0.7
    qs = QSystem()
    q = [Qubit(qs) for _ in range(2)]
    H = sp.sin(2 * x) * q[0].Z * q[1].Z + sp.sin(2 * x) * q[0].X \
        + sp.sin(2 * x) * q[1].X

    prov = diffQCProvider()
    qs_c = QSystem(); _ = [Qubit(qs_c) for _ in range(2)]
    qs_c.add_evolution(H.set_parameterizedHam({"x": x_val}), T)
    prov.compile(qs_c, "quera", "Aquila", "rydberg2d", tol=0.1, verbose=0)
    n, sol_gvars, boxes, _e, _ = prov.prog
    mapper = TweezerMapper(n_qubits=n, sol_gvars=sol_gvars, boxes=boxes,
                           ramp_time=0.01)

    np.random.seed(1)
    programs = observable_program_generator(
        H, T, n_sample=1, n_repetition=1, diff_var="x", value=x_val)
    H_list = programs[0][0][0]
    return mapper, H_list, n, T


def test_mapper_threads_start_positions_into_aod_ops():
    mapper, H_list, n, T = _compile_2q()
    ops, _, _ = mapper.map_hlist(H_list, T=T)
    aods = [op for op in ops if op["op"] == "aod"]
    assert aods, "2q ZZ branch must contain transport moves"
    for op in aods:
        assert op["positions_from"] is not None
        assert len(op["positions_from"]) == n
        assert op["positions_from"] != op["positions"], \
            "emitted moves must actually change positions"


def test_end_to_end_2q_waveforms():
    mapper, H_list, n, T = _compile_2q()
    logical, _, _ = mapper.map_hlist_tree(H_list, T=T)
    physical = pc.to_physical(logical, n)

    # fresh PulseDSL session (global singletons)
    import PulseDSL_py.schedule as dsl_schedule
    from PulseDSL_py import Channels, Schedule, PulseLib
    from PulseDSL_py.pulselib import set_platform
    from simuq.braket.diffQC_provider import to_pulsedsl_tree

    dsl_schedule.sched = None
    ch, _reg = Channels(pc.NUM_PHYSICAL_CHANNELS)
    sched = Schedule()
    set_platform(PulseLib.Rydberg)
    to_pulsedsl_tree(physical, ch, ch[pc.TRANSPORT_AOD_X], run=True)

    t, waves = compile_waveforms(sched,
                                 n_channels=pc.NUM_PHYSICAL_CHANNELS)
    assert set(waves) == set(range(pc.NUM_PHYSICAL_CHANNELS))
    assert len(t) > 0 and all(len(w) == len(t) for w in waves.values())

    active = {i for i, w in waves.items() if np.abs(w).max() > 1e-12}
    # addressing combs, dressing, gate, and both transport axes all fire
    assert {pc.ADDR_DET, pc.ADDR_RABI, pc.DRESSING_AOM, pc.GATE_AOM,
            pc.TRANSPORT_AOD_X, pc.TRANSPORT_AOD_Y} <= active

    # sample span covers the whole schedule (last entry's end)
    t_end = max(float(e._ScheduleEntry__t1)
                for row in sched._Sched__schedule[:pc.NUM_PHYSICAL_CHANNELS]
                for e in row)
    assert len(t) == int(np.ceil(t_end))

    # the transport X channel really chirps: instantaneous frequency moves
    row = sched._Sched__schedule[pc.TRANSPORT_AOD_X]
    chirps = [e for e in row
              if isinstance(e._ScheduleEntry__pulse.waveform, ChirpTone)]
    assert chirps, "2q ZZ branch must produce at least one transport chirp"
    wf = chirps[0]._ScheduleEntry__pulse.waveform
    assert not np.isclose(wf.f0_mhz, wf.f1_mhz)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} awg_compile tests passed.")
