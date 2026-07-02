"""
test_cz_kick.py — the ZZ PSR kick compiles to a digital CZ + virtual Z's.

Covers:
1. The algebraic identity behind cz_kick_decomposition:
   exp(-i·φ·Z⊗Z) = e^{iφ}·(e^{-iφZ}⊗e^{-iφZ})·CP(-4φ), exact for any φ.
2. Both PSR branch angles (π/4 and 7π/4) reduce to the SAME native CZ
   (theta_cp = π), differing only in the virtual-Z phases.
3. The mapper emits a cz_gate op/ledger entry (short fixed duration,
   branch-symmetric) instead of an analog dwell, keeps the honest "kick"
   ledger entry, and verify's reconstruction still yields [Hj, kick].
"""

import os
import sys

import numpy as np
import pytest
import qutip as qp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simuq import QSystem, Qubit
from tweezer_mapper import TweezerMapper, cz_kick_decomposition
from verify_compilation import _ledger_to_H_list


# ── helpers ───────────────────────────────────────────────────────────────────

def _cp_gate(theta):
    """CP(theta) = diag(1, 1, 1, e^{i·theta})."""
    return qp.Qobj(np.diag([1.0, 1.0, 1.0, np.exp(1j * theta)]),
                   dims=[[2, 2], [2, 2]])


def _implemented_unitary(phi):
    """Rebuild the compiled kick unitary from cz_kick_decomposition."""
    theta_cp, vz, gphase = cz_kick_decomposition(phi)
    vz_1q = (-1j * vz * qp.sigmaz()).expm()
    return np.exp(1j * gphase) * qp.tensor(vz_1q, vz_1q) * _cp_gate(theta_cp)


def _target_unitary(phi):
    ZZ = qp.tensor(qp.sigmaz(), qp.sigmaz())
    return (-1j * phi * ZZ).expm()


def _mapper_2q(**kw):
    # no boxes needed for kick-only mapping
    return TweezerMapper(n_qubits=2, sol_gvars=[6.0, 0.0], boxes=[], **kw)


def _zz_hlist(kick):
    qs = QSystem()
    q = [Qubit(qs) for _ in range(2)]
    H_eval = 1.0 * q[0].X + 1.0 * q[1].X
    Hj = 1.0 * (q[0].Z * q[1].Z)
    return [[H_eval, 0.3], [Hj, kick], [H_eval, 0.7]], Hj


# ── 1. the identity ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("phi", [np.pi / 4, 7 * np.pi / 4, -np.pi / 4,
                                 0.3, -1.234, 2.5])
def test_decomposition_exact(phi):
    diff = (_implemented_unitary(phi) - _target_unitary(phi)).norm()
    assert diff < 1e-12


def test_psr_branches_share_native_cz():
    # s=-1 branch: kick angle π/4;  s=+1 branch: 7π/4 (≡ -π/4)
    th_m, vz_m, _ = cz_kick_decomposition(np.pi / 4)
    th_p, vz_p, _ = cz_kick_decomposition(7 * np.pi / 4)
    assert th_m == pytest.approx(np.pi)       # native CZ
    assert th_p == pytest.approx(np.pi)       # SAME native CZ
    assert vz_m != pytest.approx(vz_p)        # branches differ only virtually
    # and the two branch unitaries are inverses (up to global phase they
    # implement exp(∓iπ/4·ZZ))
    U = _implemented_unitary(np.pi / 4) * _implemented_unitary(7 * np.pi / 4)
    I4 = qp.tensor(qp.qeye(2), qp.qeye(2))
    phase = U[0, 0] / abs(U[0, 0])
    assert (U / phase - I4).norm() < 1e-12


# ── 2. mapper emission ────────────────────────────────────────────────────────

@pytest.mark.parametrize("kick", [np.pi / 4, 7 * np.pi / 4])
def test_mapper_emits_cz_gate_op(kick):
    m = _mapper_2q()
    H_list, _ = _zz_hlist(kick)
    ops, _, ledger = m.map_hlist(H_list)

    zz_ch = 2 * m.n + 1
    gate_plays = [o for o in ops
                  if o["op"] == "play" and o["channel"] == zz_ch]
    assert len(gate_plays) == 1
    g = gate_plays[0]
    assert g["duration"] == pytest.approx(m.cz_gate_time)   # short, fixed
    assert g["amplitude"] == pytest.approx(np.pi)           # native CZ angle

    kinds = [e.channel_kind for e in ledger.play_entries()]
    assert "cz_gate" in kinds
    assert "kick" in kinds                                   # honest entry kept
    cz = [e for e in ledger.play_entries() if e.channel_kind == "cz_gate"][0]
    assert cz.target_qubits == [0, 1]
    assert cz.duration == pytest.approx(m.cz_gate_time)


def test_branches_have_equal_gate_duration():
    """The 7π/4 branch is no longer ~7x longer in the gate zone."""
    durs = []
    for kick in (np.pi / 4, 7 * np.pi / 4):
        m = _mapper_2q()
        H_list, _ = _zz_hlist(kick)
        ops, _, _ = m.map_hlist(H_list)
        zz_ch = 2 * m.n + 1
        durs.append([o["duration"] for o in ops
                     if o["op"] == "play" and o["channel"] == zz_ch][0])
    assert durs[0] == pytest.approx(durs[1])


def test_gate_zone_at_blockade_distance():
    m = _mapper_2q()
    H_list, _ = _zz_hlist(np.pi / 4)
    m.map_hlist(H_list)
    assert len(m.log.cz_moves) == 1
    assert m.log.cz_moves[0].R_target == pytest.approx(m.R_cz)


# ── 3. verify reconstruction unchanged ────────────────────────────────────────

def test_ledger_reconstruction_yields_kick_segment():
    m = _mapper_2q()
    kick = 7 * np.pi / 4
    H_list, Hj = _zz_hlist(kick)
    _, _, ledger = m.map_hlist(H_list)

    recon = _ledger_to_H_list(ledger, Hj.sites_type, Hj.sites_name)
    # segment 1 must be [Hj, kick] — cz_gate hardware entries not double-counted
    durs = [d for _, d in recon]
    assert any(abs(d - kick) < 1e-12 for d in durs)
    k = durs.index(kick)
    H_r = recon[k][0].to_qutip_qobj()
    H_t = Hj.to_qutip_qobj()
    assert (H_r - H_t).norm() < 1e-12
    assert not any(abs(d - m.cz_gate_time) < 1e-12 for d in durs)
