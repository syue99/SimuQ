"""
tweezer_mapper.py — maps DiffSimuQ H_list branches to hardware schedule ops.

Unit system
-----------
All quantities use the same natural units as SimuQ / DiffSimuQ:

    Distance     : μm   (micrometres)
    Time         : μs   (microseconds)
    Ang. freq.   : rad·μs⁻¹  (= Mrad/s;  1 MHz = 2π rad·μs⁻¹)

Rydberg C6 coefficient for Rb87 (~70S₁/₂ state, QuEra Aquila)
    C_6 = 862690 × 2π  [rad·μm⁶·μs⁻¹]
    This encodes: 862690 MHz·μm⁶ (angular, × 2π → rad/μs)

    Physical checks:
      • 3-atom ring equilibrium spacing  : ≈ 6 μm        ✓
      • Rydberg blockade radius (Ω=4 MHz): ≈ 7.7 μm      ✓
      • Interaction at 6 μm             : ≈ 17 MHz       ✓

    Note: interpreting 862690 as GHz (not MHz) would give a ring spacing of
    ~19 μm — inconsistent with QuEra Aquila geometry.  MHz is correct.

Distance formula for ZZ / CZ coupling
    V(R) = C_6 / R⁶  →  R_target = (C_6 / |J|)^(1/6)
    where J is the desired ZZ coupling in rad·μs⁻¹ and R in μm.

AOD ramp_time
    Typical real AOD settle times: 10–100 μs.
    Default is 10 μs (conservative; hardware-specific — adjust as needed).

PulseDSL boundary
    All op dicts use μs for time (consistent with T and tau).
    diffQC_provider.to_pulsedsl() converts μs → ns (× 1000) when building
    PulseDSL Pulse objects.

Pipeline
--------
observable_program_generator → H_list branches
    TweezerMapper.map_hlist(H_list) → (schedule_ops, TransportLog)
    diffQC_provider.to_pulsedsl(schedule_ops, channels, aod_ch) → PulseDSL

H_list structure (from observable_program_generator, one branch):
    [[evaluated_H, tau], [Hj, kick_duration], [evaluated_H, T - tau]]

    index 0, 2 : evaluated_H — full parameterised Hamiltonian at operating point
    index 1    : Hj kick — single-operator TIHamiltonian (coef = 1)

Transport priority (per segment)
---------------------------------
1. Dressing (global)  — AOD moves all atoms to the solved geometry at once
2. ZZ / CZ (pairwise) — AOD brings a specific pair to R_target = (C6/J)^(1/6)
   Only used when the box contains no dressing instruction.
3. Native only (detuning / Rabi) — no AOD transport needed

rydberg2d.py signal-line layout (n qubits)
-------------------------------------------
lines 0 .. n-1   : Detuning of site i   (native)   ins_lvars = [d]       d in rad·μs⁻¹
lines n .. 2n-1  : Rabi of site i       (native)   ins_lvars = [o, p]    o in rad·μs⁻¹, p in rad
line  2n         : dressing gloabl potential (native)  ins_lvars = [o]    dimensionless scale factor
lines 2n+1 ..    : c{q0}{q1}_zz         (derived)  ins_lvars = [o]       o = J in rad·μs⁻¹
                   ordered as link = [(i,j) for i<j]

Channel allocation in PulseDSL (n=3 example)
---------------------------------------------
ch[0..n-1]   detuning per site
ch[n..2n-1]  Rabi per site
ch[2n]       dressing (global) — not a transport channel; used as time holder
ch[2n+1]     AOD transport — single channel, sine 100 MHz carrier (placeholder)
"""

import sys
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from aod_channel import make_aod_pulse

# Rydberg C6 for Rb87 ~70S₁/₂ (QuEra Aquila).
# Units: rad·μm⁶·μs⁻¹  (= 862690 MHz·μm⁶ × 2π)
# V(R) = C_6 / R⁶  with R in μm gives V in rad·μs⁻¹.
C_6 = 862690 * 2.0 * np.pi


# ── Transport log ─────────────────────────────────────────────────────────────

@dataclass
class DressingMove:
    """One global dressing transport event."""
    t_start:   float                        # μs — time cursor when move begins
    positions: List[Tuple[float, float]]    # μm — target positions for all atoms
    o_coef:    float                        # dimensionless — dressing scale from generate_as


@dataclass
class CZMove:
    """One pairwise CZ transport event."""
    t_start:  float                 # μs
    pair:     Tuple[int, int]       # (q0, q1) qubit indices
    R_target: float                 # μm — inter-atom distance achieving coupling J
    J:        float                 # rad·μs⁻¹ — target ZZ coupling


class TransportLog:
    """
    Accumulates every AOD movement event for a single H_list branch.

    Consumers (calibration, visualisation, hardware upload) read
    .dressing_moves and .cz_moves after map_hlist() returns.
    """

    def __init__(self):
        self.dressing_moves: List[DressingMove] = []
        self.cz_moves:       List[CZMove]       = []

    def log_dressing(self, t_start, positions, o_coef):
        self.dressing_moves.append(
            DressingMove(float(t_start), list(positions), float(o_coef))
        )

    def log_cz(self, t_start, pair, R_target, J):
        self.cz_moves.append(
            CZMove(float(t_start), tuple(pair), float(R_target), float(J))
        )

    def summary(self):
        lines = [
            f"TransportLog: {len(self.dressing_moves)} dressing moves, "
            f"{len(self.cz_moves)} CZ moves"
        ]
        for m in self.dressing_moves:
            pos_str = ", ".join(f"({x:.2f},{y:.2f})" for x, y in m.positions)
            lines.append(
                f"  t={m.t_start:.4f} μs  DRESS  o={m.o_coef:.4f}  "
                f"positions=[{pos_str}] μm"
            )
        for m in self.cz_moves:
            lines.append(
                f"  t={m.t_start:.4f} μs  CZ     pair={m.pair}  "
                f"R={m.R_target:.4f} μm  J={m.J:.4f} rad/μs"
            )
        return "\n".join(lines)


# ── Schedule operation descriptors ────────────────────────────────────────────
# Plain dicts — no live PulseDSL dependency — so tests run without MMIO.
# All time fields ("duration", "ramp_time", ...) are in μs.
# diffQC_provider.to_pulsedsl() converts μs → ns (× 1000) for PulseDSL.

def _op_aod(positions, ramp_time):
    """
    AOD transport: move atoms to `positions` over `ramp_time` μs.

    ramp_time : float in μs (same unit as T and tau throughout the pipeline).
    """
    return {"op": "aod", **make_aod_pulse(positions, ramp_time)}


def _op_play(channel_idx, amplitude, duration, phase=0.0):
    """
    Laser/MW pulse on a native channel.

    amplitude : rad·μs⁻¹  (detuning d or Rabi Ω)
    phase     : rad        (Rabi pulse phase φ)
    duration  : μs
    """
    return {
        "op":        "play",
        "channel":   int(channel_idx),
        "amplitude": float(amplitude),
        "phase":     float(phase),
        "duration":  float(duration),   # μs
    }


def _op_delay(duration):
    """Hold: no pulse emitted, time advances by `duration` μs."""
    return {"op": "delay", "duration": float(duration)}   # μs


# ── Instruction classifiers ───────────────────────────────────────────────────

def classify_instruction(ins):
    """
    Classify a QMachine instruction by its name string (set in rydberg2d.py).

    Returns one of:
        ('detuning', site:int)
        ('rabi',     site:int)
        ('dressing',)
        ('zz',       q0:int, q1:int)
        ('unknown',)
    """
    name = ins.name
    if name.startswith("Detuning of site"):
        return ('detuning', int(name.split()[-1]))
    if name.startswith("Rabi of site"):
        return ('rabi', int(name.split()[-1]))
    if "dressing" in name.lower():
        return ('dressing',)
    if name.startswith("c") and name.endswith("_zz"):
        # format: "c{q0}{q1}_zz"  (single-digit indices, n ≤ 9)
        return ('zz', int(name[1]), int(name[2]))
    return ('unknown',)


# ── TweezerMapper ─────────────────────────────────────────────────────────────

class TweezerMapper:
    """
    Maps DiffSimuQ H_list branches to hardware schedule operation lists.

    Parameters
    ----------
    n_qubits  : int   — number of qubits / atoms
    sol_gvars : list  — solved global variables from generate_as().
                        Layout: [x1, y1, x2, y2, ...] for atoms 1..n-1 in μm.
                        Atom 0 is fixed at the origin (0, 0).
    boxes     : list  — solved instruction boxes from generate_as()
                        (with_sys_ham path: includes ins objects).
                        Each entry: ([((li,ji), ins, h_eval, ins_lvars), ...], duration)
    ramp_time : float — AOD ramp + settle time in μs.
                        Default 10 μs (conservative; adjust to hardware spec).
                        Typical range: 10–100 μs for real AOD systems.
    """

    def __init__(self, n_qubits, sol_gvars, boxes, ramp_time=10.0):
        self.n         = int(n_qubits)
        self.sol_gvars = list(sol_gvars)
        self.boxes     = boxes
        self.ramp_time = float(ramp_time)   # μs
        self.log       = TransportLog()

    # ── Position helpers ──────────────────────────────────────────────────────

    def rest_positions(self):
        """
        Return atom positions at rest in μm.

        For dressing, generate_as() places atoms at the optimal geometry and
        stores those coordinates in sol_gvars.  The rest positions and the
        dressing-active positions are the same.

        Returns list of (x, y) tuples in μm, length n.
        """
        pos = [(0.0, 0.0)]
        for i in range(self.n - 1):
            pos.append((
                float(self.sol_gvars[2 * i]),
                float(self.sol_gvars[2 * i + 1]),
            ))
        return pos

    def _pair_target_positions(self, q0, q1, R_target):
        """
        Compute new atom positions with pair (q0, q1) at distance R_target μm.

        The pair moves symmetrically about their current midpoint along the
        line joining them.  All other atoms remain at rest.

        Returns list of (x, y) tuples in μm, length n.
        """
        pos = list(self.rest_positions())
        p0  = np.array(pos[q0], dtype=float)
        p1  = np.array(pos[q1], dtype=float)
        mid = (p0 + p1) / 2.0
        d   = p1 - p0
        mag = np.linalg.norm(d)
        d   = d / mag if mag > 1e-12 else np.array([1.0, 0.0])
        pos[q0] = tuple(mid - d * R_target / 2.0)
        pos[q1] = tuple(mid + d * R_target / 2.0)
        return pos

    # ── Low-level schedule builders ───────────────────────────────────────────

    def _dressing_ops(self, o_coef, duration, t_cursor):
        """
        Schedule ops for a dressing segment.

        generate_as() has already placed atoms at the optimal dressing geometry
        (rest_positions == dressing positions), so the AOD pulse confirms those
        coordinates rather than moving atoms from a different location.

        Sequence: aod(positions, ramp_time) → delay(duration) → aod(positions, ramp_time)
        Logs a DressingMove entry.

        Parameters
        ----------
        o_coef   : float — dressing scale coefficient from ins_lvars
        duration : float — hold duration in μs
        t_cursor : float — current time in μs (for logging)
        """
        pos = self.rest_positions()
        self.log.log_dressing(t_cursor, pos, o_coef)
        return [
            _op_aod(pos, self.ramp_time),
            _op_delay(duration),
            _op_aod(pos, self.ramp_time),
        ]

    def _cz_ops(self, q0, q1, J, duration, t_cursor):
        """
        Schedule ops for a ZZ / CZ segment on pair (q0, q1).

        Target distance: R = (C_6 / |J|)^(1/6)  so that C_6 / R^6 = |J|.

        R > rest_distance  →  atoms move apart  (weaker coupling, longer hold)
        R < rest_distance  →  atoms move closer (stronger coupling, shorter hold)

        Sequence: aod(target, ramp) → delay(duration) → aod(rest, ramp)
        Logs a CZMove entry.

        Parameters
        ----------
        q0, q1   : int   — qubit indices
        J        : float — desired ZZ coupling in rad·μs⁻¹
        duration : float — hold duration in μs
        t_cursor : float — current time in μs (for logging)
        """
        if abs(J) < 1e-12:
            return []
        R_target   = (C_6 / abs(J)) ** (1.0 / 6.0)   # μm
        target_pos = self._pair_target_positions(q0, q1, R_target)
        rest_pos   = self.rest_positions()
        self.log.log_cz(t_cursor, (q0, q1), R_target, J)
        return [
            _op_aod(target_pos, self.ramp_time),
            _op_delay(duration),
            _op_aod(rest_pos,   self.ramp_time),
        ]

    def _native_ops(self, cls, ins_lvars, duration):
        """
        Schedule ops for native (detuning / Rabi) instructions — no AOD.

        Channel indices follow rydberg2d.py allocation:
            ch[0 .. n-1]    detuning  (amplitude = d in rad·μs⁻¹)
            ch[n .. 2n-1]   Rabi      (amplitude = Ω in rad·μs⁻¹, phase = φ in rad)

        duration : μs
        """
        kind = cls[0]
        if kind == 'detuning':
            return [_op_play(
                channel_idx=cls[1],
                amplitude=float(ins_lvars[0]),   # d  [rad·μs⁻¹]
                duration=duration,               # μs
            )]
        if kind == 'rabi':
            o = float(ins_lvars[0])                              # Ω  [rad·μs⁻¹]
            p = float(ins_lvars[1]) if len(ins_lvars) > 1 else 0.0  # φ  [rad]
            return [_op_play(
                channel_idx=self.n + cls[1],
                amplitude=o,
                duration=duration,
                phase=p,
            )]
        return []

    # ── Segment-level mappers ─────────────────────────────────────────────────

    def map_evaluated_H(self, duration, t_cursor):
        """
        Map one evaluated_H segment using pre-solved boxes from generate_as().

        Iterates all active instructions in boxes.  Dressing takes priority:
        if any box entry has a dressing instruction, ZZ entries in the same
        box are suppressed (the global dressing already encodes ZZ-like terms).

        Parameters
        ----------
        duration : float — segment duration in μs
        t_cursor : float — current time in μs (for logging)
        """
        ops = []
        for box_entries, _box_duration in self.boxes:
            has_dressing = any(
                classify_instruction(ins)[0] == 'dressing'
                for (_, _), ins, _, _ in box_entries
            )
            for (_, _), ins, _, ins_lvars in box_entries:
                cls = classify_instruction(ins)
                if cls[0] in ('detuning', 'rabi'):
                    ops.extend(self._native_ops(cls, ins_lvars, duration))
                elif cls[0] == 'dressing':
                    ops.extend(self._dressing_ops(
                        o_coef=float(ins_lvars[0]),
                        duration=duration,
                        t_cursor=t_cursor,
                    ))
                elif cls[0] == 'zz' and not has_dressing:
                    q0, q1 = cls[1], cls[2]
                    ops.extend(self._cz_ops(
                        q0, q1,
                        J=float(ins_lvars[0]),   # rad·μs⁻¹
                        duration=duration,
                        t_cursor=t_cursor,
                    ))
        return ops

    # ── Top-level entry point ─────────────────────────────────────────────────

    def map_hlist(self, H_list):
        """
        Map one H_list branch to a schedule operation list and TransportLog.

        H_list format (one branch from observable_program_generator):
            [ [evaluated_H, tau],
              [Hj,          kick_duration],
              [evaluated_H, T - tau]      ]

        All three segments are mapped through map_evaluated_H: the kick segment
        (index 1) is a single-term TIHamiltonian whose geometry is already
        encoded in sol_gvars and boxes by generate_as — no separate treatment
        is needed.

        All durations in μs (same as T and tau throughout the pipeline).

        Returns
        -------
        ops : list of schedule operation dicts (all time fields in μs)
        log : TransportLog — movement metadata for this branch
        """
        self.log = TransportLog()
        ops      = []
        t        = 0.0   # μs

        for H, duration in H_list:
            ops.extend(self.map_evaluated_H(duration, t))
            t += duration

        return ops, self.log
