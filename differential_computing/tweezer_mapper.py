"""
tweezer_mapper.py — maps DiffSimuQ H_list branches to hardware schedule ops.

Pipeline
--------
observable_program_generator → H_list branches
    TweezerMapper.map_hlist(H_list) → (schedule_ops, TransportLog)
    diffQC_provider.to_pulsedsl(schedule_ops, channels, aod_ch) → PulseDSL

H_list structure (from observable_program_generator, one branch):
    [[evaluated_H, tau], [Hj, kick_duration], [evaluated_H, T - tau]]

    index 0, 2 : evaluated_H — the full parameterised Hamiltonian at the
                 operating point; compiled against pre-solved machine state
    index 1     : Hj kick — a single-operator TIHamiltonian (coef = 1)

Transport priority (per segment)
---------------------------------
1. Dressing (global)  — AOD moves all atoms to the solved geometry at once
2. ZZ / CZ (pairwise) — AOD brings a specific pair to R_target = (C6/J)^(1/6)
   Only used when the box contains no dressing instruction.
3. Native only (detuning / Rabi) — no AOD transport

rydberg2d.py signal-line layout (n qubits)
-------------------------------------------
lines 0 .. n-1      : Detuning of site i   (native)  ins_lvars = [d]
lines n .. 2n-1     : Rabi of site i       (native)  ins_lvars = [o, p]
line  2n            : dressing gloabl potential (native)  ins_lvars = [o]
lines 2n+1 ..       : c{q0}{q1}_zz        (derived) ins_lvars = [o]
                      ordered as link = [(i,j) for i<j]

Channel allocation in PulseDSL (n=3 example)
---------------------------------------------
ch[0..n-1]   detuning per site
ch[n..2n-1]  Rabi per site
ch[2n]       dressing (global) — not a transport channel, used as hold
ch[2n+1]     AOD transport     — single channel, sine 100 MHz
"""

import sys
import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from aod_channel import make_aod_pulse

C_6 = 862690 * 2.0 * np.pi   # Rydberg C6 coefficient (μm^6 · rad/μs)


# ── Transport log ─────────────────────────────────────────────────────────────

@dataclass
class DressingMove:
    """One global dressing transport event."""
    t_start:   float                        # time cursor when move begins (same units as T)
    positions: List[Tuple[float, float]]    # target positions for all atoms (μm)
    o_coef:    float                        # dressing amplitude solved by generate_as


@dataclass
class CZMove:
    """One pairwise CZ transport event."""
    t_start:  float
    pair:     Tuple[int, int]   # (q0, q1)
    R_target: float             # target inter-atom distance (μm)
    J:        float             # ZZ coupling coefficient


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
                f"  t={m.t_start:.4f}  DRESS  o={m.o_coef:.4f}  positions=[{pos_str}]"
            )
        for m in self.cz_moves:
            lines.append(
                f"  t={m.t_start:.4f}  CZ     pair={m.pair}  "
                f"R={m.R_target:.4f}μm  J={m.J:.4f}"
            )
        return "\n".join(lines)


# ── Schedule operation descriptors ────────────────────────────────────────────
# Plain dicts — no live PulseDSL dependency — so tests can run without MMIO.
# diffQC_provider.to_pulsedsl() translates these into actual Play/Delay calls.

def _op_aod(positions, ramp_time):
    """AOD transport: move atoms to `positions` over `ramp_time` (same units as T)."""
    return {"op": "aod", **make_aod_pulse(positions, ramp_time)}


def _op_play(channel_idx, amplitude, duration, phase=0.0):
    """Laser/MW pulse: constant-amplitude pulse on a native channel."""
    return {
        "op":         "play",
        "channel":    int(channel_idx),
        "amplitude":  float(amplitude),
        "phase":      float(phase),
        "duration":   float(duration),
    }


def _op_delay(duration):
    """Hold: no pulse emitted, time passes (used during AOD hold)."""
    return {"op": "delay", "duration": float(duration)}


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


def classify_kick(Hj):
    """
    Classify a single-operator kick TIHamiltonian for transport routing.

    Hj is always constructed with exactly one product term (coef = 1):
        Hj.ham = [(productHamiltonian, 1)]

    productHamiltonian.d maps site_name (int) → operator string.
    Only non-identity sites (op not in {'', 'I'}) are considered.

    Returns one of:
        ('detuning', site:int)   — single-site Z
        ('rabi',     site:int)   — single-site X or Y
        ('zz',       q0, q1)     — two-site ZZ
        ('unknown',)
    """
    prod, _ = Hj.ham[0]
    active = {site: op for site, op in prod.d.items() if op not in ('', 'I')}
    sites = sorted(active.keys())

    if len(active) == 1:
        site, op = sites[0], active[sites[0]]
        if op == 'Z':
            return ('detuning', site)
        if op in ('X', 'Y'):
            return ('rabi', site)

    if len(active) == 2:
        if all(active[s] == 'Z' for s in sites):
            return ('zz', sites[0], sites[1])

    return ('unknown',)


# ── TweezerMapper ─────────────────────────────────────────────────────────────

class TweezerMapper:
    """
    Maps DiffSimuQ H_list branches to hardware schedule operation lists.

    Parameters
    ----------
    n_qubits     : int   — number of qubits / atoms
    sol_gvars    : list  — solved global variables from generate_as().
                          Layout: [x1, y1, x2, y2, ...] for atoms 1..n-1.
                          Atom 0 is fixed at origin (0, 0).
    boxes        : list  — solved instruction boxes from generate_as()
                          (with_sys_ham path: includes ins objects).
                          Each box: ([((li, ji), ins, h_eval, ins_lvars), ...],
                                     duration)
    ramp_time    : float — AOD ramp + settle time (same units as T / tau).
                          Default 0.01 (1% of a typical T=1 experiment).
    """

    def __init__(self, n_qubits, sol_gvars, boxes, ramp_time=0.01):
        self.n         = int(n_qubits)
        self.sol_gvars = list(sol_gvars)
        self.boxes     = boxes
        self.ramp_time = float(ramp_time)
        self.log       = TransportLog()

    # ── Position helpers ──────────────────────────────────────────────────────

    def rest_positions(self):
        """
        Return atom positions at rest.

        For dressing, the solver places atoms at the optimal geometry and
        stores those coordinates in sol_gvars.  These are both the rest
        positions and the dressing-active positions.

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
        Compute new atom positions with pair (q0, q1) at distance R_target.

        The pair moves symmetrically about their current midpoint along the
        line joining them.  All other atoms remain at rest.

        Returns list of (x, y) tuples, length n.
        """
        pos = list(self.rest_positions())
        p0 = np.array(pos[q0], dtype=float)
        p1 = np.array(pos[q1], dtype=float)
        midpoint  = (p0 + p1) / 2.0
        direction = p1 - p0
        d = np.linalg.norm(direction)
        direction = direction / d if d > 1e-12 else np.array([1.0, 0.0])
        pos[q0] = tuple(midpoint - direction * R_target / 2.0)
        pos[q1] = tuple(midpoint + direction * R_target / 2.0)
        return pos

    # ── Low-level schedule builders ───────────────────────────────────────────

    def _dressing_ops(self, o_coef, duration, t_cursor):
        """
        Schedule ops for a dressing segment.

        The solver has already placed atoms at the optimal dressing geometry
        (rest_positions == dressing positions), so the AOD pulse confirms the
        current positions rather than moving atoms.

        Emits: aod(positions) → delay(duration) → aod(positions)
        Logs the event.
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

        Target distance: R = (C6 / |J|)^(1/6)  so that C6/R^6 = |J|.
        Emits: aod(target) → delay(duration) → aod(rest)
        Logs the event.
        """
        if abs(J) < 1e-12:
            return []
        R_target   = (C_6 / abs(J)) ** (1.0 / 6.0)
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
            ch[0 .. n-1]    detuning
            ch[n .. 2n-1]   Rabi
        """
        kind = cls[0]
        if kind == 'detuning':
            return [_op_play(
                channel_idx=cls[1],
                amplitude=float(ins_lvars[0]),
                duration=duration,
            )]
        if kind == 'rabi':
            o = float(ins_lvars[0])
            p = float(ins_lvars[1]) if len(ins_lvars) > 1 else 0.0
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
        duration : float — segment duration (same units as T)
        t_cursor : float — current time position (for logging)

        Returns list of schedule op dicts.
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
                        J=float(ins_lvars[0]),
                        duration=duration,
                        t_cursor=t_cursor,
                    ))
        return ops

    def map_kick(self, Hj, kick_duration, t_cursor):
        """
        Map a single-operator kick TIHamiltonian to schedule ops.

        The kick Hamiltonian has coefficient 1; kick_duration is the
        physical hold time.  For ZZ kicks, atoms are brought to
        R = C6^(1/6) (unit-coupling distance) for the specified duration.

        Parameters
        ----------
        Hj           : TIHamiltonian with a single product term, coef = 1
        kick_duration : float — hold time (same units as T)
        t_cursor     : float — current time (for logging)

        Returns list of schedule op dicts.
        """
        cls = classify_kick(Hj)
        if cls[0] == 'detuning':
            return [_op_play(
                channel_idx=cls[1],
                amplitude=1.0,
                duration=kick_duration,
            )]
        if cls[0] == 'rabi':
            return [_op_play(
                channel_idx=self.n + cls[1],
                amplitude=1.0,
                duration=kick_duration,
            )]
        if cls[0] == 'zz':
            q0, q1 = cls[1], cls[2]
            return self._cz_ops(q0, q1, J=1.0,
                                duration=kick_duration, t_cursor=t_cursor)
        return []

    # ── Top-level entry point ─────────────────────────────────────────────────

    def map_hlist(self, H_list):
        """
        Map one H_list branch to a schedule operation list and TransportLog.

        H_list format (one branch from observable_program_generator):
            [ [evaluated_H, tau],
              [Hj,          kick_duration],
              [evaluated_H, T - tau]      ]

        Segment index 1 is always the kick; indices 0 and 2 are evaluated_H.

        Parameters
        ----------
        H_list : list of [TIHamiltonian, float] — three segments

        Returns
        -------
        ops : list of schedule operation dicts
        log : TransportLog — movement metadata for this branch
        """
        self.log = TransportLog()
        ops      = []
        t        = 0.0

        for i, (H, duration) in enumerate(H_list):
            if i % 3 == 1:
                # Kick segment (index 1 in each 3-element H_list)
                ops.extend(self.map_kick(H, duration, t))
            else:
                # evaluated_H segment (indices 0 and 2)
                ops.extend(self.map_evaluated_H(duration, t))
            t += duration

        return ops, self.log
