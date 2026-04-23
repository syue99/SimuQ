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

Zone architecture
-----------------
Three fixed zones for atom placement:
    Interaction : (0, 0) + sol_gvars offsets — dressing + single-qubit gates
    Gate        : centered on (1000, 1000)   — two-qubit gates (pair at R_target)
    Idle        : (-1000, -1000 + 5*i)       — parked qubits not in use

Position state machine:
    • Dressing  → all atoms to interaction zone (sol_gvars geometry)
    • ZZ/CZ     → pair to gate zone, others to idle
    • Single-qubit (detuning/Rabi) → atoms stay put; if a target qubit is at
      gate zone from a prior ZZ, return it to interaction zone first
    • No automatic return between segments; next op moves atoms as needed

Pipeline
--------
observable_program_generator → H_list branches
    TweezerMapper.map_hlist(H_list) → (schedule_ops, TransportLog, PulseLedger)
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
from pulse_ledger import PulseLedger, idle_position, GATE_ZONE
from simuq.hamiltonian import TIHamiltonian, productHamiltonian

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
        self.ledger = PulseLedger(self.n)
        # Position state — tracks where each atom currently is
        self.current_positions = list(self.interaction_positions())
        self.current_zones     = ["interaction"] * self.n

        # Sites metadata for building TIHamiltonians — extract from boxes
        self._sites_type = None
        self._sites_name = None
        for box_entries, _ in self.boxes:
            for (_, _), ins, h_eval, _ in box_entries:
                if hasattr(h_eval, 'sites_type') and h_eval.sites_type is not None:
                    self._sites_type = h_eval.sites_type
                    self._sites_name = h_eval.sites_name
                    break
            if self._sites_type is not None:
                break
        # Fallback: build default qubit sites
        if self._sites_type is None:
            self._sites_type = ["qubit"] * self.n
            self._sites_name = [f"q{i}" for i in range(self.n)]

    # ── Position helpers ──────────────────────────────────────────────────────

    def interaction_positions(self):
        """
        Return atom positions in the interaction zone in μm.

        generate_as() places atoms at the optimal geometry: atom 0 at origin,
        others at sol_gvars offsets.  These are used for dressing + single-qubit.

        Returns list of (x, y) tuples in μm, length n.
        """
        pos = [(0.0, 0.0)]
        for i in range(self.n - 1):
            pos.append((
                float(self.sol_gvars[2 * i]),
                float(self.sol_gvars[2 * i + 1]),
            ))
        return pos

    # Keep backward-compatible alias
    rest_positions = interaction_positions

    def gate_positions(self, q0, q1, R_target):
        """
        Compute positions for a two-qubit gate: pair at gate zone, others idle.

        The pair (q0, q1) is placed symmetrically around GATE_ZONE center
        separated by R_target along the x-axis.  All other atoms go to their
        idle parking spots.

        Returns list of (x, y) tuples in μm, length n.
        """
        gx, gy = GATE_ZONE
        pos = [idle_position(i) for i in range(self.n)]
        pos[q0] = (gx - R_target / 2.0, gy)
        pos[q1] = (gx + R_target / 2.0, gy)
        return pos

    # ── Low-level schedule builders ───────────────────────────────────────────

    def _update_positions(self, new_positions, new_zones):
        """Update current atom positions and zone assignments."""
        self.current_positions = list(new_positions)
        self.current_zones = list(new_zones)

    def _positions_match(self, target_positions):
        """Return True if current positions already match target (within 1e-9 μm)."""
        if len(self.current_positions) != len(target_positions):
            return False
        return all(
            abs(cx - tx) < 1e-9 and abs(cy - ty) < 1e-9
            for (cx, cy), (tx, ty) in zip(self.current_positions, target_positions)
        )

    def _build_dressing_H(self, o_coef, sites_type, sites_name):
        """
        Build the dressing TIHamiltonian from solver params.

        Mirrors rydberg2d.py:78-86 but uses concrete float positions from
        sol_gvars instead of QMachine Expression variables.

        H = o · C₆/R⁶ · Σ_{i<j} n_i · n_j
        where n_i = (I - Z_i)/2 is the Rydberg number operator.

        Returns a TIHamiltonian with float coefficients.
        """
        n = self.n
        pos = self.interaction_positions()

        # Build (I - Z_i)/2 in terms of TIHamiltonian ops
        # n_i = I/2 - Z_i/2
        H_total = TIHamiltonian.empty(sites_type, sites_name)

        for i in range(n):
            for j in range(i):
                dsqr = (pos[i][0] - pos[j][0]) ** 2 + (pos[i][1] - pos[j][1]) ** 2
                if dsqr < 1e-20:
                    continue
                Jij = float(o_coef) * C_6 / (dsqr ** 3)

                # n_i * n_j = (I-Z_i)/2 * (I-Z_j)/2
                # = (I·I - I·Z_j - Z_i·I + Z_i·Z_j) / 4
                I_H = TIHamiltonian.identity(sites_type, sites_name)
                Zi = TIHamiltonian.op(sites_type, sites_name, i, "Z")
                Zj = TIHamiltonian.op(sites_type, sites_name, j, "Z")
                ZiZj_prod = productHamiltonian()
                ZiZj_prod[i] = "Z"
                ZiZj_prod[j] = "Z"
                ZiZj = TIHamiltonian(sites_type, sites_name, [(ZiZj_prod, 1.0)])

                ninj = (I_H + (-1.0) * Zj + (-1.0) * Zi + ZiZj) * 0.25
                H_total = H_total + Jij * ninj

        return H_total

    def _build_zz_H(self, J, q0, q1, sites_type, sites_name):
        """
        Build a ZZ TIHamiltonian: H = J · Z_{q0} · Z_{q1}.

        Returns a TIHamiltonian with float coefficients.
        """
        prod = productHamiltonian()
        prod[q0] = "Z"
        prod[q1] = "Z"
        return TIHamiltonian(sites_type, sites_name, [(prod, float(J))])

    def _dressing_ops(self, o_coef, duration, t_cursor):
        """
        Schedule ops for a dressing segment.

        Moves all atoms to the interaction zone (sol_gvars geometry).
        Computes the dressing Hamiltonian from the solver's o_coef and
        sol_gvars, and stores it in the ledger for honest verification.

        Sequence: aod(interaction, ramp) → delay(duration) → (no return aod)
        Atoms stay at interaction zone after dressing — next op moves if needed.

        Parameters
        ----------
        o_coef      : float — dressing scale coefficient from ins_lvars
        duration    : float — hold duration in μs
        t_cursor    : float — current time in μs (for logging)
        """
        pos = self.interaction_positions()
        zones = ["interaction"] * self.n

        self.log.log_dressing(t_cursor, pos, o_coef)

        ops = []
        # Skip AOD move if atoms are already at the target positions
        if not self._positions_match(pos):
            ops.append(_op_aod(pos, self.ramp_time))
            self.ledger.record(pos, zones, "aod", duration=self.ramp_time)

        self._update_positions(pos, zones)

        ops.append(_op_delay(duration))
        # Ledger: dressing play entry with stored Hamiltonian
        # Build the solver's dressing Hamiltonian from o_coef + positions
        dressing_H = self._build_dressing_H(
            o_coef, self._sites_type, self._sites_name)

        self.ledger.record(pos, zones, "play",
                           channel_kind="dressing",
                           target_qubits=list(range(self.n)),
                           detuning=float(o_coef),
                           duration=duration,
                           hamiltonian=dressing_H)

        return ops

    def _cz_ops(self, q0, q1, J, duration, t_cursor):
        """
        Schedule ops for a ZZ / CZ segment on pair (q0, q1).

        Moves pair to gate zone at R_target separation, others to idle.
        Atoms stay at gate/idle after the gate — next op moves if needed.

        Parameters
        ----------
        q0, q1   : int   — qubit indices
        J        : float — desired ZZ coupling in rad·μs⁻¹
        duration : float — hold duration in μs
        t_cursor : float — current time in μs (for logging)
        """
        if abs(J) < 1e-12:
            return []
        R_target = (C_6 / abs(J)) ** (1.0 / 6.0)   # μm
        pos = self.gate_positions(q0, q1, R_target)
        zones = ["idle"] * self.n
        zones[q0] = "gate"
        zones[q1] = "gate"

        self.log.log_cz(t_cursor, (q0, q1), R_target, J)

        ops = []
        # Skip AOD move if atoms are already at the target positions
        if not self._positions_match(pos):
            ops.append(_op_aod(pos, self.ramp_time))
            self.ledger.record(pos, zones, "aod", duration=self.ramp_time)

        self._update_positions(pos, zones)

        ops.append(_op_delay(duration))
        # Build ZZ Hamiltonian from solver's J
        zz_H = self._build_zz_H(J, q0, q1, self._sites_type, self._sites_name)
        # Ledger: ZZ play entry
        self.ledger.record(pos, zones, "play",
                           channel_kind="zz",
                           target_qubits=[q0, q1],
                           amplitude=float(J),
                           duration=duration,
                           hamiltonian=zz_H)

        return ops

    def _ensure_interaction_zone(self, qubit_idx):
        """
        If a qubit is currently at the gate zone (from a prior ZZ),
        return it to its interaction-zone position.

        Returns list of schedule ops (empty if no move needed).
        """
        if self.current_zones[qubit_idx] != "gate":
            return []

        # Move all atoms back to interaction zone
        pos = self.interaction_positions()
        zones = ["interaction"] * self.n

        ops = []
        if not self._positions_match(pos):
            ops.append(_op_aod(pos, self.ramp_time))
            self.ledger.record(pos, zones, "aod", duration=self.ramp_time)

        self._update_positions(pos, zones)
        return ops

    def _native_ops(self, cls, ins_lvars, duration):
        """
        Schedule ops for native (detuning / Rabi) instructions — no AOD.

        If the target qubit is at the gate zone from a prior ZZ, emits an
        AOD move to return it to interaction zone first.

        Channel indices follow rydberg2d.py allocation:
            ch[0 .. n-1]    detuning  (amplitude = d in rad·μs⁻¹)
            ch[n .. 2n-1]   Rabi      (amplitude = Ω in rad·μs⁻¹, phase = φ in rad)

        duration : μs
        """
        kind = cls[0]
        if kind == 'detuning':
            site = cls[1]
            retrieval_ops = self._ensure_interaction_zone(site)
            amp = float(ins_lvars[0])
            ops = retrieval_ops + [_op_play(
                channel_idx=site,
                amplitude=amp,
                duration=duration,
            )]
            self.ledger.record(
                self.current_positions, self.current_zones, "play",
                channel_kind="detuning",
                target_qubits=[site],
                amplitude=amp,
                duration=duration)
            return ops
        if kind == 'rabi':
            site = cls[1]
            retrieval_ops = self._ensure_interaction_zone(site)
            o = float(ins_lvars[0])
            p = float(ins_lvars[1]) if len(ins_lvars) > 1 else 0.0
            ops = retrieval_ops + [_op_play(
                channel_idx=self.n + site,
                amplitude=o,
                duration=duration,
                phase=p,
            )]
            self.ledger.record(
                self.current_positions, self.current_zones, "play",
                channel_kind="rabi",
                target_qubits=[site],
                amplitude=o,
                phase=p,
                duration=duration)
            return ops
        return []

    # ── Segment-level mappers ─────────────────────────────────────────────────
    def map_evaluated_H(self, duration, t_cursor):
        """
        Map one segment using pre-solved boxes from generate_as().

        Iterates all active instructions in boxes.  Dressing takes priority:
        if any box entry has a dressing instruction, ZZ entries in the same
        box are suppressed (the global dressing already encodes ZZ-like terms).

        Each channel builds and stores its own Hamiltonian in the ledger:
        - Dressing: computed from solver's o_coef + sol_gvars
        - ZZ: computed from solver's J and qubit pair
        - Detuning/Rabi: meta-params stored; H reconstructed during verify

        Parameters
        ----------
        duration    : float — segment duration in μs
        t_cursor    : float — current time in μs (for logging)
        """
        ops = []
        for box_entries, _box_duration in self.boxes:
            for (_, _), ins, _, ins_lvars in box_entries:
                cls = classify_instruction(ins)
                if cls[0] in ('detuning', 'rabi'):
                    # Skip near-zero amplitude ops (solver noise)
                    amp = float(ins_lvars[0])
                    if abs(amp) < 1e-12:
                        continue
                    ops.extend(self._native_ops(cls, ins_lvars, duration))
                elif cls[0] == 'dressing':
                    ops.extend(self._dressing_ops(
                        o_coef=float(ins_lvars[0]),
                        duration=duration,
                        t_cursor=t_cursor,
                    ))
                elif cls[0] == 'zz':
                    q0, q1 = cls[1], cls[2]
                    ops.extend(self._cz_ops(
                        q0, q1,
                        J=float(ins_lvars[0]),   # rad·μs⁻¹
                        duration=duration,
                        t_cursor=t_cursor,
                    ))
        return ops

    def _map_kick_segment(self, Hj, duration, t_cursor):
        """
        Map the PSR kick segment directly from Hj's operator terms.

        Unlike evolution segments (which use the solver's boxes), the kick
        is a PSR perturbation with its own Hamiltonian. Each term in Hj.ham
        maps directly to a hardware channel:
            Z_i     → detuning on site i
            X_i     → rabi on site i (phase=0)
            Y_i     → rabi on site i (phase=π/2)
            Z_iZ_j  → ZZ gate: AOD to gate zone → delay → AOD back

        The original Hj is stored in the ledger as the Hamiltonian for this
        segment (it IS the correct H — no solver decomposition needed).
        """
        ops = []

        for mprod, mc in Hj.ham:
            coeff = float(mc) if not hasattr(mc, 'exp_eval') else float(mc.exp_eval([], []))
            if abs(coeff) < 1e-15:
                continue

            # Find non-identity sites
            active = [(site, op) for site, op in mprod.d.items() if op != ""]

            if len(active) == 0:
                # Identity — skip
                continue

            elif len(active) == 1:
                site, op = active[0]
                # Return to interaction zone if needed (single-qubit ops
                # should run at interaction zone, not gate zone)
                retrieval_ops = self._ensure_interaction_zone(site)
                if op == "Z":
                    # Detuning: H = -d*(I-Z)/2 → Z coeff = d/2
                    # We want total Z coeff = coeff, so d = 2*coeff
                    amp = 2.0 * coeff
                    ops.extend(retrieval_ops + [_op_play(
                        channel_idx=site, amplitude=amp, duration=duration)])
                    self.ledger.record(
                        self.current_positions, self.current_zones, "play",
                        channel_kind="detuning", target_qubits=[site],
                        amplitude=amp, duration=duration)
                elif op == "X":
                    # Rabi: H = Ω/2*(cosφ*X - sinφ*Y), X coeff = Ω/2
                    # We want coeff*X, so Ω = 2*coeff, phase=0
                    amp = 2.0 * coeff
                    ops.extend(retrieval_ops + [_op_play(
                        channel_idx=self.n + site, amplitude=amp,
                        duration=duration, phase=0.0)])
                    self.ledger.record(
                        self.current_positions, self.current_zones, "play",
                        channel_kind="rabi", target_qubits=[site],
                        amplitude=amp, phase=0.0, duration=duration)
                elif op == "Y":
                    # Rabi with phase=π/2: H = Ω/2*(cos(π/2)*X - sin(π/2)*Y) = -Ω/2*Y
                    # We want coeff*Y, so Ω = -2*coeff, phase=π/2
                    amp = -2.0 * coeff
                    phase = np.pi / 2.0
                    ops.extend(retrieval_ops + [_op_play(
                        channel_idx=self.n + site, amplitude=amp,
                        duration=duration, phase=phase)])
                    self.ledger.record(
                        self.current_positions, self.current_zones, "play",
                        channel_kind="rabi", target_qubits=[site],
                        amplitude=amp, phase=phase, duration=duration)

            elif len(active) == 2:
                (s0, op0), (s1, op1) = active
                if op0 == "Z" and op1 == "Z":
                    # ZZ gate: need AOD transport to gate zone and back
                    q0, q1 = s0, s1
                    J = coeff
                    ops.extend(self._cz_ops(q0, q1, J, duration, t_cursor))
                    # After CZ, return to interaction zone for subsequent segments
                    pos_int = self.interaction_positions()
                    if not self._positions_match(pos_int):
                        zones_int = ["interaction"] * self.n
                        ops.append(_op_aod(pos_int, self.ramp_time))
                        self.ledger.record(pos_int, zones_int, "aod",
                                           duration=self.ramp_time)
                        self._update_positions(pos_int, zones_int)

        # Store original Hj as the kick Hamiltonian in the ledger.
        # The per-channel entries above provide hardware ops; this entry
        # provides the exact H for verification (avoids identity-shift
        # artifacts from detuning convention).
        self.ledger.record(
            self.current_positions, self.current_zones, "play",
            channel_kind="kick",
            target_qubits=list(range(self.n)),
            duration=duration,
            hamiltonian=Hj)

        return ops

    # ── Top-level entry point ─────────────────────────────────────────────────

    def map_hlist(self, H_list, T=None):
        """
        Map one H_list branch to a schedule operation list, TransportLog,
        and PulseLedger.

        H_list format (one branch from observable_program_generator):
            [ [evaluated_H, tau],
              [Hj,          kick_duration],
              [evaluated_H, T - tau]      ]

        All three segments are mapped through map_evaluated_H.

        Note: T is accepted but currently unused — duration rescaling between
        PSR time and solver time is deferred to a future change once the
        verify architecture supports it.

        Returns
        -------
        ops    : list of schedule operation dicts (all time fields in μs)
        log    : TransportLog — movement metadata for this branch
        ledger : PulseLedger — position + pulse meta-data for every step
        """
        self.log    = TransportLog()
        self.ledger = PulseLedger(self.n)
        # Reset position state to interaction zone at start of each branch
        self.current_positions = list(self.interaction_positions())
        self.current_zones     = ["interaction"] * self.n

        ops = []
        t   = 0.0   # μs

        for seg_idx, (H, duration) in enumerate(H_list):
            if seg_idx == 1 and len(H_list) == 3:
                # Segment 1 is the PSR kick — compile Hj directly
                ops.extend(self._map_kick_segment(H, duration, t))
            else:
                # Evolution segments — use solver's boxes
                ops.extend(self.map_evaluated_H(duration, t))
            t += duration

        return ops, self.log, self.ledger
