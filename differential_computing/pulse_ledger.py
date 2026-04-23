"""
pulse_ledger.py — Position + pulse meta-data ledger for the tweezer mapper.

Records a snapshot of all atom positions and pulse parameters at every
schedule step, enabling:
  1. AOD driving — position data tells the AOD where atoms are.
  2. QuTiP round-trip verification — meta-parameters (+ stored Hamiltonians
     for dressing) let verify_compilation.py reconstruct the physics and
     compare against the direct H_list simulation.

Unit system (same as tweezer_mapper.py / SimuQ):
    Distance : μm       Time : μs       Frequency : rad·μs⁻¹
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ── Zone constants ────────────────────────────────────────────────────────────

INTERACTION_ZONE = (0.0, 0.0)   # origin — dressing + single-qubit ops
GATE_ZONE        = (1000.0, 1000.0)  # two-qubit gate region
IDLE_ZONE_BASE   = (-1000.0, -1000.0)  # idle parking; qubit i → (-1000, -1000+5*i)


def idle_position(qubit_idx: int) -> Tuple[float, float]:
    """Return the idle-zone parking position for a given qubit index."""
    return (IDLE_ZONE_BASE[0], IDLE_ZONE_BASE[1] + 5.0 * qubit_idx)


# ── Ledger entry ──────────────────────────────────────────────────────────────

@dataclass
class LedgerEntry:
    """One snapshot in the pulse ledger.

    Every schedule step (AOD move, play pulse, or delay) produces one entry.
    Pulse meta-data fields are populated for ``op_type="play"`` and left
    ``None`` for AOD moves and delays.

    Attributes
    ----------
    step_idx       : int — sequential index within the branch
    positions      : list of (x, y) — all atom positions in μm at this step
    zone           : list of str — per-qubit zone label
    op_type        : str — "play", "aod", or "delay"
    channel_kind   : str | None — "detuning", "rabi", "dressing", "zz"
    target_qubits  : list[int] | None — qubits this pulse acts on
    amplitude      : float | None — Ω or d in rad·μs⁻¹
    phase          : float | None — φ in rad (Rabi pulses)
    detuning       : float | None — dressing scale / Z-interaction strength
    duration       : float | None — μs
    hamiltonian    : TIHamiltonian | None — stored directly for dressing segments
    """
    step_idx:      int
    positions:     List[Tuple[float, float]]
    zone:          List[str]

    op_type:       str
    channel_kind:  Optional[str]           = None
    target_qubits: Optional[List[int]]     = None
    amplitude:     Optional[float]         = None
    phase:         Optional[float]         = None
    detuning:      Optional[float]         = None
    duration:      Optional[float]         = None
    hamiltonian:   object                  = None   # TIHamiltonian (avoid import cycle)


# ── Pulse ledger ──────────────────────────────────────────────────────────────

class PulseLedger:
    """Accumulates LedgerEntry snapshots for one H_list branch."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.entries: List[LedgerEntry] = []
        self._step_counter = 0

    def record(self, positions, zone, op_type, **meta):
        """Append a new LedgerEntry.

        Parameters
        ----------
        positions : list of (x, y)
        zone      : list of str
        op_type   : "play" | "aod" | "delay"
        **meta    : pulse meta-data keys (channel_kind, target_qubits,
                    amplitude, phase, detuning, duration, hamiltonian)
        """
        entry = LedgerEntry(
            step_idx=self._step_counter,
            positions=list(positions),
            zone=list(zone),
            op_type=op_type,
            channel_kind=meta.get("channel_kind"),
            target_qubits=meta.get("target_qubits"),
            amplitude=meta.get("amplitude"),
            phase=meta.get("phase"),
            detuning=meta.get("detuning"),
            duration=meta.get("duration"),
            hamiltonian=meta.get("hamiltonian"),
        )
        self.entries.append(entry)
        self._step_counter += 1

    def play_entries(self):
        """Yield only entries with op_type='play' (pulse segments)."""
        return [e for e in self.entries if e.op_type == "play"]

    def summary(self):
        """Human-readable summary string."""
        lines = [f"PulseLedger: {len(self.entries)} entries, {self.n_qubits} qubits"]
        for e in self.entries:
            pos_str = ", ".join(f"({x:.1f},{y:.1f})" for x, y in e.positions)
            zone_str = ",".join(e.zone)
            if e.op_type == "play":
                meta = (
                    f"  kind={e.channel_kind}  qubits={e.target_qubits}  "
                    f"amp={e.amplitude}  phase={e.phase}  "
                    f"det={e.detuning}  dur={e.duration} μs"
                    f"  H={'yes' if e.hamiltonian is not None else 'no'}"
                )
            elif e.op_type == "aod":
                meta = "  (AOD move)"
            else:
                meta = f"  (delay {e.duration} μs)" if e.duration else "  (delay)"
            lines.append(
                f"  [{e.step_idx}] {e.op_type:5s}  zones=[{zone_str}]  "
                f"pos=[{pos_str}]{meta}"
            )
        return "\n".join(lines)
