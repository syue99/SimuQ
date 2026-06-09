"""
pulse_tree.py — DSL-agnostic op-tree IR for the tweezer → PulseDSL bridge.

Why this exists
---------------
The PulseDSL Python scheduler is now *declarative*: `Play` / `Delay` / `SEQ` /
`PARA` only build nodes, and nothing is scheduled until `RUN(tree)` walks them
(see PulseDSL_py/SCHEDULER.md).  Time comes *only* from how the tree nests —
`SEQ` threads time child-to-child, `PARA` forks it.  There is no per-channel
auto-advance any more.

DiffSimuQ's schedule is naturally a tree: within a segment, several channels
fire simultaneously (`PARA`); segments run back-to-back (`SEQ`); AOD transport
relocates atoms between them.  The TweezerMapper is the only layer that knows
whether two ops can actually run concurrently, because concurrency depends on
atom positions (an atom cannot be at the interaction zone and the gate zone at
once).  So the mapper builds this IR *natively*, and the PulseDSL bridge becomes
a dumb translator (IR → SEQ/PARA/Play/Delay → RUN).

This module has **no PulseDSL dependency** so the mapper and its tests run
without an MMIO / PulseDSL session.  `make_aod_pulse` (from aod_channel) is the
only import, mirroring how tweezer_mapper builds AOD descriptors.

Backward compatibility
-----------------------
`flatten(tree)` reproduces the *exact* flat op-dict list that
`TweezerMapper.map_hlist` produced before this IR existed, so every existing
flat-list consumer (the ledger/verify path, the 128 tests) keeps working
unchanged.  Each leaf's `to_op()` emits the same canonical dict as the original
`_op_aod` / `_op_play` / `_op_delay` helpers.

Unit system
-----------
All durations are in μs (the convention throughout DiffSimuQ).  The PulseDSL
bridge performs the μs → ns conversion when it builds real Pulse objects.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from aod_channel import make_aod_pulse


# ── Leaf nodes ────────────────────────────────────────────────────────────────

@dataclass
class PlayNode:
    """A laser/MW pulse on one native channel.

    channel   : int   — PulseDSL channel index (rydberg2d layout:
                         ch[0..n-1] detuning, ch[n..2n-1] rabi,
                         ch[2n] dressing, ch[2n+1] ZZ gate)
    amplitude : float — detuning d or Rabi Ω, in rad·μs⁻¹
    duration  : float — μs
    phase     : float — rad (Rabi phase φ; 0 for detuning/dressing/ZZ)
    kind      : str   — channel role for downstream pulse-shape selection
                        ('detuning' | 'rabi' | 'dressing' | 'zz' | 'kick').
                        Metadata only — NOT part of the flattened op dict.
    """
    channel: int
    amplitude: float
    duration: float
    phase: float = 0.0
    kind: Optional[str] = None

    def to_op(self) -> dict:
        # Mirrors tweezer_mapper._op_play exactly (kind is intentionally omitted).
        return {
            "op":        "play",
            "channel":   int(self.channel),
            "amplitude": float(self.amplitude),
            "phase":     float(self.phase),
            "duration":  float(self.duration),   # μs
        }


@dataclass
class AodNode:
    """An AOD transport move to `positions` over `ramp_time` μs.

    An AodNode is always a *position barrier*: the atoms are somewhere else
    after it, so any plays that need the new configuration must be sequenced
    (SEQ) after it, never PARA'd alongside it.
    """
    positions: List[Tuple[float, float]]
    ramp_time: float

    def to_op(self) -> dict:
        # Mirrors tweezer_mapper._op_aod exactly.
        return {"op": "aod", **make_aod_pulse(self.positions, self.ramp_time)}


@dataclass
class DelayNode:
    """A hold: no pulse emitted, time advances by `duration` μs (all channels)."""
    duration: float

    def to_op(self) -> dict:
        # Mirrors tweezer_mapper._op_delay exactly.
        return {"op": "delay", "duration": float(self.duration)}   # μs


# ── Block nodes ───────────────────────────────────────────────────────────────

@dataclass
class Seq:
    """Children run back-to-back: each starts when the previous ends."""
    children: List["object"] = field(default_factory=list)

    def add(self, node) -> "Seq":
        self.children.append(node)
        return self


@dataclass
class Para:
    """Children all start together; the block ends at the max child end.

    Only ops that share the *same atom-position state* (no AOD between them)
    and target distinct channels may live in one Para.
    """
    children: List["object"] = field(default_factory=list)

    def add(self, node) -> "Para":
        self.children.append(node)
        return self


LEAF_TYPES = (PlayNode, AodNode, DelayNode)
BLOCK_TYPES = (Seq, Para)


# ── Flattening (exact backward-compat with the old flat op list) ──────────────

def flatten(node) -> List[dict]:
    """Walk the tree left-to-right and return the canonical flat op-dict list.

    Reproduces the exact ordering that `map_hlist` produced before this IR:
    leaves are emitted in depth-first, left-to-right order.  Block nodes
    contribute no op of their own — they only structure timing.
    """
    if node is None:
        return []
    if isinstance(node, LEAF_TYPES):
        return [node.to_op()]
    if isinstance(node, BLOCK_TYPES):
        ops: List[dict] = []
        for child in node.children:
            ops.extend(flatten(child))
        return ops
    raise TypeError(f"flatten: unknown node type {type(node).__name__}")


# ── Pretty-printing (for the validation walkthrough) ──────────────────────────

def pretty(node, indent: int = 0) -> str:
    """Render the tree as indented text — handy for inspecting structure."""
    pad = "  " * indent
    if isinstance(node, PlayNode):
        tag = f"[{node.kind}]" if node.kind else ""
        return (f"{pad}Play ch{node.channel} amp={node.amplitude:.4g} "
                f"phase={node.phase:.4g} dur={node.duration:.4g} {tag}".rstrip())
    if isinstance(node, AodNode):
        return f"{pad}Aod -> {len(node.positions)} atoms, ramp={node.ramp_time:.4g}"
    if isinstance(node, DelayNode):
        return f"{pad}Delay dur={node.duration:.4g}"
    if isinstance(node, (Seq, Para)):
        head = "SEQ" if isinstance(node, Seq) else "PARA"
        lines = [f"{pad}{head}"]
        for child in node.children:
            lines.append(pretty(child, indent + 1))
        return "\n".join(lines)
    raise TypeError(f"pretty: unknown node type {type(node).__name__}")
