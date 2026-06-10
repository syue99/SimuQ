"""
physical_channels.py — logical op-tree → physical AOM/AOD channel tree.

The logical tree from TweezerMapper.map_hlist_tree assigns one channel per qubit
per role (ch[site] detuning, ch[n+site] rabi, …) — the faithful Hamiltonian
decomposition.  Real neutral-atom hardware does NOT have a wire per qubit: per-
atom control is multiplexed as RF *tones* inside a few shared modulators, and
two-qubit coupling is mediated by atom *position* plus a global beam, not a wire.

This module consolidates the logical tree onto a FIXED set of physical channels
(independent of qubit count), turning per-qubit detuning/Rabi into multi-tone
COMBs on shared addressing AODs:

    ① Detuning of site i  -> tone on ADDR_DET   (addressing AOD light shift)
    ② Rabi of site i      -> tone on ADDR_RABI  (local Rabi addressing comb)
    ③ dressing (global)   -> Play on DRESSING_AOM
    ④ ZZ(q0,q1)           -> TRANSPORT_AOD move (already in the tree as an AOD
                             node) + Play on GATE_AOM

The AAIS, solver, boxes, ledger, and verify are untouched — this is purely the
logical→physical channel layer feeding the PulseDSL scheduler.

Tone frequency is the addressing knob.  The addressing-frequency assignment and
the AOD position→frequency encoding here are PLACEHOLDERS (single-frequency
proxies); the real 2-D (fx, fy) comb encoding is deferred.
"""

from pulse_tree import Seq, Para, PlayNode, CombNode, AodNode, DelayNode, Tone


# ── Fixed physical channel ids (independent of n) ─────────────────────────────
TRANSPORT_AOD = 0   # trap/transport AOD: positions atoms (COMB, position tones)
ADDR_DET      = 1   # addressing AOD: per-atom detuning (COMB, drive tones)
ADDR_RABI     = 2   # addressing AOD: per-atom Rabi      (COMB, drive tones)
DRESSING_AOM  = 3   # global dressing beam (interaction zone)
GATE_AOM      = 4   # global gate beam (gate zone)

NUM_PHYSICAL_CHANNELS = 5

CHANNEL_NAMES = {
    TRANSPORT_AOD: "TRANSPORT_AOD",
    ADDR_DET:      "ADDR_DET",
    ADDR_RABI:     "ADDR_RABI",
    DRESSING_AOM:  "DRESSING_AOM",
    GATE_AOM:      "GATE_AOM",
}

# Placeholder addressing-frequency comb (MHz): one tone slot per atom.
ADDR_BASE_FREQ_MHZ = 80.0
ADDR_FREQ_SPACING_MHZ = 10.0


def addr_frequency(atom):
    """Placeholder addressing frequency for atom `atom` (MHz).

    Real hardware deflects the addressing beam to an atom by RF frequency; here
    we just hand out evenly spaced tone slots.  The true position→frequency map
    (and the 2-D fx/fy split) is deferred.
    """
    return ADDR_BASE_FREQ_MHZ + atom * ADDR_FREQ_SPACING_MHZ


def position_to_freq(pos):
    """Placeholder AOD position→frequency proxy (MHz) for transport tones."""
    x, y = pos
    return ADDR_BASE_FREQ_MHZ + abs(x) * 0.01 + abs(y) * 0.01


def _site_of(play, n):
    """Recover the qubit index a logical detuning/Rabi PlayNode addresses."""
    if play.kind == "detuning":
        return int(play.channel)            # ch[site]
    if play.kind == "rabi":
        return int(play.channel) - n        # ch[n + site]
    return None


# ── Transform ─────────────────────────────────────────────────────────────────

def _consolidate_para(para, n):
    """Map one logical Para of PlayNodes onto physical channels.

    Detuning plays collapse to one ADDR_DET comb, Rabi plays to one ADDR_RABI
    comb; dressing → DRESSING_AOM, ZZ → GATE_AOM (single global tones).  The
    results sit in a Para (distinct physical channels run concurrently).
    """
    det_tones, rabi_tones, globals_ = [], [], []
    duration = None

    for child in para.children:
        if not isinstance(child, PlayNode):
            # Logical Paras only hold PlayNodes; pass anything else through.
            globals_.append(to_physical(child, n))
            continue
        duration = child.duration
        if child.kind == "detuning":
            site = _site_of(child, n)
            det_tones.append(Tone(atom=site, frequency=addr_frequency(site),
                                  amplitude=child.amplitude, phase=child.phase))
        elif child.kind == "rabi":
            site = _site_of(child, n)
            rabi_tones.append(Tone(atom=site, frequency=addr_frequency(site),
                                   amplitude=child.amplitude, phase=child.phase))
        elif child.kind == "dressing":
            globals_.append(PlayNode(DRESSING_AOM, child.amplitude,
                                     child.duration, child.phase,
                                     kind="dressing"))
        elif child.kind == "zz":
            globals_.append(PlayNode(GATE_AOM, child.amplitude, child.duration,
                                     child.phase, kind="zz"))
        else:
            globals_.append(child)

    out = []
    if det_tones:
        out.append(CombNode(ADDR_DET, det_tones, duration, kind="detuning"))
    if rabi_tones:
        out.append(CombNode(ADDR_RABI, rabi_tones, duration, kind="rabi"))
    out.extend(globals_)
    return Para(out)


def _transport_comb(aod):
    """Map a logical AOD move to a TRANSPORT_AOD comb (one tone per atom).

    Tone frequency encodes the target position (placeholder proxy); amplitude is
    a unit marker that the tone is present.  The 2-D fx/fy encoding is deferred.
    """
    tones = [Tone(atom=i, frequency=position_to_freq(pos), amplitude=1.0)
             for i, pos in enumerate(aod.positions)]
    return CombNode(TRANSPORT_AOD, tones, aod.ramp_time, kind="transport")


def _map_play(play, n):
    """Map a standalone logical PlayNode (rare — usually inside a Para)."""
    return _consolidate_para(Para([play]), n)


def to_physical(node, n):
    """Transform a logical op-tree (map_hlist_tree) into a physical-channel tree.

    Parameters
    ----------
    node : pulse_tree node — the logical tree (root is a Seq)
    n    : int — number of qubits (to decode logical channel → site)

    Returns
    -------
    A pulse_tree using the fixed physical channels, with per-qubit detuning/Rabi
    consolidated into COMBs.  Feed it to diffQC_provider.to_pulsedsl_tree with
    Channels(NUM_PHYSICAL_CHANNELS) and aod_ch = ch[TRANSPORT_AOD].
    """
    if isinstance(node, Seq):
        return Seq([to_physical(c, n) for c in node.children])
    if isinstance(node, Para):
        return _consolidate_para(node, n)
    if isinstance(node, AodNode):
        return _transport_comb(node)
    if isinstance(node, DelayNode):
        return DelayNode(node.duration)
    if isinstance(node, PlayNode):
        return _map_play(node, n)
    if isinstance(node, CombNode):
        return node                     # already physical
    raise TypeError(f"to_physical: unknown node {type(node).__name__}")
