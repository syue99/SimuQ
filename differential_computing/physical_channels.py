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

Tone frequency is the addressing knob.  The addressing-frequency assignment for
detuning/Rabi is still a PLACEHOLDER (evenly spaced tone slots).  Transport is
REAL infrastructure now: crossed X/Y AODs, one tone per trapped atom, with a
linear position→frequency calibration map — a tweezer move is a linear
frequency CHIRP f(start)→f(target) on each axis over the ramp time, and a
parked atom keeps a constant hold tone (the trap must keep holding it).
"""

from pulse_tree import Seq, Para, PlayNode, CombNode, AodNode, DelayNode, Tone


# ── Fixed physical channel ids (independent of n) ─────────────────────────────
TRANSPORT_AOD_X = 0  # crossed transport AOD, X axis (COMB, one tone per atom)
ADDR_DET        = 1  # addressing AOD: per-atom detuning (COMB, drive tones)
ADDR_RABI       = 2  # addressing AOD: per-atom Rabi      (COMB, drive tones)
DRESSING_AOM    = 3  # global dressing beam (interaction zone)
GATE_AOM        = 4  # global gate beam (gate zone)
TRANSPORT_AOD_Y = 5  # crossed transport AOD, Y axis (COMB, one tone per atom)

TRANSPORT_AOD = TRANSPORT_AOD_X   # backward-compatible alias

NUM_PHYSICAL_CHANNELS = 6

CHANNEL_NAMES = {
    TRANSPORT_AOD_X: "TRANSPORT_AOD_X",
    ADDR_DET:        "ADDR_DET",
    ADDR_RABI:       "ADDR_RABI",
    DRESSING_AOM:    "DRESSING_AOM",
    GATE_AOM:        "GATE_AOM",
    TRANSPORT_AOD_Y: "TRANSPORT_AOD_Y",
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


# Transport AOD calibration: linear position→frequency map per axis.
# Zone coordinates span roughly ±1000 μm (idle/gate zones), so κ = 0.05 MHz/μm
# keeps every tone inside a 100 ± 50 MHz RF band — a realistic AOD bandwidth.
TRANSPORT_BASE_FREQ_MHZ = 100.0
TRANSPORT_KAPPA_MHZ_PER_UM = 0.05


def coord_to_freq(coord_um):
    """Transport AOD calibration: axis coordinate (μm) → RF tone frequency (MHz).

    Crossed-AOD encoding: the X-axis AOD's tone at coord_to_freq(x) deflects a
    tweezer to column x; the Y-axis AOD's tone at coord_to_freq(y) selects row
    y.  Moving an atom = chirping its tone between the two frequencies.
    """
    return TRANSPORT_BASE_FREQ_MHZ + TRANSPORT_KAPPA_MHZ_PER_UM * float(coord_um)


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


def _transport_combs(aod):
    """Map a logical AOD move to crossed X/Y transport combs (one tone per atom).

    For each atom the X-axis comb carries a tone chirping coord_to_freq(x_start)
    → coord_to_freq(x_target) over the ramp time, and the Y-axis comb likewise
    for y — the frequency ramp IS the tweezer move.  Atoms that stay put get a
    constant hold tone (frequency_end=None): the trap keeps holding them while
    others move.  Without start positions (legacy AodNode), all tones are
    constant holds at the target frequencies.

    Returns a Para of the two CombNodes — the two AOD axes drive concurrently
    and the move ends when both ramps end.
    """
    starts = (aod.positions_from if aod.positions_from is not None
              else aod.positions)
    x_tones, y_tones = [], []
    for i, ((x1, y1), (x0, y0)) in enumerate(zip(aod.positions, starts)):
        fx0, fx1 = coord_to_freq(x0), coord_to_freq(x1)
        fy0, fy1 = coord_to_freq(y0), coord_to_freq(y1)
        x_tones.append(Tone(atom=i, frequency=fx0, amplitude=1.0,
                            frequency_end=(fx1 if fx1 != fx0 else None)))
        y_tones.append(Tone(atom=i, frequency=fy0, amplitude=1.0,
                            frequency_end=(fy1 if fy1 != fy0 else None)))
    return Para([
        CombNode(TRANSPORT_AOD_X, x_tones, aod.ramp_time, kind="transport"),
        CombNode(TRANSPORT_AOD_Y, y_tones, aod.ramp_time, kind="transport"),
    ])


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
        return _transport_combs(node)
    if isinstance(node, DelayNode):
        return DelayNode(node.duration)
    if isinstance(node, PlayNode):
        return _map_play(node, n)
    if isinstance(node, CombNode):
        return node                     # already physical
    raise TypeError(f"to_physical: unknown node {type(node).__name__}")
