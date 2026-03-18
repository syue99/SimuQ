"""
aod_channel.py — AOD transport pulse encoding for the tweezer mapper.

Unit system
-----------
This module uses the same natural unit system as the rest of SimuQ / DiffSimuQ:

    Distance : μm  (micrometres)
    Time     : μs  (microseconds)
    Frequency: MHz (megahertz); angular frequency in rad·μs⁻¹

The Rydberg C6 coefficient lives in the same system:
    C_6 = 862690 × 2π   [rad·μm⁶·μs⁻¹]   (Rb87, ~70S₁/₂ Rydberg state)

So V(R) = C_6 / R⁶  with  R in μm  gives  V in rad/μs.

AOD channel
-----------
The AOD is modelled as a single PulseDSL channel carrying a sine wave at
AOD_FREQ_MHZ (100 MHz placeholder).  The amplitude field encodes the target
atom-position configuration as a scalar proxy; the real multi-tone frequency
encoding is deferred until a WaveformFunction is added later.

The public API returns plain dicts (not live PulseDSL Pulse objects) so the
mapper is fully testable without a running MMIO / PulseDSL session.
diffQC_provider.to_pulsedsl() translates these dicts into actual Pulse objects.
Note: PulseDSL Pulse.duration is in nanoseconds; to_pulsedsl() performs the
μs → ns conversion (× 1000) when building the Pulse.
"""

import numpy as np

AOD_FREQ_MHZ = 100.0   # fixed carrier frequency (placeholder for multi-tone comb)


def encode_positions(positions):
    """
    Encode a list of atom positions as a scalar amplitude placeholder.

    Parameters
    ----------
    positions : list of (x, y) tuples in μm

    Returns
    -------
    float — maximum displacement from origin across all atoms (μm).

    Note
    ----
    This is a temporary encoding.  The real AOD uses one RF tone per atom,
    with tone frequency proportional to position.  That will be implemented
    as a WaveformFunction when multi-tone control is available.
    """
    if not positions:
        return 0.0
    return float(np.max([np.sqrt(x ** 2 + y ** 2) for x, y in positions]))


def make_aod_pulse(positions, ramp_time):
    """
    Build an AOD transport pulse descriptor.

    Returns a plain dict so the mapper can be tested without PulseDSL.
    The dict is translated into a real PulseDSL Pulse by
    diffQC_provider.to_pulsedsl(), which converts ramp_time from μs to ns.

    Parameters
    ----------
    positions  : list of (x, y) tuples — target atom positions in μm
    ramp_time  : float — AOD ramp + settle duration in μs

    Returns
    -------
    dict with keys:
        "type"      : "aod"
        "positions" : list of (x, y) — full position record for logging
        "amplitude" : float — encoded scalar proxy (max displacement in μm)
        "freq_MHz"  : float — AOD carrier frequency (MHz)
        "duration"  : float — ramp duration in μs  (to_pulsedsl converts → ns)
    """
    return {
        "type":      "aod",
        "positions": list(positions),
        "amplitude": encode_positions(positions),
        "freq_MHz":  AOD_FREQ_MHZ,
        "duration":  float(ramp_time),
    }
