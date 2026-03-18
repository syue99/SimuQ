"""
aod_channel.py — AOD transport pulse encoding for the tweezer mapper.

The AOD is modelled as a single PulseDSL channel carrying a sine wave at
AOD_FREQ_MHZ (100 MHz placeholder).  The amplitude field encodes the target
atom-position configuration as a scalar proxy; the real multi-tone frequency
encoding is deferred until a WaveformFunction is added later.

The public API returns plain dicts (not live PulseDSL Pulse objects) so the
mapper is fully testable without a running MMIO / PulseDSL session.
The diffQC_provider is responsible for translating these dicts into actual
Pulse objects when submitting to hardware.
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
    with frequency proportional to position.  That will be implemented as a
    WaveformFunction when multi-tone control is available.
    """
    if not positions:
        return 0.0
    return float(np.max([np.sqrt(x ** 2 + y ** 2) for x, y in positions]))


def make_aod_pulse(positions, ramp_time_ns):
    """
    Build an AOD transport pulse descriptor.

    Returns a plain dict so the mapper can be tested without PulseDSL.
    The dict is translated into a real PulseDSL Pulse by
    diffQC_provider.to_pulsedsl().

    Parameters
    ----------
    positions    : list of (x, y) tuples — target atom positions in μm
    ramp_time_ns : int — AOD ramp + settle duration in nanoseconds

    Returns
    -------
    dict with keys:
        "type"        : "aod"
        "positions"   : list of (x, y) — full position record for logging
        "amplitude"   : float — encoded scalar (max displacement, μm)
        "freq_MHz"    : float — AOD carrier frequency
        "duration_ns" : int   — ramp duration
    """
    return {
        "type":        "aod",
        "positions":   list(positions),
        "amplitude":   encode_positions(positions),
        "freq_MHz":    AOD_FREQ_MHZ,
        "duration_ns": int(ramp_time_ns),
    }
