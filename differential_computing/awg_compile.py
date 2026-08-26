"""
awg_compile.py — AWG waveform synthesis from a PulseDSL schedule.

The missing last mile of the pipeline: everything upstream (mapper → physical
channels → to_pulsedsl_tree → RUN) produces a *symbolic* per-channel schedule
of timed entries; this module turns that schedule into actual complex baseband
sample arrays, one per physical channel — what an AWG would emit.

Division of labor (user-set design):
  - PulseDSL's scheduler is reused ONLY for timing — SEQ/PARA semantics place
    every entry at its [t0, t1) ns window on its channel.
  - The waveform CONTENT lives here: `to_pulsedsl_tree` attaches a callable
    from this module to each Pulse's `waveform` field (PulseDSL carries it
    through untouched), and `compile_waveforms` evaluates + sums them.
  - Summing co-temporal same-channel entries is what makes COMB physically
    real: a multi-tone comb becomes an actual multi-frequency sample array,
    and a transport chirp becomes a moving tweezer.

Units
-----
Schedule time is ns (integers — PulseDSL convention, μs × 1000 upstream).
Tone frequencies are MHz, so phase(t) = 2π · f_MHz · 1e-3 · t_ns.
Waveform callables take t in ns RELATIVE TO THE ENTRY START and return complex
baseband amplitude; they accept numpy arrays (vectorized).

This module has no PulseDSL dependency: `compile_waveforms` reads the Sched
entries generically (the same fields `Sched.view()` reads).
"""

import numpy as np

MHZ_NS = 1e-3   # cycles per ns for a 1 MHz tone


class ConstantTone:
    """Constant-envelope tone: A · exp(i(2π f t + φ)), t in ns, f in MHz.

    f = 0 gives a plain constant envelope A·e^{iφ} — the faithful sampled form
    of the solver's piecewise-constant pulses (detuning/Rabi/dressing/ZZ plays).
    """

    def __init__(self, amplitude, phase=0.0, freq_mhz=0.0):
        self.amplitude = float(amplitude)
        self.phase = float(phase)
        self.freq_mhz = float(freq_mhz)

    def __call__(self, t_ns):
        t = np.asarray(t_ns, dtype=float)
        return self.amplitude * np.exp(
            1j * (2.0 * np.pi * self.freq_mhz * MHZ_NS * t + self.phase))

    def __repr__(self):
        return (f"ConstantTone(A={self.amplitude:.4g}, "
                f"f={self.freq_mhz:.4g} MHz, phi={self.phase:.4g})")


class ChirpTone:
    """Phase-continuous minimum-jerk frequency chirp f0 → f1 over duration_ns.

    The AOD deflection angle (∝ RF frequency) is the tweezer position, so the
    chirp profile IS the transport trajectory. The minimum-jerk trajectory
    [Cicali et al., Phys. Rev. Applied 24, 024070 (2025), Eq. (6)]

        x(t) = d·(10 s³ − 15 s⁴ + 6 s⁵),   s = t/T,

    has zero velocity AND zero acceleration at both endpoints (it minimizes
    ∫|jerk|² over the path), avoiding the transport heating of a constant
    sweep rate's velocity discontinuities. In frequency terms:

        f(t)  = f0 + Δf·(10 s³ − 15 s⁴ + 6 s⁵),          Δf = f1 − f0,
        φ(t)  = 2π·1e-3·(f0·t + Δf·T·(2.5 s⁴ − 3 s⁵ + s⁶)) + φ0,

    the phase being the exact integral of f — analytic and phase-continuous,
    with the same total accrued phase as the linear chirp (mean frequency is
    still (f0+f1)/2 by the profile's time-reversal symmetry).

    profile="linear" keeps the old constant-sweep-rate chirp (the paper's
    Eq. (4)) for comparison.
    """

    def __init__(self, amplitude, f0_mhz, f1_mhz, duration_ns, phase=0.0,
                 profile="minjerk"):
        if duration_ns <= 0:
            raise ValueError("ChirpTone needs duration_ns > 0")
        if profile not in ("minjerk", "linear"):
            raise ValueError(f"unknown chirp profile: {profile!r}")
        self.amplitude = float(amplitude)
        self.f0_mhz = float(f0_mhz)
        self.f1_mhz = float(f1_mhz)
        self.duration_ns = float(duration_ns)
        self.phase = float(phase)
        self.profile = profile

    def instantaneous_freq_mhz(self, t_ns):
        s = np.asarray(t_ns, dtype=float) / self.duration_ns
        df = self.f1_mhz - self.f0_mhz
        if self.profile == "linear":
            return self.f0_mhz + df * s
        return self.f0_mhz + df * (10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5)

    def __call__(self, t_ns):
        t = np.asarray(t_ns, dtype=float)
        s = t / self.duration_ns
        df = self.f1_mhz - self.f0_mhz
        if self.profile == "linear":
            ramp_int = 0.5 * df * t * s
        else:
            ramp_int = df * self.duration_ns * (2.5 * s**4 - 3.0 * s**5 + s**6)
        phi = 2.0 * np.pi * MHZ_NS * (self.f0_mhz * t + ramp_int)
        return self.amplitude * np.exp(1j * (phi + self.phase))

    def __repr__(self):
        return (f"ChirpTone(A={self.amplitude:.4g}, "
                f"{self.f0_mhz:.4g}->{self.f1_mhz:.4g} MHz, "
                f"T={self.duration_ns:.4g} ns, {self.profile})")


class SampledTone:
    """Measured pulse shape on a carrier: A(t) · exp(i(2π f_c t + φ(t) + φ₀)).

    A(t) and φ(t) are given as sample tables (linearly interpolated between
    points); the real hardware drive is the real part, A(t)·cos(2πf_c t + φ(t)).
    Outside the table's time support the tone is 0 — the shape defines its own
    duration.  `scale` multiplies the (typically normalized Ω/Ω₀) amplitude
    table by the calibration amplitude Ω₀.
    """

    def __init__(self, t_ns, amp, phase_rad, carrier_mhz=0.0, scale=1.0,
                 phase_offset=0.0, label="sampled"):
        self.t_ns = np.asarray(t_ns, dtype=float)
        self.amp = np.asarray(amp, dtype=float)
        self.phase_rad = np.asarray(phase_rad, dtype=float)
        if not (len(self.t_ns) == len(self.amp) == len(self.phase_rad)):
            raise ValueError("SampledTone: t/amp/phase lengths differ")
        if len(self.t_ns) < 2:
            raise ValueError("SampledTone needs at least 2 samples")
        self.carrier_mhz = float(carrier_mhz)
        self.scale = float(scale)
        self.phase_offset = float(phase_offset)
        self.label = label

    @property
    def duration_ns(self):
        return float(self.t_ns[-1] - self.t_ns[0])

    def __call__(self, t_ns):
        t = np.asarray(t_ns, dtype=float) + self.t_ns[0]
        a = np.interp(t, self.t_ns, self.amp, left=0.0, right=0.0)
        p = np.interp(t, self.t_ns, self.phase_rad)
        return self.scale * a * np.exp(
            1j * (2.0 * np.pi * self.carrier_mhz * MHZ_NS * t
                  + p + self.phase_offset))

    def __repr__(self):
        return (f"SampledTone({self.label}, {len(self.t_ns)} pts, "
                f"T={self.duration_ns:.4g} ns, "
                f"fc={self.carrier_mhz:.4g} MHz, scale={self.scale:.4g})")


class GateShape:
    """A fixed, calibrated gate pulse shape shared by every gate of one kind.

    Wraps the measured (t, Ω/Ω₀, φ) table + carrier so the emission layer can
    stamp out one SampledTone per gate instance, differing only in the
    per-gate phase offset (e.g. the CZ kick's virtual-Z angle) — the shape,
    duration, and carrier are identical for all instances by construction.
    """

    def __init__(self, t_ns, amp, phase_rad, carrier_mhz, omega0=1.0,
                 label="gate"):
        self.t_ns = np.asarray(t_ns, dtype=float)
        self.amp = np.asarray(amp, dtype=float)
        self.phase_rad = np.asarray(phase_rad, dtype=float)
        self.carrier_mhz = float(carrier_mhz)
        self.omega0 = float(omega0)
        self.label = label

    @classmethod
    def from_csv(cls, path, carrier_mhz, omega0=1.0, label=None):
        """Load a (t_ns, Omega_over_Omega0, phi_rad) CSV with header row."""
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        return cls(data[:, 0], data[:, 1], data[:, 2], carrier_mhz,
                   omega0=omega0,
                   label=label or path.rsplit("/", 1)[-1])

    @property
    def duration_ns(self):
        return float(self.t_ns[-1] - self.t_ns[0])

    def tone(self, phase_offset=0.0):
        return SampledTone(self.t_ns, self.amp, self.phase_rad,
                           carrier_mhz=self.carrier_mhz, scale=self.omega0,
                           phase_offset=phase_offset, label=self.label)


def tone_waveform(tone, duration_ns):
    """Build the waveform callable for one pulse_tree.Tone.

    Constant tone unless the Tone declares a chirp target (frequency_end).
    """
    f_end = getattr(tone, "frequency_end", None)
    if f_end is None or float(f_end) == float(tone.frequency):
        return ConstantTone(tone.amplitude, tone.phase, tone.frequency)
    return ChirpTone(tone.amplitude, tone.frequency, f_end, duration_ns,
                     phase=tone.phase)


def _fallback_waveform(pulse):
    """Constant-envelope fallback for entries without an attached waveform.

    Covers PulseDSL-internal pulses (Delay's amplitude-0 Constant) and legacy
    placeholder plays: amplitude/phase/frequency are read off the Pulse fields
    and treated as a constant tone. v1 scope — every shape without an explicit
    waveform samples as its constant envelope.
    """
    return ConstantTone(
        float(getattr(pulse, "amplitude", 0.0) or 0.0),
        float(getattr(pulse, "phase", 0.0) or 0.0),
        float(getattr(pulse, "frequency", 0.0) or 0.0),
    )


def compile_waveforms(sched, n_channels=None, dt_ns=1.0):
    """Compile a PulseDSL Sched into per-channel complex sample arrays.

    Walks every ScheduleEntry on every channel row, evaluates its waveform on
    its [t0, t1) window (t relative to entry start), and SUMS co-temporal
    entries — realizing COMB tone superposition on shared modulators.

    Parameters
    ----------
    sched      : PulseDSL Sched (the object Schedule() returns after RUN)
    n_channels : int or None — number of channel rows to compile (decoder rows
                 excluded).  None compiles every row that has entries.
    dt_ns      : float — sample period in ns (default 1.0 = 1 GS/s)

    Returns
    -------
    t_ns  : float ndarray — the common time grid [0, t_end)
    waves : dict {channel index: complex ndarray} — one waveform per channel
            (all-zero rows included so callers see silent channels explicitly)
    """
    rows = sched._Sched__schedule
    if n_channels is None:
        n_channels = len(rows)
    rows = rows[:n_channels]

    t_end = 0.0
    for row in rows:
        for e in row:
            t_end = max(t_end, float(e._ScheduleEntry__t1))

    n_samp = int(np.ceil(t_end / dt_ns)) if t_end > 0 else 0
    t_ns = np.arange(n_samp, dtype=float) * dt_ns

    waves = {}
    for ch_idx, row in enumerate(rows):
        w = np.zeros(n_samp, dtype=complex)
        for e in row:
            pulse = e._ScheduleEntry__pulse
            t0 = float(e._ScheduleEntry__t0)
            t1 = float(e._ScheduleEntry__t1)
            i0 = int(np.ceil(t0 / dt_ns))
            i1 = min(int(np.ceil(t1 / dt_ns)), n_samp)
            if i1 <= i0:
                continue
            fn = pulse.waveform if pulse.waveform is not None \
                else _fallback_waveform(pulse)
            w[i0:i1] += fn(t_ns[i0:i1] - t0)
        waves[ch_idx] = w
    return t_ns, waves


def waveform_summary(t_ns, waves, names=None):
    """Human-readable per-channel summary (samples, active time, peak |A|)."""
    lines = []
    dt = float(t_ns[1] - t_ns[0]) if len(t_ns) > 1 else 1.0
    for ch_idx in sorted(waves):
        w = waves[ch_idx]
        active = np.abs(w) > 1e-12
        name = (names or {}).get(ch_idx, f"ch{ch_idx}")
        lines.append(
            f"{name:16s}: {len(w)} samples, "
            f"{active.sum() * dt:.0f} ns active, "
            f"peak |A| = {np.abs(w).max() if len(w) else 0.0:.4f}")
    return "\n".join(lines)
