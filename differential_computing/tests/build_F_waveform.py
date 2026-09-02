"""
build_F_waveform.py — App F: the running example as it is actually EMITTED,
both differentiation lanes, all six physical channels, real time axis.

Companion to Fig 5 (build_branch_anatomy.py).  Fig 5 is one PSR branch on an
event-spaced axis with axis breaks and a simplified direct move; this figure is
the honest to-scale timeline of the SAME running instance, with the realistic
transit lanes, and it puts the two lanes side by side:

  PSR lane  — the kick program [evolve tau, kick, evolve T-tau].  The kick is a
              digital op in the gate zone, so the schedule must transport the
              pair 100 um and back: transport + CZ + transport dominate the
              wall clock.
  NSR lane  — the Nyquist waveform shift.  Same frozen geometry, same single
              evolution segment, no transport and no gate: only the amplitudes
              on the drive/dressing channels change.

How the NSR lane is built (this is the paper's claimed mechanism, verified
here, not asserted).  The running example's three terms share one coefficient,
    H(x) = sin(2x) * (Z0Z1 + X0 + X1),
so the shifted program is a UNIFORM rescale of the source target,
    H(x + s) = scale * H(x),    scale = sin(2(x+s)) / sin(2x).
Every machine instruction is linear in its amplitude lvar at frozen geometry
(dressing o, detunings, Rabi Omega with the phase held), so scaling those
amplitudes by `scale` realizes H(x+s) EXACTLY: the residual of the shifted
branch equals the source residual times |scale| (checked in extract(), reported
in the JSON as `residual_exact`).  No solve, no re-mapping, geometry shared.

Rejected alternatives, both measured (see the data note): a generic recompile
at x+s moves the atoms (positions [-10.49, 0.88] -> [-10.05, -3.21]) and jumps
the amplitudes by ~10x -- that is the FD lane's full recompile, not a shift; a
frozen-geometry RE-SOLVE lands in a degenerate dressing/detuning direction
(~1e5 cancelling amplitudes) because the shifted problem is ill-conditioned
without the source as a warm start.

Sign convention: for modes where scale < 0 the Rabi amplitude is kept positive
and the sign is absorbed into the drive phase (phi -> phi + pi), which is what
a real modulator does.  The headline mode n = 0 has scale > 0.

Phases: extract (runs the pipeline twice, caches figures/F_waveform_data.npz +
_meta.json) and render (reads the cache).  REBUILD=1 forces re-extraction.
Never re-run the pipeline to tweak the plot.

Run:  conda run -n qec_pg python differential_computing/tests/build_F_waveform.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/Users/syue99/research/RISC-Q/PulseDSL/src/DSL/")

import numpy as np

FIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
NPZ = os.path.join(FIG_DIR, "F_waveform_data.npz")
META = os.path.join(FIG_DIR, "F_waveform_meta.json")

DT_OVERVIEW_NS = 2.0      # overview sample period.  Must resolve the COMB BEAT,
                          # not just the segment structure: the addressing comb
                          # carries tones 10 MHz apart, so |A| beats with a
                          # ~100 ns period.  At 20 ns that beat aliases into a
                          # fake ragged envelope in the (5 us wide) NSR column.
                          # The renderer peak-holds down to display resolution.
DT_FINE_NS = 0.5          # carrier-resolved inset sample period
INSET_NS = 250.0          # drive inset width
NSR_MODE = 0              # Nyquist mode drawn: s = (n + 1/2) / (2K)

# lane colours (paper-wide strategy colours) and per-channel accent
C_PSR, C_NSR = "#0072B2", "#009E73"
C_INK, C_SEC, C_GRID, C_SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"


# ── extraction ────────────────────────────────────────────────────────────────

def _machine_H_qobj(boxes, scale=1.0, nq=2):
    """Sum of the evaluated per-instruction Hamiltonians, as a QuTiP operator."""
    import qutip as qp
    M = {"": qp.qeye(2), "X": qp.sigmax(), "Y": qp.sigmay(), "Z": qp.sigmaz()}
    tot = 0
    for box_entries, _dur in boxes:
        for _key, _ins, h_eval, _lv in box_entries:
            for prod, c in h_eval.ham:
                tot = tot + scale * complex(c) * qp.tensor(
                    [M[prod.d.get(i, "")] for i in range(nq)])
    return tot


def _strip_identity(A):
    import qutip as qp
    n = len(A.dims[0])
    return A - (A.tr() / A.shape[0]) * qp.tensor([qp.qeye(2)] * n)


def _scale_boxes(boxes, scale):
    """Source boxes with the AMPLITUDE lvar scaled (phase lvars untouched).

    Rabi instructions carry [Omega, phi]; only Omega scales.  A negative scale
    is absorbed as Omega -> |scale|*Omega, phi -> phi + pi.
    """
    out = []
    for box_entries, dur in boxes:
        ents = []
        for key, ins, h_eval, lv in box_entries:
            lv2 = [float(v) for v in lv]
            if len(lv2) >= 2:                      # (amplitude, phase)
                lv2[0] = abs(scale) * lv2[0]
                if scale < 0:
                    lv2[1] = lv2[1] + np.pi
            else:
                lv2[0] = scale * lv2[0]
            ents.append((key, ins, h_eval, lv2))
        out.append((ents, dur))
    return out


def _lane_schedule(physical, gate_shape):
    """Fresh PulseDSL session -> Schedule for one physical op-tree."""
    from PulseDSL_py import Channels, Schedule, PulseLib
    from PulseDSL_py.pulselib import set_platform
    import PulseDSL_py.schedule as dsl_schedule
    import physical_channels as pc
    from simuq.braket.diffQC_provider import to_pulsedsl_tree

    dsl_schedule.sched = None                      # global singleton
    ch, _reg = Channels(pc.NUM_PHYSICAL_CHANNELS)
    schedule = Schedule()
    set_platform(PulseLib.Rydberg)
    to_pulsedsl_tree(physical, ch, ch[pc.TRANSPORT_AOD], run=True,
                     gate_shape=gate_shape)
    return schedule


def _fine_window(sched, ch_idx, t0_ns, width_ns, dt=DT_FINE_NS):
    """Carrier-resolved excerpt of one channel over [t0, t0+width)."""
    from awg_compile import _fallback_waveform
    rows = sched._Sched__schedule
    t = np.arange(0.0, width_ns, dt)
    w = np.zeros(len(t), dtype=complex)
    for e in rows[ch_idx]:
        e0 = float(e._ScheduleEntry__t0); e1 = float(e._ScheduleEntry__t1)
        p = e._ScheduleEntry__pulse
        m = (t + t0_ns >= e0) & (t + t0_ns < e1)
        if not m.any():
            continue
        fn = p.waveform if p.waveform is not None else _fallback_waveform(p)
        w[m] += fn(t[m] + t0_ns - e0)
    return t, w


def _tone_traces(sched, ch_idx):
    """Per-tone (t_ns, position_um) traces for a transport AOD channel.

    The AOD plays constant-amplitude tones, so |A| carries no information
    (it only shows the COMB beat); the tone FREQUENCY is the atom position
    via the device calibration, and that is what the row should show.
    """
    from awg_compile import ChirpTone
    import physical_channels as pc
    rows = sched._Sched__schedule

    def pos_of(f_mhz):
        return ((np.asarray(f_mhz, dtype=float) - pc.TRANSPORT_BASE_FREQ_MHZ)
                / pc.TRANSPORT_KAPPA_MHZ_PER_UM)

    out = []
    for e in rows[ch_idx]:
        t0 = float(e._ScheduleEntry__t0); t1 = float(e._ScheduleEntry__t1)
        wf = e._ScheduleEntry__pulse.waveform
        if isinstance(wf, ChirpTone):
            tt = np.linspace(0.0, wf.duration_ns, 160)
            f = wf.instantaneous_freq_mhz(tt)
            out.append((t0 + tt, pos_of(f)))
        elif getattr(wf, "freq_mhz", 0.0):
            f = float(wf.freq_mhz)
            out.append((np.array([t0, t1]), pos_of([f, f])))
    return out


def _bounds(sched, n_ch):
    rows = sched._Sched__schedule
    bset = {0.0}
    for r in rows[:n_ch]:
        for e in r:
            bset.add(float(e._ScheduleEntry__t0))
            bset.add(float(e._ScheduleEntry__t1))
    out = []
    for b in sorted(bset):
        if not out or b - out[-1] > 1.0:
            out.append(b)
    return out


def extract():
    import sympy as sp
    from simuq import QSystem, Qubit
    from simuq.braket.diffQC_provider import diffQCProvider
    from observable_program_generator import observable_program_generator
    from tweezer_mapper import TweezerMapper
    from nyquist_shift import tangent_hamiltonian, bandwidth_K
    from awg_compile import compile_waveforms
    import physical_channels as pc
    import physical_walkthrough as pw
    import pulse_ledger
    import tweezer_mapper as tm_mod

    fifo = "/Users/syue99/research/RISC-Q/PulseDSL/src/DSL/tmp_pulse_mmio.txt"
    if os.path.exists(fifo):
        os.remove(fifo)
    pulse_ledger.GATE_ZONE = (pw.D_ZONE_UM, 0.0)
    tm_mod.GATE_ZONE = (pw.D_ZONE_UM, 0.0)

    T, x_val, TOL = pw.T_EVOLVE_US, 0.7, 0.1
    x = sp.Symbol("x")
    qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    H = sp.sin(2 * x) * q[0].Z * q[1].Z + sp.sin(2 * x) * q[0].X \
        + sp.sin(2 * x) * q[1].X

    # ── ONE compile, shared by both lanes (the whole point of the figure) ──
    prov = diffQCProvider()
    qsc = QSystem(); _ = [Qubit(qsc) for _ in range(2)]
    qsc.add_evolution(H.set_parameterizedHam({"x": x_val}), T)
    prov.compile(qsc, "quera", "Aquila", "rydberg2d", tol=TOL, verbose=0)
    n, sol_gvars, boxes, _edges, _targs = prov.prog

    # Nyquist shift for the drawn mode
    _, A = tangent_hamiltonian(H, "x", x_val)
    K = float(bandwidth_K(A, T))
    s = (NSR_MODE + 0.5) / (2 * K)
    scale = float(np.sin(2 * (x_val + s)) / np.sin(2 * x_val))

    # exactness witness: residual of the scaled branch vs the shifted target
    import qutip as qp
    PM = {"": qp.qeye(2), "X": qp.sigmax(), "Y": qp.sigmay(), "Z": qp.sigmaz()}

    def target(val):
        h = H.set_parameterizedHam({"x": float(val)})
        tot = 0
        for prod, c in h.ham:
            tot = tot + complex(c) * qp.tensor(
                [PM[prod.d.get(i, "")] for i in range(2)])
        return _strip_identity(tot)

    res_src = float(np.abs((_strip_identity(_machine_H_qobj(boxes))
                            - target(x_val)).full()).max())
    res_nsr = float(np.abs((_strip_identity(_machine_H_qobj(boxes, scale))
                            - target(x_val + s)).full()).max())
    residual_exact = bool(abs(res_nsr - res_src * abs(scale))
                          <= 1e-12 + 1e-9 * abs(res_src * scale))

    # ── PSR lane: the kick program, realistic transit lanes ──
    np.random.seed(1)
    programs = observable_program_generator(H, T, n_sample=1, n_repetition=1,
                                            diff_var="x", value=x_val)
    H_list_psr = programs[0][0][0]
    tau_us = float(H_list_psr[0][1])

    def mapper_for(bx):
        return TweezerMapper(n_qubits=n, sol_gvars=sol_gvars, boxes=bx,
                             ramp_time=pw.AOD_SETTLE_US,
                             cz_gate_time=pw.CZ_GATE_US,
                             aod_vmax=pw.V_MAX_UM_US,
                             transit_dy=pw.TRANSIT_DY_UM)

    m_psr = mapper_for(boxes)
    tree_psr, _log, ledger_psr = m_psr.map_hlist_tree(H_list_psr, T=T)
    sched_psr = _lane_schedule(pc.to_physical(tree_psr, n), pw.GATE_SHAPE)

    # ── NSR lane: one evolution segment on the SAME geometry, scaled boxes ──
    boxes_nsr = _scale_boxes(boxes, scale)
    m_nsr = mapper_for(boxes_nsr)
    H_list_nsr = [[H.set_parameterizedHam({"x": x_val + s}), T]]
    tree_nsr, _log2, ledger_nsr = m_nsr.map_hlist_tree(H_list_nsr, T=T)
    sched_nsr = _lane_schedule(pc.to_physical(tree_nsr, n), pw.GATE_SHAPE)

    # ── per-channel waveforms, both lanes, common sample period ──
    nch = pc.NUM_PHYSICAL_CHANNELS
    arrays, lanes = {}, {}
    for tag, sched in (("psr", sched_psr), ("nsr", sched_nsr)):
        t_ns, waves = compile_waveforms(sched, n_channels=nch,
                                        dt_ns=DT_OVERVIEW_NS)
        arrays[f"{tag}_t"] = t_ns
        for c in range(nch):
            arrays[f"{tag}_ch{c}"] = waves[c]
        bd = _bounds(sched, nch)
        # branch duration is a property of the SCHEDULE, not of the plotting
        # sample rate: taking t_ns[-1]+dt made the reported wall clock drift
        # when DT_OVERVIEW_NS changed.
        lanes[tag] = dict(bounds_ns=bd, t_end_ns=float(bd[-1]),
                          active=[bool(np.abs(waves[c]).max() > 1e-12)
                                  for c in range(nch)])
        # transport rows carry position, not envelope
        for axis, cid in (("x", pc.TRANSPORT_AOD_X), ("y", pc.TRANSPORT_AOD_Y)):
            tones = _tone_traces(sched, cid)
            lanes[tag][f"n_tones_{axis}"] = len(tones)
            for i, (tt, uu) in enumerate(tones):
                arrays[f"{tag}_tone{axis}{i}_t"] = tt
                arrays[f"{tag}_tone{axis}{i}_um"] = uu

    # carrier-resolved insets: the drive comb early in each lane, and the gate
    for tag, sched in (("psr", sched_psr), ("nsr", sched_nsr)):
        tt, ww = _fine_window(sched, pc.ADDR_RABI, 0.0, INSET_NS)
        arrays[f"{tag}_inset_t"] = tt
        arrays[f"{tag}_inset_w"] = ww
    gate_entries = [e for e in sched_psr._Sched__schedule[pc.GATE_AOM]]
    if gate_entries:
        g0 = float(gate_entries[0]._ScheduleEntry__t0)
        g1 = float(gate_entries[0]._ScheduleEntry__t1)
        tt, ww = _fine_window(sched_psr, pc.GATE_AOM, g0, g1 - g0)
        arrays["gate_t"] = tt
        arrays["gate_w"] = ww
        lanes["gate_window_ns"] = [g0, g1]

    # transport accounting (the figure's headline number)
    bd = lanes["psr"]["bounds_ns"]
    seg = [(bd[i], bd[i + 1]) for i in range(len(bd) - 1)]
    aod_rows = sched_psr._Sched__schedule[pc.TRANSPORT_AOD_X]
    union = sorted((float(e._ScheduleEntry__t0), float(e._ScheduleEntry__t1))
                   for e in aod_rows)
    merged = []
    for a, b in union:
        if merged and a <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    move_ns = float(sum(b - a for a, b in merged))

    meta = dict(
        n=n, T_us=T, x_val=x_val, tol=TOL, tau_us=tau_us,
        sol_gvars=[float(v) for v in sol_gvars],
        K=K, nsr_mode=NSR_MODE, s=float(s), scale=scale,
        residual_source=res_src, residual_nsr=res_nsr,
        residual_exact=residual_exact,
        coeff_source=float(np.sin(2 * x_val)),
        coeff_shifted=float(np.sin(2 * (x_val + s))),
        dt_overview_ns=DT_OVERVIEW_NS, dt_fine_ns=DT_FINE_NS,
        channels={int(c): pc.CHANNEL_NAMES[c] for c in range(nch)},
        lanes=lanes,
        psr_segments_ns=seg,
        psr_transport_ns=move_ns,
        psr_wall_ns=lanes["psr"]["t_end_ns"],
        nsr_wall_ns=lanes["nsr"]["t_end_ns"],
        gate_us=pw.CZ_GATE_US, zone_um=pw.D_ZONE_UM, v_max=pw.V_MAX_UM_US,
        transit_dy=pw.TRANSIT_DY_UM,
    )
    os.makedirs(FIG_DIR, exist_ok=True)
    np.savez_compressed(NPZ, **arrays)
    with open(META, "w") as f:
        json.dump(meta, f, indent=1)
    print(f"extracted -> {NPZ}\n            {META}")
    print(f"  K={K:.4f}  mode n={NSR_MODE}  s={s:.6f}  scale={scale:.6f}")
    print(f"  residual: source={res_src:.3e}  nsr={res_nsr:.3e}  "
          f"exact-rescale={residual_exact}")
    print(f"  wall clock: PSR={meta['psr_wall_ns']/1e3:.3f} us "
          f"(transport {move_ns/1e3:.3f} us = "
          f"{100*move_ns/meta['psr_wall_ns']:.1f}%)  "
          f"NSR={meta['nsr_wall_ns']/1e3:.3f} us")
    return meta


def load():
    with open(META) as f:
        meta = json.load(f)
    return meta, np.load(NPZ)


def main():
    if os.environ.get("REBUILD") == "1" or not (os.path.exists(NPZ)
                                                and os.path.exists(META)):
        extract()
    meta, arrays = load()
    try:
        from F_waveform_render import render      # rendering lives next door
    except ImportError:
        print("extract-only: F_waveform_render not present yet")
        return
    render(meta, arrays)


if __name__ == "__main__":
    main()
