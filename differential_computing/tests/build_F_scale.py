"""
build_F_scale.py — F-scale (RQ3) per FSCALE_REVISION 2026-08-19.

Claim: the INCREMENTAL cost of differentiation over the source compilation
path is modest. NOT "SimuQ scales"; the generic-vs-specialized solver story
is appendix material.

Timing scope (R5.2, measured): compile runs through machine-native schedule
ops + pulse ledger. It does NOT reach pulse-shape synthesis: the PulseDSL
emission layer has a 16-channel logical cap and its physical-channel COMB
encoding does not complete at n=100 — excluded and flagged as engineering.

Series (R1):
  source   — specialized-path compile (target -> schedule ops), seconds
  +NSR     — per-branch coefficient table (specializer.nsr_shift_table):
             the Nyquist branch shares the source segment structure, so its
             marginal compile is an O(n) closed-form table. Measured, not
             assumed flat.
  +PSR     — per-branch map_hlist (schedule ops + ledger re-emitted; the
             kick segment forces a structural re-map — branch-specific tau
             splits the evolution, kick inserts AOD+CZ). Marginal accounting:
             the evolution SOLVE is cached; the mapper walk is per-branch.

Two-regime PSR slope (R4a, profiled): below n~200 the branch is dominated by
rebuilding the dressing Hamiltonian for the ledger; above, by the ledger's
per-play position snapshots (n positions x ~4n plays -> O(n^2)). Slopes are
fitted per window; the per-pass breakdown is measured with instrumented
wrappers at every n.

Outputs: figures/F_scale_data.json (cache; delete to re-time),
figures/F_scale_strip.{png,pdf} (Fig C strip, ~1/3 page),
figures/F_scale_appendix.{png,pdf}, *_caption.txt, F_scale_data_note.md.

Run:  conda run -n qec_pg python differential_computing/tests/build_F_scale.py
"""

import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sympy as sp

from simuq import QSystem, Qubit
from simuq import specializer
from simuq.aais import rydberg2d
from simuq.braket.diffQC_provider import diffQCProvider
from observable_program_generator import observable_program_generator

T = 1.0
X_VAL = 0.8
TOL = 0.1
M_TAU = 48            # G2: tau-sample count m, identical to F6/F-loop/F-phase
N_NYQ = 8             # NSR deterministic Nyquist order N -> 2N branches total
NS_GENERIC = [2, 3, 4, 5, 6, 8, 10, 12]
NS_SPECIAL = [4, 6, 8, 10, 20, 50, 100, 200, 500, 1000]
GRIDS_SPECIAL = [(4, 4), (7, 7), (10, 10), (14, 14), (22, 22), (32, 32)]
COMPILE_REPS = 3      # G6: every timing point gets repetitions
BRANCH_REPS = 10      # PSR branches timed individually per n
NSR_REPS = 2 * N_NYQ  # one table per Nyquist branch

FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
CACHE = os.path.join(FIGDIR, "F_scale_data.json")


# ── targets ──────────────────────────────────────────────────────────────────

def chain_qs(n):
    x = sp.Symbol("x")
    qs = QSystem()
    q = [Qubit(qs) for _ in range(n)]
    H = x * sum((q[i].Z * q[i + 1].Z for i in range(n - 1)), 0 * q[0].Z)
    for i in range(n):
        H = H + q[i].X
    qs.add_evolution(H.set_parameterizedHam({"x": X_VAL}), T)
    return qs, H


def grid_qs(m, k):
    x = sp.Symbol("x")
    qs = QSystem()
    q = [Qubit(qs) for _ in range(m * k)]
    bonds = []
    for r in range(m):
        for c in range(k):
            i = r * k + c
            if c + 1 < k:
                bonds.append((i, i + 1))
            if r + 1 < m:
                bonds.append((i, i + k))
    H = x * sum((q[a].Z * q[b].Z for a, b in bonds), 0 * q[0].Z)
    for i in range(m * k):
        H = H + q[i].X
    qs.add_evolution(H.set_parameterizedHam({"x": X_VAL}), T)
    return qs, H


def compiled_H_error(prov, qs):
    comp = {}
    _n, _gv, boxes, _e, _tr = prov.prog
    for entries, _dur in boxes:
        for (_, ins, h_eval, _lv) in entries:
            for prod, c in h_eval.ham:
                k = prod.to_tuple()
                if k:
                    comp[k] = comp.get(k, 0.0) + float(c)
    targ = {}
    for prod, c in qs.evos[0][0].ham:
        k = prod.to_tuple()
        if k:
            targ[k] = targ.get(k, 0.0) + float(c)
    return max(abs(comp.get(k, 0.0) - targ.get(k, 0.0))
               for k in set(comp) | set(targ))


# ── per-pass instrumentation (breakdown runs only; headline runs unwrapped) ──

class PassTimers:
    def __init__(self):
        self.record_s = 0.0
        self.dressH_s = 0.0

    def install(self, mapper_mod, ledger_mod):
        self._orig_record = ledger_mod.PulseLedger.record
        self._orig_dress = mapper_mod.TweezerMapper._build_dressing_H
        timers = self

        def timed_record(self_, *a, **k):
            t0 = time.perf_counter()
            r = timers._orig_record(self_, *a, **k)
            timers.record_s += time.perf_counter() - t0
            return r

        def timed_dress(self_, *a, **k):
            t0 = time.perf_counter()
            r = timers._orig_dress(self_, *a, **k)
            timers.dressH_s += time.perf_counter() - t0
            return r

        ledger_mod.PulseLedger.record = timed_record
        mapper_mod.TweezerMapper._build_dressing_H = timed_dress

    def uninstall(self, mapper_mod, ledger_mod):
        ledger_mod.PulseLedger.record = self._orig_record
        mapper_mod.TweezerMapper._build_dressing_H = self._orig_dress


# ── timing ───────────────────────────────────────────────────────────────────

def time_point(builder, series):
    import tweezer_mapper as tm
    import pulse_ledger as pl
    from tweezer_mapper import TweezerMapper

    qs, H = builder()
    n = qs.num_sites
    prov = diffQCProvider()
    compile_times = []
    t0 = time.perf_counter()
    prov.compile(qs, "quera", "Aquila", "rydberg2d", tol=TOL,
                 specialize=(series != "generic"), verbose=0)
    compile_times.append(time.perf_counter() - t0)
    for _ in range(COMPILE_REPS - 1):
        qs_r, _ = builder()
        prov_r = diffQCProvider()
        t0 = time.perf_counter()
        prov_r.compile(qs_r, "quera", "Aquila", "rydberg2d", tol=TOL,
                       specialize=(series != "generic"), verbose=0)
        compile_times.append(time.perf_counter() - t0)

    row = dict(n=n, series=series,
               compile_s=float(np.median(compile_times)),
               compile_s_all=[float(t) for t in compile_times],
               max_dH=float(compiled_H_error(prov, qs)))
    if series == "generic":
        return row

    plan = prov._spec_plan
    row["dropped_zz"] = float(plan.dropped_zz_l1)

    # PSR branches: BRANCH_REPS distinct branches (distinct tau), timed one by
    # one on an unwrapped mapper. Marginal accounting: solver cached in boxes;
    # each branch re-emits schedule ops + ledger via map_hlist.
    np.random.seed(0)
    progs = observable_program_generator(H, T, n_sample=BRANCH_REPS,
                                         n_repetition=1,
                                         diff_var="x", value=X_VAL)
    row["M"] = len(progs)
    _n, gv, boxes, _e, _tr = prov.prog
    mapper = TweezerMapper(n_qubits=n, sol_gvars=gv, boxes=boxes,
                           ramp_time=0.01, dressing_pairs=plan.dressing_pairs)
    branch_lists = progs[0][0]
    mapper.map_hlist(branch_lists[0], T=T)          # warm-up
    psr_ms = []
    for H_list in branch_lists[:BRANCH_REPS]:
        t0 = time.perf_counter()
        mapper.map_hlist(H_list, T=T)
        psr_ms.append((time.perf_counter() - t0) * 1e3)
    row["psr_branch_ms"] = float(np.median(psr_ms))
    row["psr_branch_ms_all"] = [float(t) for t in psr_ms]
    led = mapper.map_hlist(branch_lists[0], T=T)[2]
    row["ledger_entries"] = len(led.entries)

    # per-pass breakdown (separate wrapped run; wrapper overhead noted)
    timers = PassTimers()
    timers.install(tm, pl)
    try:
        t0 = time.perf_counter()
        for H_list in branch_lists[:BRANCH_REPS]:
            mapper.map_hlist(H_list, T=T)
        total_wrapped = time.perf_counter() - t0
    finally:
        timers.uninstall(tm, pl)
    row["pass_ledger_ms"] = timers.record_s / BRANCH_REPS * 1e3
    row["pass_dressH_ms"] = timers.dressH_s / BRANCH_REPS * 1e3
    row["pass_total_wrapped_ms"] = total_wrapped / BRANCH_REPS * 1e3

    # NSR branches: closed-form coefficient table on the shared schedule.
    K = max(1.0, float(n))                          # only sets shift magnitudes
    shifts = [sgn * (j + 0.5) / (2 * K) for j in range(N_NYQ) for sgn in (1, -1)]
    specializer.nsr_shift_table(plan, shifts[0])    # warm-up
    nsr_ms = []
    for s in shifts[:NSR_REPS]:
        t0 = time.perf_counter()
        specializer.nsr_shift_table(plan, s)
        nsr_ms.append((time.perf_counter() - t0) * 1e3)
    row["nsr_branch_ms"] = float(np.median(nsr_ms))
    row["nsr_branch_ms_all"] = [float(t) for t in nsr_ms]
    return row


def run_timing():
    """Resumable: the cache is rewritten after every point; a killed run
    continues from the last completed point on the next invocation."""
    meta = dict(machine=platform.platform(), python=sys.version.split()[0],
                timing="wall-clock (time.perf_counter)",
                compile_reps=COMPILE_REPS, branch_reps=BRANCH_REPS,
                nsr_reps=NSR_REPS, m_tau=M_TAU, N_nyq=N_NYQ,
                T=T, x_val=X_VAL, tol=TOL, complete=False)
    if os.path.exists(CACHE):
        data = json.load(open(CACHE))
        rows = data["rows"]
        print(f"resuming ({len(rows)} points cached)")
    else:
        rows = []
    done = {(r["series"], r["n"]) for r in rows}

    def save():
        json.dump(dict(meta=meta, rows=rows), open(CACHE, "w"), indent=1,
                  default=float)

    points = ([("generic", n, lambda n=n: chain_qs(n)) for n in NS_GENERIC]
              + [("specialized", n, lambda n=n: chain_qs(n))
                 for n in NS_SPECIAL]
              + [("specialized2d", m * k, lambda m=m, k=k: grid_qs(m, k))
                 for m, k in GRIDS_SPECIAL])
    grid_of = {m * k: [m, k] for m, k in GRIDS_SPECIAL}
    for series, n, builder in points:
        if (series, n) in done:
            continue
        r = time_point(builder, series)
        if series == "specialized2d":
            r["grid"] = grid_of[n]
        rows.append(r)
        save()
        if series == "generic":
            print(f"  [{series}] n={n:>4} compile={r['compile_s']:8.2f}s",
                  flush=True)
        else:
            print(f"  [{series}] n={n:>4} compile={r['compile_s']:8.2f}s "
                  f"psr={r['psr_branch_ms']:7.1f}ms "
                  f"nsr={r['nsr_branch_ms']:.4f}ms", flush=True)
    meta["complete"] = True
    save()
    print(f"cached: {CACHE}")


# ── figures ──────────────────────────────────────────────────────────────────

BLUE = "#2a78d6"    # source compile (specialized path, 1D)
ORANGE = "#eb6834"  # +PSR increment
GREEN = "#008300"   # +NSR increment
AQUA = "#1baf7a"    # 2D source compile (appendix panel a only)
INK = "#0b0b0b"
SEC = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def series_rows(rows, name):
    return sorted((r for r in rows if r["series"] == name), key=lambda r: r["n"])


def fit_slope(n, t, lo, hi):
    n, t = np.asarray(n, float), np.asarray(t, float)
    m = (n >= lo) & (n <= hi)
    return np.polyfit(np.log(n[m]), np.log(t[m]), 1)[0] if m.sum() >= 2 else np.nan


def style_axis(a):
    a.set_facecolor(SURFACE)
    a.grid(True, which="major", color=GRID, linewidth=0.6)
    a.tick_params(colors=MUTED, labelsize=7)
    for s in a.spines.values():
        s.set_color(GRID)


def band(ax, ns, all_lists, color, scale=1.0):
    lo = [np.percentile(v, 25) * scale for v in all_lists]
    hi = [np.percentile(v, 75) * scale for v in all_lists]
    ax.fill_between(ns, lo, hi, color=color, alpha=0.18, linewidth=0)


def key_numbers(rows):
    spe = series_rows(rows, "specialized")
    big = spe[-1]
    ratio = big["psr_branch_ms"] / 1e3 / big["compile_s"]
    total_psr_s = 2 * M_TAU * big["M"] * big["psr_branch_ms"] / 1e3
    total_nsr_s = 2 * N_NYQ * big["nsr_branch_ms"] / 1e3
    return spe, big, ratio, total_psr_s, total_nsr_s


def build_strip(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = data["rows"]
    spe, big, ratio, _tp, _tn = key_numbers(rows)
    keep = [r for r in spe if r["n"] in (10, 50, 100, 500, 1000)]
    ns = [r["n"] for r in keep]
    src = [r["compile_s"] for r in keep]
    psr = [r["psr_branch_ms"] / 1e3 for r in keep]
    nsr = [r["nsr_branch_ms"] / 1e3 for r in keep]

    fig, ax = plt.subplots(figsize=(3.25, 2.55), dpi=300)
    fig.subplots_adjust(left=0.19, right=0.97, top=0.9, bottom=0.17)
    fig.patch.set_facecolor(SURFACE)
    style_axis(ax)

    ax.loglog(ns, src, "-o", color=BLUE, linewidth=1.6, markersize=4,
              markerfacecolor=BLUE, markeredgecolor=SURFACE, markeredgewidth=0.7)
    ax.loglog(ns, psr, "-o", color=ORANGE, linewidth=1.6, markersize=4,
              markerfacecolor=ORANGE, markeredgecolor=SURFACE, markeredgewidth=0.7)
    ax.loglog(ns, nsr, "-o", color=GREEN, linewidth=1.6, markersize=4,
              markerfacecolor=GREEN, markeredgecolor=SURFACE, markeredgewidth=0.7)
    band(ax, ns, [r["compile_s_all"] for r in keep], BLUE)
    band(ax, ns, [r["psr_branch_ms_all"] for r in keep], ORANGE, scale=1e-3)
    band(ax, ns, [r["nsr_branch_ms_all"] for r in keep], GREEN, scale=1e-3)

    # direct labels — legend lives in the caption at strip width
    ax.annotate("source compile", xy=(ns[1], src[1]), xytext=(-2, 10),
                textcoords="offset points", fontsize=7, color=BLUE)
    ax.annotate("+PSR / branch", xy=(ns[1], psr[1]), xytext=(2, -13),
                textcoords="offset points", fontsize=7, color=ORANGE)
    ax.annotate("+NSR / branch", xy=(ns[1], nsr[1]), xytext=(-2, 9),
                textcoords="offset points", fontsize=7, color=GREEN)
    ax.annotate(f"per branch =\n{ratio*100:.1f}% of source",
                xy=(ns[-1], psr[-1]), xytext=(-6, 14),
                textcoords="offset points", ha="right",
                fontsize=7, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6))

    ax.set_xlabel("qubits  n", fontsize=7.5, color=SEC)
    ax.set_ylabel("wall-time  (s)", fontsize=7.5, color=SEC)
    ax.set_title("Derivative compile cost over the source path",
                 fontsize=8, color=INK, pad=6)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"F_scale_strip.{ext}"))
    plt.close(fig)


def build_appendix(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = data["rows"]
    gen = series_rows(rows, "generic")
    spe, big, ratio, _tp, _tn = key_numbers(rows)
    sp2 = series_rows(rows, "specialized2d")
    ng = [r["n"] for r in gen]; tg = [r["compile_s"] for r in gen]
    ns = [r["n"] for r in spe]; ts = [r["compile_s"] for r in spe]
    n2 = [r["n"] for r in sp2]; t2 = [r["compile_s"] for r in sp2]
    psr = [r["psr_branch_ms"] / 1e3 for r in spe]
    nsr = [r["nsr_branch_ms"] / 1e3 for r in spe]

    sg = fit_slope(ng, tg, 5, 12)
    ss = fit_slope(ns, ts, 50, 1000)
    s2 = fit_slope(n2, t2, 50, 1024)
    psr_lo = fit_slope(ns, psr, 10, 200)
    psr_hi = fit_slope(ns, psr, 200, 1000)

    fig, (ax, axb) = plt.subplots(
        1, 2, figsize=(8.6, 3.5), dpi=200,
        gridspec_kw=dict(wspace=0.30, left=0.08, right=0.985,
                         top=0.86, bottom=0.15))
    fig.patch.set_facecolor(SURFACE)
    for a in (ax, axb):
        style_axis(a)
        a.tick_params(labelsize=8)

    # (a) base compilation — the solver story (appendix-scoped)
    ax.loglog(ng, tg, "-o", color=MUTED, linewidth=1.6, markersize=4.5,
              markerfacecolor=MUTED, markeredgecolor=SURFACE,
              markeredgewidth=0.8,
              label=f"generic (slope {sg:.1f}; caps n=12)")
    ax.loglog(ns, ts, "-o", color=BLUE, linewidth=1.8, markersize=5,
              markerfacecolor=BLUE, markeredgecolor=SURFACE,
              markeredgewidth=0.8,
              label=f"specialized 1D chain (slope {ss:.1f})")
    ax.loglog(n2, t2, "-o", color=AQUA, linewidth=1.8, markersize=5,
              markerfacecolor=AQUA, markeredgecolor=SURFACE,
              markeredgewidth=0.8,
              label=f"specialized 2D grid (slope {s2:.1f})")
    band(ax, ns, [r["compile_s_all"] for r in spe], BLUE)
    band(ax, n2, [r["compile_s_all"] for r in sp2], AQUA)
    guide = ts[-1] * (np.asarray(ns) / ns[-1]) ** 2
    ax.loglog(ns, guide, "--", color=MUTED, linewidth=1.0)
    ax.annotate(r"$\propto n^2$", xy=(ns[-4], guide[-4]), fontsize=8,
                color=MUTED, xytext=(2, -14), textcoords="offset points")
    ax.set_xlabel("qubits  n", fontsize=9, color=SEC)
    ax.set_ylabel("source compile wall-time  (s)", fontsize=9, color=SEC)
    ax.set_title("(a) Source compilation → schedule ops + ledger",
                 fontsize=9.5, color=INK, pad=8)
    ax.legend(fontsize=7, loc="lower right", frameon=False, labelcolor=SEC,
              handlelength=1.2, borderaxespad=0.4)

    # (b) incremental cost of differentiation — three series
    axb.loglog(ns, ts, "-", color=BLUE, linewidth=1.2, alpha=0.45,
               label="source compile (ref)")
    axb.loglog(ns, psr, "-o", color=ORANGE, linewidth=1.8, markersize=5,
               markerfacecolor=ORANGE, markeredgecolor=SURFACE,
               markeredgewidth=0.8, label="+PSR, one branch")
    axb.loglog(ns, nsr, "-o", color=GREEN, linewidth=1.8, markersize=5,
               markerfacecolor=GREEN, markeredgecolor=SURFACE,
               markeredgewidth=0.8, label="+NSR, one branch")
    band(axb, ns, [r["psr_branch_ms_all"] for r in spe], ORANGE, scale=1e-3)
    band(axb, ns, [r["nsr_branch_ms_all"] for r in spe], GREEN, scale=1e-3)
    axb.annotate(f"slope ≈ {psr_lo:.1f}  (n ≤ 200)", xy=(30, np.interp(30, ns, psr)),
                 fontsize=7.5, color=ORANGE, xytext=(-10, 16),
                 textcoords="offset points", ha="center")
    axb.annotate(f"slope ≈ {psr_hi:.1f}  (n ≥ 200)", xy=(ns[-1], psr[-1]),
                 fontsize=7.5, color=ORANGE, xytext=(-4, 10),
                 textcoords="offset points", ha="right")
    axb.annotate(f"{ratio*100:.1f}% of source\nat n=1000",
                 xy=(ns[-1], psr[-1]), xytext=(-4, -18),
                 textcoords="offset points", ha="right", fontsize=7.5,
                 color=INK)
    axb.set_xlabel("qubits  n", fontsize=9, color=SEC)
    axb.set_ylabel("wall-time  (s)", fontsize=9, color=SEC)
    axb.set_title("(b) Incremental cost per derivative branch",
                  fontsize=9.5, color=INK, pad=8)
    axb.legend(fontsize=7.5, loc="upper left", frameon=False, labelcolor=SEC,
               handlelength=1.5)

    # per-pass breakdown inset at n=1000 (wrapped-run share attribution)
    tot = big["pass_total_wrapped_ms"]
    parts = [("ledger", big["pass_ledger_ms"], "#a9a79f"),
             ("dress-H", big["pass_dressH_ms"], "#c3c2b7"),
             ("ops", tot - big["pass_ledger_ms"]
              - big["pass_dressH_ms"], "#e1e0d9")]
    ins = axb.inset_axes([0.52, 0.34, 0.4, 0.11])
    left = 0.0
    for lab, v, c in parts:
        ins.barh([0], [v], left=left, color=c, height=0.7)
        ins.text(left + v / 2, 0, f"{lab}\n{v/tot*100:.0f}%",
                 ha="center", va="center", fontsize=5.5, color=INK)
        left += v
    ins.set_xlim(0, tot)
    ins.set_yticks([])
    ins.set_xticks([])
    for spine in ins.spines.values():
        spine.set_color(GRID)
    ins.set_title("PSR per-pass split, n=1000", fontsize=6.5, color=SEC, pad=2)

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"F_scale_appendix.{ext}"))
    plt.close(fig)
    return sg, ss, s2, psr_lo, psr_hi


def write_notes(data, sg, ss, s2, psr_lo, psr_hi):
    rows = data["rows"]
    meta = data["meta"]
    spe, big, ratio, total_psr_s, total_nsr_s = key_numbers(rows)

    strip_cap = (
        "Fig C strip (F-scale-L): derivative compile cost over the source "
        "compilation path, specialized-path 1D TFIM chain, n=10..1000. Blue: "
        "source compile (target -> machine-native schedule ops + pulse "
        "ledger; pulse-shape synthesis excluded, see data note). Orange: "
        "marginal compile of ONE PSR branch (structural re-map: branch tau "
        "splits the evolution, kick inserts AOD+CZ; solver never re-run). "
        "Green: marginal compile of ONE NSR branch (closed-form coefficient "
        "table on the shared schedule structure, +0 segments). At n=1000 one "
        f"PSR branch is {big['psr_branch_ms']:.0f} ms = "
        f"{ratio*100:.1f}% of the {big['compile_s']:.0f} s source compile; "
        f"one NSR branch is {big['nsr_branch_ms']*1e3:.0f} µs. Bands: IQR "
        f"over {meta['compile_reps']} compiles / {meta['branch_reps']} "
        "branches per point. Generic-path ceiling, 2D grids, two-regime PSR "
        "slopes, and the per-pass breakdown: appendix F-scale figure.\n")
    open(os.path.join(FIGDIR, "F_scale_strip_caption.txt"), "w").write(strip_cap)

    app_cap = (
        "F-scale appendix. (a) Source compilation to machine-native schedule "
        f"ops + pulse ledger: the generic all-pairs path (slope ~{sg:.1f}) is "
        "practically capped at n=12; the target-aware specialized path (this "
        "claim is scoped to 1D chains and full 2D NN grids) reaches n=1000 "
        f"(1D, slope ~{ss:.1f}, {big['compile_s']:.0f} s) and 32x32=1024 "
        f"(2D, slope ~{s2:.1f}); IQR bands over {data['meta']['compile_reps']} "
        "compiles. (b) Marginal cost per derivative branch. PSR is "
        f"two-regime: slope ~{psr_lo:.1f} below n~200 and ~{psr_hi:.1f} "
        "above; the steepening is driven by the ledger position-snapshot "
        "pass (n positions x O(n) plays, structurally O(n^2) — its share "
        "grows from ~20% to ~36%), on top of the near-linear dressing-H "
        "rebuild, the largest single pass at n=1000 (inset: measured "
        "per-pass shares). NSR needs only an O(n) coefficient table "
        "on the shared schedule. Totals for one full gradient at n=1000 "
        f"(m={M_TAU} tau-samples, G2): PSR 2mM = {2*M_TAU*big['M']:,} "
        f"branches -> {total_psr_s/3600:.1f} h of mapping (amortizable: "
        "branches share segment structure, tau enters only as two "
        f"durations); NSR 2N = {2*N_NYQ} branches -> "
        f"{total_nsr_s*1e3:.1f} ms total.\n")
    open(os.path.join(FIGDIR, "F_scale_appendix_caption.txt"), "w").write(app_cap)

    note = f"""# F_scale data note (FSCALE_REVISION compliance)

Machine: {meta['machine']}; Python {meta['python']}; {meta['timing']}.
Repetitions (G6): {meta['compile_reps']} compiles/point (median, IQR band);
{meta['branch_reps']} PSR branches/point timed individually (distinct tau);
{meta['nsr_reps']} NSR tables/point. Warm-up: one untimed branch map / table
per point before timing. Cache: {os.path.basename(CACHE)} (delete to re-time).

## Accounting (R5)

**"One PSR branch" is MARGINAL cost**: the evolution solve is cached in the
compiled boxes; the timed operation is TweezerMapper.map_hlist re-emitting
schedule ops + pulse ledger for one [evolve(tau), kick, evolve(T-tau)]
branch. The re-map is structurally forced (branch-specific tau and kick
insertion) — it is not a recompile.

**"One NSR branch"** = specializer.nsr_shift_table: the Nyquist branch
B + s·A shares the source segment structure, so its marginal compile is an
O(n) closed-form coefficient rescale (dressing amplitude + detunings) bound
to the shared schedule at execution. Exactness is unit-tested
(test_nsr_shift_table_realizes_shifted_target). Measured
{big['nsr_branch_ms']*1e3:.0f} µs at n=1000 — well above the
~1 µs timer floor, so it is drawn at its measured value (R1).

**Totals for one gradient** (representative m={M_TAU}, matching
F6/F-loop/F-phase per G2; P=1 scalar parameter, M={big['M']} tangent
components at n=1000): PSR = 2·m·M = {2*M_TAU*big['M']:,} branch maps x
{big['psr_branch_ms']:.0f} ms = {total_psr_s/3600:.2f} h; NSR = 2·N =
{2*N_NYQ} tables = {total_nsr_s*1e3:.1f} ms. The PSR total is a mapping
(not solver) cost and is amortizable — branches share segment structure and
tau enters only as two durations — but the amortization is future work and
the un-amortized number is the honest one today.

## Scope (R5.2)

Timing runs target -> machine-native schedule ops (concrete amplitudes,
phases, durations, positions) + pulse ledger. It does NOT include
pulse-shape synthesis: the PulseDSL emission layer (placeholder shapes,
16-channel logical cap, physical-channel COMB encoding) does not complete
at n=100 and is excluded as engineering work. Figure titles say "schedule
ops + ledger", not "device-ready pulses"; G0 wording should follow.

## Two-regime PSR slope (R4a)

Fitted per window: slope ~{psr_lo:.1f} on n in [10, 200], ~{psr_hi:.1f} on
n in [200, 1000] (windows stated per R4b; the global average is never
annotated). Measured per-pass attribution across n: the dressing-H ledger
rebuild is the largest single pass throughout (~48% at n=1000,
{big['pass_dressH_ms']:.0f} ms) and grows near-linearly with a heavy
constant; the ledger position-snapshot pass (n positions x ~4n plays,
structurally O(n^2)) is the FASTEST-GROWING pass — its share rises from
~20% (n<=100) to ~36% at n=1000 ({big['pass_ledger_ms']:.0f} ms) — and is
what drives the tail steepening; op emission is the remainder
({big['pass_total_wrapped_ms']-big['pass_ledger_ms']-big['pass_dressH_ms']:.0f} ms).
Caveat: the breakdown comes from a separate instrumented run whose total
({big['pass_total_wrapped_ms']:.0f} ms) exceeds the unwrapped headline
median ({big['psr_branch_ms']:.0f} ms) by more than raw wrapper
arithmetic (12k wrapped calls plus allocator state); we therefore report
the breakdown as SHARES of the instrumented run, not as headline-additive
milliseconds.

## Headline ratio (R2)

One PSR branch at n=1000: {big['psr_branch_ms']:.0f} ms vs
{big['compile_s']:.0f} s source compile = **{ratio*100:.1f}%**.

Suggested 6.4 sentence: "At n=1000, compiling one PSR derivative branch
costs {big['psr_branch_ms']:.0f} ms — {ratio*100:.1f}% of the source
compilation it attaches to — and an NSR branch only an O(n) coefficient
table ({big['nsr_branch_ms']*1e3:.0f} µs): differentiation is not the
compilation bottleneck; the source path is."

## Scoping n=1000 (R3)

All n>12 numbers are the SPECIALIZED path (target-aware layer: frozen
chain/grid geometry, pruned bonds, analytic warm start); the generic
all-pairs path caps at n=12. Stated in both captions and panel (a).
"""
    open(os.path.join(FIGDIR, "F_scale_data_note.md"), "w").write(note)


def main():
    if not (os.path.exists(CACHE)
            and json.load(open(CACHE))["meta"].get("complete")):
        run_timing()
    data = json.load(open(CACHE))
    build_strip(data)
    slopes = build_appendix(data)
    write_notes(data, *slopes)
    spe, big, ratio, tp, tn = key_numbers(data["rows"])
    print(f"strip + appendix written; ratio at n=1000: {ratio*100:.2f}%  "
          f"PSR total (m={M_TAU}): {tp/3600:.2f} h  NSR total: {tn*1e3:.1f} ms")


if __name__ == "__main__":
    main()
