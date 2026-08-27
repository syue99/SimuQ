"""
sec6_compile_timing.py — SEC6 handover D (tab:strategies + Appendix G curve).

D1-D3: extend the resumable F_scale cache with n=30, 300 (spec wants
n = 10, 30, 100, 300, 1000) and report source / +PSR / +NSR at 10^3.
D4  (\\owed{FD}): does an FD branch re-run the source solve or reuse it?
Measured: wall-time of a full specialized compile at the FD-shifted value
x+ε (what a black-box FD branch must do — the pipeline has no FD path),
vs the specializer's closed-form coefficient table (the only reuse path,
which IS the differentiation infrastructure's own shift table).
D5  (\\owed{P/k scan}): per-branch increment for PSR and NSR at
P ∈ {1,5,20} × k ∈ {1,4,14}, fixed n=300. The source compile is shared
across cells (numeric target identical); only differentiation structure
varies. NSR timed two ways: full O(n) channel-table emission (what the
runtime ships per branch) and the k-scoped arithmetic update.
D6: paper_fig_3/figs/F_scale_app.pdf — log-log increments vs n + D5 panel.

Run: conda run -n qec_pg python differential_computing/tests/sec6_compile_timing.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sympy as sp

import build_F_scale as fs
from simuq import QSystem, Qubit
from simuq import specializer
from simuq.braket.diffQC_provider import diffQCProvider
from observable_program_generator import observable_program_generator

FIGDIR = fs.FIGDIR
OUT3 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "paper_fig_3", "figs"))
RESULT = os.path.join(FIGDIR, "sec6_compile_timing.json")
N_SCAN = 300
FD_EPS = 0.05
PS_SCAN, KS_SCAN = [1, 5, 20], [1, 4, 14]


def chain_pk(n, P, k):
    """Chain TFIM with P differentiated coefficients, each over k disjoint
    bonds (bonds (l-1)k..lk-1); remaining bonds carry the fixed numeric
    coupling. All values = X_VAL, so the numeric target — and hence the
    source compile — is identical across (P, k) cells."""
    assert P * k <= n - 1
    syms = [sp.Symbol(f"th{l}") for l in range(P)]
    qs = QSystem()
    q = [Qubit(qs) for _ in range(n)]
    H = 0 * q[0].Z
    for b in range(n - 1):
        zz = q[b].Z * q[b + 1].Z
        l = b // k
        H = H + (syms[l] * zz if l < P else fs.X_VAL * zz)
    for i in range(n):
        H = H + q[i].X
    qs.add_evolution(H.set_parameterizedHam(
        {f"th{l}": fs.X_VAL for l in range(P)}), fs.T)
    return qs, H


def main():
    out = {}

    # ── D1-D3: fill n=30, 300 into the resumable F_scale cache ──
    fs.NS_SPECIAL = [30, 300]
    fs.NS_GENERIC = []
    fs.GRIDS_SPECIAL = []
    fs.run_timing()
    rows = json.load(open(fs.CACHE))["rows"]
    spec = {r["n"]: r for r in rows if r["series"] == "specialized"}
    out["D1_D3_rows"] = {n: dict(compile_s=spec[n]["compile_s"],
                                 psr_branch_ms=spec[n]["psr_branch_ms"],
                                 nsr_branch_ms=spec[n]["nsr_branch_ms"])
                         for n in (10, 30, 100, 300, 1000) if n in spec}
    r1k = spec[1000]
    print(f"D1: source@1000 = {r1k['compile_s']:.1f}s   "
          f"D2: +PSR = {r1k['psr_branch_ms']:.0f}ms "
          f"({r1k['psr_branch_ms']/10/r1k['compile_s']:.2f}% of source)   "
          f"D3: +NSR = {r1k['nsr_branch_ms']:.3f}ms "
          f"(10^{np.log10(r1k['nsr_branch_ms']/1e3):.1f} s)")

    # ── D4: FD branch — full recompile at the shifted value vs table reuse ──
    qs0, _ = fs.chain_qs(N_SCAN)
    prov = diffQCProvider()
    t0 = time.perf_counter()
    prov.compile(qs0, "quera", "Aquila", "rydberg2d", tol=fs.TOL,
                 specialize=True, verbose=0)
    src_s = time.perf_counter() - t0
    plan = prov._spec_plan

    xs = fs.X_VAL
    fs.X_VAL = xs + FD_EPS          # FD + branch target
    try:
        qs_fd, _ = fs.chain_qs(N_SCAN)
    finally:
        fs.X_VAL = xs
    prov_fd = diffQCProvider()
    t0 = time.perf_counter()
    prov_fd.compile(qs_fd, "quera", "Aquila", "rydberg2d", tol=fs.TOL,
                    specialize=True, verbose=0)
    fd_full_s = time.perf_counter() - t0
    specializer.nsr_shift_table(plan, FD_EPS)      # warm
    t0 = time.perf_counter()
    for _ in range(20):
        specializer.nsr_shift_table(plan, FD_EPS)
    fd_table_ms = (time.perf_counter() - t0) / 20 * 1e3
    out["D4"] = dict(n=N_SCAN, source_s=src_s, fd_full_recompile_s=fd_full_s,
                     fd_pct_of_source=100.0 * fd_full_s / src_s,
                     fd_table_reuse_ms=fd_table_ms)
    print(f"D4 (n={N_SCAN}): source={src_s:.2f}s, FD branch full recompile="
          f"{fd_full_s:.2f}s ({100*fd_full_s/src_s:.0f}% of source); "
          f"table-reuse path {fd_table_ms:.3f}ms (= the specializer's own "
          f"shift table)")

    # ── D5: P/k scan at n=300, shared source compile ──
    from tweezer_mapper import TweezerMapper
    _n, gv, boxes, _e, _tr = prov.prog
    scan = {"PSR": {}, "NSR_full": {}, "NSR_scoped": {}}
    for P in PS_SCAN:
        for k in KS_SCAN:
            qs_pk, H_pk = chain_pk(N_SCAN, P, k)
            np.random.seed(0)
            progs = observable_program_generator(H_pk, fs.T, n_sample=6,
                                                 n_repetition=1,
                                                 diff_var="th0", value=fs.X_VAL)
            mapper = TweezerMapper(n_qubits=N_SCAN, sol_gvars=gv, boxes=boxes,
                                   ramp_time=0.01,
                                   dressing_pairs=plan.dressing_pairs)
            branch_lists = progs[0][0]
            mapper.map_hlist(branch_lists[0], T=fs.T)          # warm-up
            ts = []
            for H_list in branch_lists[:6]:
                t0 = time.perf_counter()
                mapper.map_hlist(H_list, T=fs.T)
                ts.append((time.perf_counter() - t0) * 1e3)
            scan["PSR"][f"P{P}_k{k}"] = float(np.median(ts))

            # NSR: full table emission (all n channels — what ships per
            # branch) and the k-scoped arithmetic update
            s = 0.05
            t0 = time.perf_counter()
            for _ in range(50):
                specializer.nsr_shift_table(plan, s)
            scan["NSR_full"][f"P{P}_k{k}"] = (time.perf_counter() - t0) / 50 * 1e3
            sites = sorted({b for bond in range(k) for b in (bond, bond + 1)})
            scale = (plan.theta + s) / plan.theta
            t0 = time.perf_counter()
            for _ in range(200):
                o = plan.dressing_init * scale
                d = {i: plan.detuning_init[i]
                     + 2.0 * plan.theta_sum[i] * (scale - 1.0) for i in sites}
            scan["NSR_scoped"][f"P{P}_k{k}"] = (time.perf_counter() - t0) / 200 * 1e3
            print(f"  D5 P={P:2d} k={k:2d}: PSR {scan['PSR'][f'P{P}_k{k}']:7.1f}ms  "
                  f"NSR full {scan['NSR_full'][f'P{P}_k{k}']:.4f}ms  "
                  f"scoped {scan['NSR_scoped'][f'P{P}_k{k}']:.5f}ms", flush=True)
    out["D5"] = scan
    out["meta"] = dict(n_scan=N_SCAN, fd_eps=FD_EPS, T=fs.T, x_val=fs.X_VAL,
                       tol=fs.TOL, branch_reps=6)
    json.dump(out, open(RESULT, "w"), indent=1, default=float)

    # ── D6: appendix figure — increments vs n + D5 panel ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ns = sorted(n for n in spec if n >= 4)
    src = [spec[n]["compile_s"] for n in ns]
    psr = [spec[n]["psr_branch_ms"] / 1e3 for n in ns]
    nsr = [spec[n]["nsr_branch_ms"] / 1e3 for n in ns]
    plt.rcParams.update({"font.size": 7})
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(3.4, 4.6), dpi=300,
                                   height_ratios=[1.5, 1])
    axA.loglog(ns, src, "o-", color=fs.BLUE, lw=1.3, ms=3.5, label="source compile")
    axA.loglog(ns, psr, "s-", color=fs.ORANGE, lw=1.3, ms=3.5, label="+PSR / branch")
    axA.loglog(ns, nsr, "^-", color=fs.GREEN, lw=1.3, ms=3.5, label="+NSR / branch")
    axA.set_xlabel("qubits  n", fontsize=7.5)
    axA.set_ylabel("wall time (s)", fontsize=7.5)
    axA.tick_params(labelsize=7)
    axA.grid(True, which="both", alpha=0.15)
    axA.legend(fontsize=7, loc="upper left", framealpha=0.85)
    axA.text(0.97, 0.03, "specialized path, 1D chain",
             transform=axA.transAxes, fontsize=7, color="#52514e",
             ha="right", va="bottom")
    for k, col, mk in zip(KS_SCAN, ["#f2a175", "#eb6834", "#a34518"],
                          ["o", "s", "^"]):
        axB.plot(PS_SCAN, [scan["PSR"][f"P{P}_k{k}"] for P in PS_SCAN],
                 marker=mk, ls="-", color=col, lw=1.2, ms=3.2,
                 label=f"PSR, k={k}")
    axB.axhline(np.mean(list(scan["NSR_full"].values())), color=fs.GREEN,
                lw=1.2, ls="--")
    axB.text(PS_SCAN[-1], np.mean(list(scan["NSR_full"].values())) * 1.4,
             "NSR (all cells)", fontsize=7, color=fs.GREEN, ha="right")
    axB.set_yscale("log")
    axB.set_xticks(PS_SCAN)
    axB.set_xlabel("differentiated coefficients  P   (n=300)", fontsize=7.5)
    axB.set_ylabel("per-branch (ms)", fontsize=7.5)
    axB.tick_params(labelsize=7)
    axB.grid(True, which="both", alpha=0.15)
    axB.legend(fontsize=7, framealpha=0.85)
    fig.tight_layout(pad=0.5)
    os.makedirs(OUT3, exist_ok=True)
    fig.savefig(os.path.join(OUT3, "F_scale_app.pdf"), bbox_inches="tight",
                pad_inches=0.02)
    fig.savefig(os.path.join(OUT3, "F_scale_app.png"), bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)
    print("wrote paper_fig_3/figs/F_scale_app.pdf/.png + sec6_compile_timing.json")


if __name__ == "__main__":
    main()
