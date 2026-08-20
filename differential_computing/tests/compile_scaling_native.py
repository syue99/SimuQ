"""
compile_scaling_native.py (RQ3) — compile-time scaling of the specialized
(device-native) 1D TFIM path vs the generic all-pairs path, plus the
per-branch incremental cost of differentiation.

Measures, per n:
  generic      — prov.compile(specialize=False): free positions, all-pairs
                 ZZ/dressing machine (the path benchmarked in
                 compile_scaling.py; practical ceiling n≈12).
  specialized  — prov.compile(specialize=True): frozen chain geometry,
                 pruned bonds, analytic warm start, sparse Jacobian.
  branch_ms    — wall-clock per PSR branch for the hardware map
                 (map_hlist → ops + pulse ledger); this is the entire
                 incremental cost of differentiation per branch, since the
                 evolution solve is shared across all 2·M·n_sample branches.
  max_dH       — max |coefficient| deviation of the compiled Hamiltonian
                 (sum of active instruction h_evals) from the target.
  dropped_zz   — declared 1/R^6 dressing-truncation bound from the plan.

Results are cached to figures/compile_scaling_native.json — delete the file
to re-time.  Plotting lives in plot_compile_scaling_native.py.

Run:  conda run -n qec_pg python differential_computing/tests/compile_scaling_native.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sympy as sp

from simuq import QSystem, Qubit
from simuq.braket.diffQC_provider import diffQCProvider
from observable_program_generator import observable_program_generator

T = 1.0
X_VAL = 0.8
TOL = 0.1
NS_GENERIC = [2, 3, 4, 5, 6, 8, 10, 12]
NS_SPECIAL = [4, 6, 8, 10, 20, 50, 100, 200, 500, 1000]
# 2D series: full m×k NN grids (n = m·k), diagonal 1/R^6 tail declared-dropped
GRIDS_SPECIAL = [(4, 4), (7, 7), (10, 10), (14, 14), (22, 22), (32, 32)]
REPS_SMALL = 3          # repetitions for dispersion when a compile is cheap
SMALL_CUTOFF_S = 5.0    # single rep beyond this


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
    """max |coef| deviation of Σ active instruction h_evals vs the target."""
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


def time_compile(builder, series, specialize):
    qs, H = builder()
    n = qs.num_sites
    prov = diffQCProvider()
    times = []
    t0 = time.perf_counter()
    prov.compile(qs, "quera", "Aquila", "rydberg2d", tol=TOL,
                 specialize=specialize, verbose=0)
    times.append(time.perf_counter() - t0)
    if times[0] < SMALL_CUTOFF_S:
        for _ in range(REPS_SMALL - 1):
            qs_r, _ = builder()
            prov_r = diffQCProvider()
            t0 = time.perf_counter()
            prov_r.compile(qs_r, "quera", "Aquila", "rydberg2d", tol=TOL,
                           specialize=specialize, verbose=0)
            times.append(time.perf_counter() - t0)
    err = compiled_H_error(prov, qs)
    row = dict(n=n, series=series,
               compile_s=float(np.median(times)),
               compile_s_all=[float(t) for t in times],
               segs=len(prov.prog[2]), max_dH=float(err))
    if specialize:
        row["dropped_zz"] = float(prov._spec_plan.dropped_zz_l1)
        # incremental cost: hardware-map 2 PSR branches of one program
        np.random.seed(0)
        progs = observable_program_generator(H, T, n_sample=1, n_repetition=1,
                                             diff_var="x", value=X_VAL)
        row["M"] = len(progs)
        # throwaway warm-up map: the first map_hlist call pays module import
        # and allocation costs that would otherwise pollute the smallest n
        prov.run(progs[:1], None, T, backend="hardware", verbose=0)
        t0 = time.perf_counter()
        prov.run(progs[:1], None, T, backend="hardware", verbose=0)
        nb = sum(len(b) for b, _, _ in prov._branch_ops)
        row["branch_ms"] = (time.perf_counter() - t0) / nb * 1e3
        led = prov.get_pulse_ledger(program_idx=0, branch_idx=0)
        row["ledger_entries"] = len(led.entries)
    return row


def main():
    figdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    os.makedirs(figdir, exist_ok=True)
    cache = os.path.join(figdir, "compile_scaling_native.json")
    rows = json.load(open(cache)) if os.path.exists(cache) else []
    have = {r["series"] for r in rows}
    if rows:
        print(f"loaded cache ({len(rows)} rows) — delete {cache} to re-time")

    def spec_print(r, extra=""):
        print(f"  n={r['n']:>4}  compile={r['compile_s']:8.2f}s  "
              f"branch={r['branch_ms']:6.1f}ms  M={r['M']:>4}  "
              f"ledger={r['ledger_entries']:>5}  "
              f"max|dH|={r['max_dH']:.1e}  dropped_zz={r['dropped_zz']:.3f}"
              f"{extra}", flush=True)

    dirty = False
    if "generic" not in have:
        print("series=generic (all-pairs machine, free positions)")
        for n in NS_GENERIC:
            r = time_compile(lambda n=n: chain_qs(n), "generic", specialize=False)
            rows.append(r)
            print(f"  n={n:>4}  compile={r['compile_s']:8.2f}s  "
                  f"max|dH|={r['max_dH']:.1e}", flush=True)
        dirty = True
    if "specialized" not in have:
        print("series=specialized (frozen chain, pruned, warm start)")
        for n in NS_SPECIAL:
            r = time_compile(lambda n=n: chain_qs(n), "specialized",
                             specialize=True)
            rows.append(r)
            spec_print(r)
        dirty = True
    if "specialized2d" not in have:
        print("series=specialized2d (frozen m×k grid; diagonal J/8 tail "
              "declared-dropped)")
        for m, k in GRIDS_SPECIAL:
            r = time_compile(lambda m=m, k=k: grid_qs(m, k), "specialized2d",
                             specialize=True)
            r["grid"] = [m, k]
            rows.append(r)
            spec_print(r, extra=f"  grid={m}x{k}")
        dirty = True
    if dirty:
        json.dump(rows, open(cache, "w"), indent=1, default=float)
        print(f"cached: {cache}")

    gen = [r for r in rows if r["series"] == "generic"]
    spe = [r for r in rows if r["series"] == "specialized"]
    if gen and spe:
        both = {r["n"]: r["compile_s"] for r in gen}
        print("\nn where both series exist — speedup:")
        for r in spe:
            if r["n"] in both:
                print(f"  n={r['n']:>3}: {both[r['n']]:.2f}s → "
                      f"{r['compile_s']:.3f}s  ({both[r['n']]/r['compile_s']:.0f}x)")


if __name__ == "__main__":
    main()
