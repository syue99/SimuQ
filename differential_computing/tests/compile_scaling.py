"""
compile_scaling.py (C7) — does the differentiable COMPILATION scale?  (systems, no
dynamics simulation needed.)

For TFIM chains θ·ΣZ_iZ_{i+1}+ΣX_i, n=2.., we compile the analog evolution to a
neutral-atom (rydberg2d) pulse schedule and record the resources of the GRADIENT
program:
  - M            = # terms the parameter multiplies (= n-1 ZZ bonds) → # PSR programs
  - branches     = 2 · M · n_sample  (the gradient program's quantum-execution count)
  - compile time = wall-clock of prov.compile (the SimuQ analog solver + mapper)
  - #segments / total pulse duration of the compiled evolution (from prov.prog boxes)

The gradient program is 2·M·n_sample branches, each a 3-segment [evolve, kick,
evolve] H_list that compiles via the SAME evolution machinery benchmarked in SimuQ
to 96 sites.  So the differentiable compile cost = branch-count × per-evolution
compile — both polynomial.  This is the systems-scalability evidence that pairs with
the light-cone locality of the noise findings (small-n dynamics is representative).

Run:  conda run -n qec_pg python differential_computing/tests/compile_scaling.py
"""

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
N_SAMPLE = 100
NS = [2, 3, 4, 5, 6, 8, 10, 12]   # extend if the solver stays fast; prints per-n


def build_H(n, x):
    qs = QSystem(); q = [Qubit(qs) for _ in range(n)]
    H = x * sum((q[i].Z*q[i+1].Z for i in range(n-1)), 0*q[0].Z)
    for i in range(n): H = H + q[i].X
    return H


def resources(boxes):
    """#segments and total pulse duration from the compiled boxes."""
    nseg = len(boxes)
    dur = 0.0
    for b in boxes:
        try:
            dur += float(b[1])
        except (TypeError, ValueError, IndexError):
            pass
    return nseg, dur


def run_n(n):
    x = sp.Symbol("x")
    H = build_H(n, x)
    x_val = 0.8

    # gradient-program structure (cheap, no compile)
    np.random.seed(0)
    progs = observable_program_generator(H, T, n_sample=N_SAMPLE, n_repetition=1,
                                         diff_var="x", value=x_val)
    M = len(progs)
    branches = 2 * M * N_SAMPLE

    # compile the analog evolution to a neutral-atom schedule (time it)
    prov = diffQCProvider()
    qs_c = QSystem(); _ = [Qubit(qs_c) for _ in range(n)]
    qs_c.add_evolution(H.set_parameterizedHam({"x": x_val}), T)
    t0 = time.perf_counter()
    try:
        prov.compile(qs_c, "quera", "Aquila", "rydberg2d", tol=0.1, verbose=0)
        ct = time.perf_counter() - t0
        _n, _g, boxes, _e, _tr = prov.prog
        nseg, dur = resources(boxes)
        status = "ok"
    except Exception as e:
        ct = time.perf_counter() - t0
        nseg, dur, status = 0, 0.0, f"FAIL({type(e).__name__})"

    print(f"  n={n:>2}  M={M:>2}  branches={branches:>5}  compile={ct:>8.2f}s  "
          f"segs={nseg:>3}  pulse_dur={dur:>6.2f}  [{status}]", flush=True)
    return dict(n=n, M=M, branches=branches, compile_s=ct, segs=nseg, dur=dur,
                status=status)


def main():
    print(f"T={T}, n_sample={N_SAMPLE}, AAIS=rydberg2d\n")
    print(f"{'n':>4}{'M':>4}{'branches':>10}{'compile(s)':>12}"
          f"{'segs':>6}{'pulse_dur':>11}")
    rows = []
    for n in NS:
        r = run_n(n)
        rows.append(r)

    # LaTeX-ready table dump
    print("\n% --- LaTeX table body (C7) ---")
    for r in rows:
        print(f"{r['n']} & {r['M']} & {r['branches']} & "
              f"{r['compile_s']:.2f} & {r['segs']} & {r['dur']:.2f} \\\\")
    ok = [r for r in rows if r["status"] == "ok"]
    if ok:
        print(f"\ncompiled {len(ok)}/{len(rows)} sizes; "
              f"largest n={max(r['n'] for r in ok)} in "
              f"{max(r['compile_s'] for r in ok):.1f}s")


if __name__ == "__main__":
    main()
