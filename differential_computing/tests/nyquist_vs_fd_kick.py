"""
nyquist_vs_fd_kick.py — three-way differentiation-strategy comparison harness.

Computes ∂⟨O⟩/∂θ on the SAME analog program by three sound routes plus the FD
baseline, on a shared QuTiP runner, and reports accuracy vs the number of
program executions (the shot-normalized cost unit):

  * FD            — central difference, 2 executions per ε (baseline, biased).
  * kick-PSR      — Algorithm 1 of arXiv:2210.15812; 2·n_sample executions PER
                    Pauli generator term (branch count grows with the sum).
  * Nyquist       — waveform shift (arXiv:2207.01587); the tangent folds ALL
                    terms into ONE direction, so 2N executions regardless of how
                    many H_j the parameter touches (deterministic), or n_sample
                    executions (stochastic).

This is the noiseless accuracy-vs-cost skeleton; the noisy/finite-shot layer
reuses NoisyQuTiPRunner with the same programs.  Caches figures/nyquist_vs_fd_kick.json.
Run:  conda run -n qec_pg python differential_computing/tests/nyquist_vs_fd_kick.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sympy as sp
import qutip as qp

from simuq import QSystem, Qubit
from qutip_sequential import QuTiPSequentialRunner
from observable_program_generator import observable_program_generator
from combine_gradient import combine_gradient_results
from nyquist_shift import nyquist_program_generator, combine_nyquist_results

FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
T, X0, NSTEPS = 1.5, 0.7, 300000


def build_1q():
    x = sp.Symbol("x"); qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    return x * q[0].Z + q[0].X, 1                       # 1 x-dependent term

def build_coupled():
    x = sp.Symbol("x"); qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    J = sp.sin(2 * x)
    return J * q[0].Z * q[1].Z + J * q[0].X + J * q[1].X, 3   # 3 x-dependent terms


def n_exec_kick(programs):
    return sum(len(H_tot) for H_tot, _, _ in programs)  # total H_lists executed

def compare(label, H, n_terms, expfn):
    def f(xv):
        return expfn([[H.set_parameterizedHam({"x": xv}), T]])
    truth = (f(X0 + 1e-4) - f(X0 - 1e-4)) / 2e-4        # fine-FD ground truth

    out = {"label": label, "n_terms": n_terms, "truth": truth,
           "fd": [], "kick": [], "nyquist_det": [], "nyquist_stoch": []}

    # FD baseline swept over ε (2 executions each)
    for eps in (0.5, 0.2, 0.1, 0.05, 0.02, 0.01):
        est = (f(X0 + eps) - f(X0 - eps)) / (2 * eps)
        out["fd"].append({"eps": eps, "n_exec": 2, "est": est, "err": abs(est - truth)})

    # kick-PSR swept over n_sample (2·n_sample·n_terms executions)
    for ns in (8, 32, 128, 512):
        np.random.seed(0)
        progs = observable_program_generator(H, T, n_sample=ns, n_repetition=1,
                                             diff_var="x", value=X0)
        est = combine_gradient_results(progs, expfn, T)
        out["kick"].append({"n_sample": ns, "n_exec": n_exec_kick(progs),
                            "est": est, "err": abs(est - truth)})

    # Nyquist deterministic swept over N pairs (2N executions, term-count-free)
    for N in (2, 4, 8, 16, 32):
        progs, info = nyquist_program_generator(H, T, "x", X0, N=N, mode="deterministic")
        est = combine_nyquist_results(progs, expfn)
        out["nyquist_det"].append({"N": N, "K": info["K"], "n_exec": len(progs),
                                   "est": est, "err": abs(est - truth)})

    # Nyquist stochastic swept over n_sample (n_sample executions)
    for ns in (32, 128, 512, 2000):
        progs, info = nyquist_program_generator(H, T, "x", X0, mode="stochastic",
                                                n_sample=ns, seed=0, max_n=32)
        est = combine_nyquist_results(progs, expfn)
        out["nyquist_stoch"].append({"n_sample": ns, "n_exec": len(progs),
                                     "est": est, "err": abs(est - truth)})
    return out


def show(o):
    print(f"\n=== {o['label']}  (θ-dependent terms: {o['n_terms']}, truth={o['truth']:+.5f}) ===")
    print("  FD baseline:")
    for r in o["fd"]:
        print(f"    ε={r['eps']:.2f}  exec={r['n_exec']:<4d} est={r['est']:+.5f} err={r['err']:.2e}")
    print("  kick-PSR (2·n_sample·n_terms exec):")
    for r in o["kick"]:
        print(f"    n_sample={r['n_sample']:<4d} exec={r['n_exec']:<5d} est={r['est']:+.5f} err={r['err']:.2e}")
    print(f"  Nyquist deterministic (2N exec, K={o['nyquist_det'][0]['K']:.3f}, term-count-free):")
    for r in o["nyquist_det"]:
        print(f"    N={r['N']:<3d} exec={r['n_exec']:<4d} est={r['est']:+.5f} err={r['err']:.2e}")
    print("  Nyquist stochastic (n_sample exec):")
    for r in o["nyquist_stoch"]:
        print(f"    n_sample={r['n_sample']:<4d} exec={r['n_exec']:<5d} est={r['est']:+.5f} err={r['err']:.2e}")


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    runner = QuTiPSequentialRunner(2, nsteps=NSTEPS)
    expfn = runner.make_expectation_fn(runner.zero_state(),
                                       qp.tensor(qp.sigmaz(), qp.qeye(2)))
    results = []
    for label, (H, nt) in (("1q  H=x·Z+X", build_1q()),
                           ("coupled  sin(2x)(ZZ+X+X)", build_coupled())):
        o = compare(label, H, nt, expfn); show(o); results.append(o)
    cache = os.path.join(FIGDIR, "nyquist_vs_fd_kick.json")
    json.dump({"T": T, "x0": X0, "systems": results}, open(cache, "w"), indent=2, default=float)
    print(f"\ncached: {cache}")


if __name__ == "__main__":
    main()
