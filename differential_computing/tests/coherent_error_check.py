"""
coherent_error_check.py — verify the "slow/coherent-error suppression" cell.

A coherent error in the DIFFERENTIATION MACHINERY (the kick angle / the waveform
shift / the FD step), miscalibrated by a factor (1+η). We measure how the gradient
error scales with η: slope 1 = O(η) (NOT suppressed), slope 2 = O(η²) (branch-
symmetric suppression). A static error in the BASE program corrupts every method
equally (all differentiate the same errored landscape), so it cannot distinguish
them — the machinery error is the right probe.

Prediction: kick's two branches share the SAME base (only the ±kick flips), so the
shift sits at an extremum of the response → O(η²). Nyquist's ±s branches sit at
DIFFERENT operating points θ±s (like FD's θ±ε), so a shift miscalibration is only
O(η). This tells us whether Nyquist inherits the kick's suppression.

Noiseless (coherent errors only). Run:
  conda run -n qec_pg python differential_computing/tests/coherent_error_check.py
"""
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
from nyquist_shift import tangent_hamiltonian, bandwidth_K, nyquist_program_generator

T, X0, EPS_FD = 1.5, 0.7, 0.1


def main():
    x = sp.Symbol("x"); qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    H = x * q[0].Z + q[0].X
    runner = QuTiPSequentialRunner(2, nsteps=400000)
    expfn = runner.make_expectation_fn(runner.zero_state(), qp.tensor(qp.sigmaz(), qp.qeye(2)))
    C = lambda th: expfn([[H.set_parameterizedHam({"x": float(th)}), T]])
    g_ideal = (C(X0 + 1e-4) - C(X0 - 1e-4)) / 2e-4

    # nominal kick programs (deterministic midpoint τ). short_kick=True → both
    # branches use a ±π/4 kick on the SAME generator (symmetric), the structure the
    # branch-symmetric cancellation needs; a common over-rotation lands symmetrically
    # about the sin-extremum of the response.
    ns = 8
    tau = (np.arange(ns) + 0.5) / ns * T
    kick_progs = observable_program_generator(H, T, ns, 1, "x", X0, tau_list=tau,
                                              short_kick=True)
    # nominal Nyquist deterministic
    B, A = tangent_hamiltonian(H, "x", X0); K = bandwidth_K(A, T)
    nyq_progs, info = nyquist_program_generator(H, T, "x", X0, N=16)
    shifts = np.array(info["shifts"]); weights = np.array([p["weight"] for p in nyq_progs])

    def kick_eta(eta):                       # over-rotate the kick duration by (1+η)
        progs = []
        for H_tot, ug, nr in kick_progs:
            new = []
            for hl in H_tot:                 # hl = [[H,τ], [Hk,kick], [H,T-τ]]
                new.append([hl[0], [hl[1][0], (1 + eta) * hl[1][1]], hl[2]])
            progs.append([new, ug, nr])
        return combine_gradient_results(progs, expfn, T)

    def nyq_eta(eta):                        # miscalibrate the shift amplitude by (1+η)
        return float(sum(w * expfn([[B + (1 + eta) * s * A, T]])
                         for s, w in zip(shifts, weights)))

    def fd_eta(eta):                         # actual step (1+η)ε, nominal ε in denom
        e = (1 + eta) * EPS_FD
        return (C(X0 + e) - C(X0 - e)) / (2 * EPS_FD)

    # isolate the coherent-error effect: Δ(η) = |g(η) − g(0)|, subtracting each
    # method's own η=0 baseline (τ-/truncation/ε bias) that would otherwise mask it.
    g0 = {"kick": kick_eta(0.0), "nyq": nyq_eta(0.0), "fd": fd_eta(0.0)}
    fn = {"kick": kick_eta, "nyq": nyq_eta, "fd": fd_eta}
    etas = np.array([0.005, 0.01, 0.02, 0.04, 0.08])
    print(f"∇C_ideal = {g_ideal:+.5f}   (T={T}, x0={X0})")
    print(f"baselines g(0): kick={g0['kick']:+.5f} nyq={g0['nyq']:+.5f} fd={g0['fd']:+.5f}")
    print(f"\nΔ(η)=|g(η)−g(0)|  (coherent-error-induced change)")
    print(f"{'η':>8}  {'kick':>12}  {'Nyquist':>12}  {'FD':>12}")
    d = {"kick": [], "nyq": [], "fd": []}
    for eta in etas:
        row = {k: abs(fn[k](eta) - g0[k]) for k in fn}
        for k in fn:
            d[k].append(row[k])
        print(f"{eta:8.3f}  {row['kick']:12.2e}  {row['nyq']:12.2e}  {row['fd']:12.2e}")

    print("\nlog-log slope (1 = O(η) NOT suppressed, 2 = O(η²) branch-symmetric suppression):")
    for k in ("kick", "nyq", "fd"):
        y = np.maximum(np.array(d[k]), 1e-14)
        sl = np.polyfit(np.log(etas), np.log(y), 1)[0]
        verdict = "SUPPRESSED O(η²) ✓" if sl > 1.6 else "not suppressed O(η) ✗"
        print(f"  {k:8}: slope {sl:.2f}  → {verdict}")


if __name__ == "__main__":
    main()
