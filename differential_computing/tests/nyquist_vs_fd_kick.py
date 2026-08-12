"""
[DEPRECATED — superseded by build_F3.py / noisy_nyquist_vs_fd_kick.py]
The NOISELESS accuracy-vs-executions view here is misleading: the 1-qubit case is
trivial and the ~1e-6 error floor is unreachable on a real device. The realistic
comparison (finite shots + δ + dephasing) is noisy_nyquist_vs_fd_kick.py, and the
shot-cost scaling is case_study_kick_vs_nyquist.py; both feed Figure F3. Figure
moved to figures/deprecated/. Kept for reference only.

nyquist_vs_fd_kick.py — three-way differentiation-strategy numerics.

Computes ∂⟨O⟩/∂θ on the SAME analog program by three sound routes plus the FD
baseline, on a shared QuTiP runner, as accuracy vs number of program executions
(the shot-normalized cost unit).  All estimators use their BEST-CASE
DETERMINISTIC picking (no gratuitous sampling variance):

  * FD            — central difference, 2 executions per ε (baseline, biased).
  * kick-PSR      — Algorithm 1 (arXiv:2210.15812) with deterministic midpoint
                    split times τ_k=(k+½)/n·T; 2·n·n_terms executions.
  * Nyquist       — waveform shift (arXiv:2207.01587), deterministic truncated
                    paired sum; 2N executions, term-count-FREE (one tangent
                    folds the whole extensive sum).  "none" = exact weights,
                    "lanczos" = σ-factor apodization.

The STOCHASTIC Nyquist estimator is kept in the plot to show the gap that
deterministic picking closes — the analogue of deterministic vs random τ for
kick.  Caches figures/nyquist_vs_fd_kick.json and a convergence figure.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simuq import QSystem, Qubit
from qutip_sequential import QuTiPSequentialRunner
from observable_program_generator import observable_program_generator
from combine_gradient import combine_gradient_results
from nyquist_shift import nyquist_program_generator, combine_nyquist_results

FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
T, X0, NSTEPS = 1.5, 0.7, 400000


def build_1q():
    x = sp.Symbol("x"); qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    return x * q[0].Z + q[0].X, 1

def build_coupled():
    x = sp.Symbol("x"); qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    J = sp.sin(2 * x)
    return J * q[0].Z * q[1].Z + J * q[0].X + J * q[1].X, 3


def n_exec_kick(programs):
    return sum(len(H_tot) for H_tot, _, _ in programs)


def compare(label, H, n_terms, expfn):
    def f(xv):
        return expfn([[H.set_parameterizedHam({"x": xv}), T]])
    truth = (f(X0 + 1e-4) - f(X0 - 1e-4)) / 2e-4

    out = {"label": label, "n_terms": n_terms, "truth": truth,
           "fd": [], "kick_det": [], "nyquist_none": [],
           "nyquist_lanczos": [], "nyquist_stoch": []}

    for eps in (0.5, 0.2, 0.1, 0.05, 0.02, 0.01):
        est = (f(X0 + eps) - f(X0 - eps)) / (2 * eps)
        out["fd"].append({"n_exec": 2, "est": est, "err": abs(est - truth)})

    for ns in (2, 4, 8, 16, 32):                        # kick, deterministic midpoint τ
        tau = (np.arange(ns) + 0.5) / ns * T
        progs = observable_program_generator(H, T, ns, 1, "x", X0, tau_list=tau)
        est = combine_gradient_results(progs, expfn, T)
        out["kick_det"].append({"n_exec": n_exec_kick(progs), "est": est, "err": abs(est - truth)})

    for N in (2, 4, 8, 16, 32, 48):                     # Nyquist deterministic
        for win, key in (("none", "nyquist_none"), ("lanczos", "nyquist_lanczos")):
            progs, info = nyquist_program_generator(H, T, "x", X0, N=N,
                                                    mode="deterministic", window=win)
            est = combine_nyquist_results(progs, expfn)
            out[key].append({"n_exec": len(progs), "K": info["K"],
                             "est": est, "err": abs(est - truth)})

    for ns in (32, 128, 512, 2000):                     # Nyquist stochastic (the gap)
        progs, _ = nyquist_program_generator(H, T, "x", X0, mode="stochastic",
                                             n_sample=ns, seed=0, max_n=32)
        est = combine_nyquist_results(progs, expfn)
        out["nyquist_stoch"].append({"n_exec": len(progs), "est": est, "err": abs(est - truth)})
    return out


def show(o):
    print(f"\n=== {o['label']}  (θ-terms {o['n_terms']}, truth {o['truth']:+.5f}) ===")
    for key, tag in (("fd", "FD"), ("kick_det", "kick det-τ"),
                     ("nyquist_none", "Nyquist none"), ("nyquist_lanczos", "Nyquist lanczos"),
                     ("nyquist_stoch", "Nyquist stoch")):
        pts = ", ".join(f"{r['n_exec']}:{r['err']:.1e}" for r in o[key])
        print(f"  {tag:16s} (exec:err)  {pts}")


STYLE = {
    "fd":             ("FD (biased)",        "#D55E00", "s", "--"),
    "kick_det":       ("kick-PSR (det-τ)",   "#009E73", "o", "-"),
    "nyquist_none":   ("Nyquist det",        "#0072B2", "^", "-"),
    "nyquist_lanczos":("Nyquist det+Lanczos","#56B4E9", "v", "-"),
    "nyquist_stoch":  ("Nyquist stochastic", "#999999", "x", ":"),
}


def figure(results):
    fig, axs = plt.subplots(1, len(results), figsize=(4.4 * len(results), 3.6), squeeze=False)
    for ax, o in zip(axs[0], results):
        for key, (lab, c, mk, ls) in STYLE.items():
            xs = [r["n_exec"] for r in o[key]]
            ys = [max(r["err"], 1e-12) for r in o[key]]
            order = np.argsort(xs)
            ax.loglog(np.array(xs)[order], np.array(ys)[order], ls, color=c,
                      marker=mk, ms=5, label=lab)
        ax.set_title(f"{o['label']}  ({o['n_terms']} θ-term%s)" % ("s" if o["n_terms"] > 1 else ""),
                     fontsize=9)
        ax.set_xlabel("program executions"); ax.set_ylabel("|gradient error|")
        ax.grid(True, which="both", alpha=0.15)
    axs[0][0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    out = os.path.join(FIGDIR, "nyquist_vs_fd_kick.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\nfigure: {out}")


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    runner = QuTiPSequentialRunner(2, nsteps=NSTEPS)
    expfn = runner.make_expectation_fn(runner.zero_state(),
                                       qp.tensor(qp.sigmaz(), qp.qeye(2)))
    results = []
    for label, (H, nt) in (("1q  x·Z+X", build_1q()),
                           ("coupled  sin(2x)(ZZ+X+X)", build_coupled())):
        o = compare(label, H, nt, expfn); show(o); results.append(o)
    json.dump({"T": T, "x0": X0, "systems": results},
              open(os.path.join(FIGDIR, "nyquist_vs_fd_kick.json"), "w"), indent=2, default=float)
    figure(results)


if __name__ == "__main__":
    main()
