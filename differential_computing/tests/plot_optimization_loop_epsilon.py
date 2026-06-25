"""
plot_optimization_loop_epsilon.py — the optimization loop, sweeping FD's ε.

Extends optimization_loop_demo: descend the noisy cost with PSR (no ε) and FD at
SEVERAL ε, same per-step shot budget, faithful noise model.  The point is the
ε-dilemma made visible in a loop: FD evaluates at θ±ε, so its gradient variance
never vanishes near the minimum —
  - small ε → 1/(2ε) blows up the variance → it orbits widely,
  - large ε → the secant spans the landscape's curvature → it settles to a
    biased/offset point,
and NO ε settles as precisely as PSR, which evaluates AT θ (variance → 0 at the
minimum) and pins exactly.

Two panels: cost vs iteration, and the parameter θ trajectory.  Saves to
differential_computing/figures/optimization_loop_epsilon.png.

Run:  conda run -n qec_pg python differential_computing/tests/plot_optimization_loop_epsilon.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import optimization_loop_demo as d


def main():
    grid = np.linspace(0.35, 1.25, 91)
    cgrid = np.array([d.ideal_cost(float(x)) for x in grid])
    x_star = float(grid[np.argmin(cgrid)])
    c_star = float(cgrid.min())

    x0, eta, n_steps, N, seeds = 0.45, 0.5, 30, 300, 6
    fd_eps = [0.05, 0.15, 0.4, 0.8]
    steps = np.arange(n_steps + 1)

    runs = [("PSR (no ε)", "PSR", None)] + [(f"FD ε={e}", "FD", e) for e in fd_eps]
    res = {}
    for label, method, eps in runs:
        C = np.zeros((seeds, n_steps + 1)); X = np.zeros((seeds, n_steps + 1))
        for s in range(seeds):
            rng = np.random.default_rng(1000 + s)
            costs, xs = d.descend(method, x0, x_star, eta, n_steps, N, eps or 0.15, rng)
            C[s], X[s] = costs, xs
        res[label] = (C, X)

    print(f"Optimization loop, FD ε-sweep.  x*={x_star:.3f}, C*={c_star:.4f}, "
          f"N={N} shots/step, {seeds} seeds.\n")
    print(f"{'method':>12}{'final cost':>12}{'gap to C*':>11}{'|x-x*|':>9}"
          f"{'settle jitter':>14}")
    for label, (C, X) in res.items():
        fc = C[:, -1].mean()
        dx = np.abs(X[:, -1] - x_star).mean()
        jit = X[:, -10:].std(1).mean()
        print(f"{label:>12}{fc:>12.4f}{fc - c_star:>11.4f}{dx:>9.4f}{jit:>14.4f}")

    colors = {"PSR (no ε)": "#1f77b4"}
    fdc = ["#d62728", "#ff7f0e", "#9467bd", "#8c564b"]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=150)
    for i, (label, (C, X)) in enumerate(res.items()):
        c = colors.get(label, fdc[(i - 1) % len(fdc)])
        lw = 2.6 if label.startswith("PSR") else 1.7
        for ax, D in ((axA, C), (axB, X)):
            mu, sd = D.mean(0), D.std(0)
            ax.plot(steps, mu, label=label, color=c, lw=lw)
            ax.fill_between(steps, mu - sd, mu + sd, color=c, alpha=0.12)
    axA.axhline(c_star, ls="--", color="k", lw=1, label="true min $C^*$")
    axA.set_xlabel("iteration"); axA.set_ylabel(r"cost $\langle Z_0Z_1\rangle$")
    axA.set_title("(A) cost vs iteration")
    axA.legend(frameon=False, fontsize=8, loc="upper right")
    axB.axhline(x_star, ls="--", color="k", lw=1, label=r"true min $\theta^*$")
    axB.set_xlabel("iteration"); axB.set_ylabel(r"parameter $\theta$")
    axB.set_title(r"(B) $\theta$ trajectory")
    axB.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Optimization loop, FD ε-sweep (faithful noise): PSR pins the "
                 "minimum; no FD ε settles\n(small ε → variance orbits, "
                 "large ε → biased offset)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.abspath(os.path.join(out_dir, "optimization_loop_epsilon.png"))
    fig.savefig(out)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
