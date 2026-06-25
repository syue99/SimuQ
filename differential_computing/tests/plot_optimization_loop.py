"""
plot_optimization_loop.py — render the optimization-loop capstone as a figure.

Reuses optimization_loop_demo's machinery (same cost, noise, estimators) and
produces a 2-panel figure:
  (A) ideal cost vs iteration — PSR vs FD, mean ± spread over seeds, true min C*.
  (B) parameter θ vs iteration — PSR settles at x*, FD orbits it.

Saves to differential_computing/figures/optimization_loop.png.

Run:  conda run -n qec_pg python differential_computing/tests/plot_optimization_loop.py
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
    # true minimum (single basin)
    grid = np.linspace(0.35, 1.25, 91)
    cgrid = np.array([d.ideal_cost(float(x)) for x in grid])
    x_star = float(grid[np.argmin(cgrid)])
    c_star = float(cgrid.min())

    x0, eta, n_steps, N, fd_eps, seeds = 0.45, 0.5, 30, 300, 0.15, 6
    steps = np.arange(n_steps + 1)

    traj = {}
    for method in ("PSR", "FD"):
        C = np.zeros((seeds, n_steps + 1))
        X = np.zeros((seeds, n_steps + 1))
        for s in range(seeds):
            rng = np.random.default_rng(1000 + s)
            costs, xs = d.descend(method, x0, x_star, eta, n_steps, N, fd_eps, rng)
            C[s], X[s] = costs, xs
        traj[method] = (C, X)

    colors = {"PSR": "#1f77b4", "FD": "#d62728"}
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)

    # Panel A: cost vs iteration
    for m in ("PSR", "FD"):
        C = traj[m][0]
        mu, sd = C.mean(0), C.std(0)
        axA.plot(steps, mu, color=colors[m], lw=2, label=f"{m} gradient")
        axA.fill_between(steps, mu - sd, mu + sd, color=colors[m], alpha=0.20)
    axA.axhline(c_star, ls="--", color="k", lw=1, label="true min $C^*$")
    axA.set_xlabel("descent iteration")
    axA.set_ylabel(r"ideal cost  $\langle Z_0 Z_1\rangle(\theta)$")
    axA.set_title("(A) cost vs iteration")
    axA.legend(frameon=False, fontsize=9)

    # Panel B: parameter trajectory
    for m in ("PSR", "FD"):
        X = traj[m][1]
        mu, sd = X.mean(0), X.std(0)
        axB.plot(steps, mu, color=colors[m], lw=2, label=f"{m} gradient")
        axB.fill_between(steps, mu - sd, mu + sd, color=colors[m], alpha=0.20)
    axB.axhline(x_star, ls="--", color="k", lw=1, label=r"true min $\theta^*$")
    axB.set_xlabel("descent iteration")
    axB.set_ylabel(r"parameter $\theta$")
    axB.set_title(r"(B) $\theta$ trajectory")
    axB.legend(frameon=False, fontsize=9)

    fig.suptitle(
        f"Gradient descent under noise (dephasing T2=5 + gate err), "
        f"N={N} shots/step, {seeds} seeds — PSR converges & settles, FD orbits",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.abspath(os.path.join(out_dir, "optimization_loop.png"))
    fig.savefig(out)
    print(f"saved: {out}")
    # quick numeric recap
    for m in ("PSR", "FD"):
        C, X = traj[m]
        print(f"  {m}: final cost {C[:, -1].mean():.4f} (gap {C[:, -1].mean()-c_star:+.4f}), "
              f"|x-x*|={np.abs(X[:, -1]-x_star).mean():.4f}, "
              f"last-10 θ jitter={X[:, -10:].std(1).mean():.4f}")


if __name__ == "__main__":
    main()
