"""
plot_h2_vqe.py — render the H2 VQE PSR-vs-FD comparison (reproducing Fig 2b of
Leng et al. 2022) as a figure: ground-state energy vs epoch for PSR (no ε) and
FD at several ε, mean ± spread over seeds, with the true ground energy E0.

Saves to differential_computing/figures/h2_vqe_psr_vs_fd.png.

Run:  conda run -n qec_pg python differential_computing/tests/plot_h2_vqe.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import h2_vqe_psr_vs_fd as v


def main():
    b_obs, eta, n_epochs, seeds = 100, 0.10, 60, 4
    fd_eps_list = [0.01, 0.1, 0.5]
    runner = v.NoisyQuTiPRunner(2, noise=None)

    rng0 = np.random.RandomState(3)
    v0 = rng0.uniform(-1.0, 1.0, v.NP)

    runs = [("PSR", None)] + [("FD", e) for e in fd_eps_list]
    res = {}
    for method, eps in runs:
        key = "PSR (no ε)" if method == "PSR" else f"FD ε={eps}"
        E = np.zeros((seeds, n_epochs + 1))
        for s in range(seeds):
            E[s] = v.descend(method, v0, runner, b_obs, eta, eps or 0.1,
                             n_epochs, seed=10 + s)
        res[key] = E

    steps = np.arange(n_epochs + 1)
    styles = {
        "PSR (no ε)": dict(color="#1f77b4", lw=2.4, ls="-"),
        "FD ε=0.01":  dict(color="#d62728", lw=2.0, ls="-"),
        "FD ε=0.1":   dict(color="#ff7f0e", lw=1.8, ls="--"),
        "FD ε=0.5":   dict(color="#9467bd", lw=1.8, ls=":"),
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    for k, E in res.items():
        mu, sd = E.mean(0), E.std(0)
        st = styles[k]
        ax.plot(steps, mu, label=k, **st)
        ax.fill_between(steps, mu - sd, mu + sd, color=st["color"], alpha=0.15)
    ax.axhline(v.E0, ls="--", color="k", lw=1, label=f"true ground $E_0$={v.E0:.3f}")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"energy  $\langle H_{\mathrm{H_2}}\rangle$")
    ax.set_title("H$_2$ VQE under shot noise (b_obs=100): PSR vs FD\n"
                 "PSR needs no ε; FD at small ε=0.01 stalls (shot noise amplified)")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.abspath(os.path.join(out_dir, "h2_vqe_psr_vs_fd.png"))
    fig.savefig(out)
    print(f"saved: {out}")
    for k, E in res.items():
        print(f"  {k:>12}: final {E[:, -1].mean():.4f}  (gap {E[:, -1].mean()-v.E0:+.4f})")


if __name__ == "__main__":
    main()
