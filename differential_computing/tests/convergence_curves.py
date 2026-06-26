"""
convergence_curves.py — actual convergence trajectories (not endpoints) for the
fair paired PSR-vs-FD comparisons.

Panel A — H2 VQE: energy vs epoch, PSR (n=4) vs FD (ε=0.1), faithful T2=2, EQUAL
  budget, mean ± std over the SAME starts × seeds (paired).
Panel B — MaxCut QAOA (4-cycle): cut vs epoch, PSR vs FD, shot noise, paired.
  True max cut = 4 (dashed); plateaus below 4 are local optima, not the optimum.

Saves figures/convergence_curves.png.

Run:  conda run -n qec_pg python differential_computing/tests/convergence_curves.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import vqe_noisy_comparison as q
import h2_fair_comparison as f
import maxcut_psr_vs_fd as mc
from observable_program_generator import observable_program_generator
from combine_gradient import combine_gradient_results
from noisy_qutip import NoisyQuTiPRunner
from noise_model import NoiseModel


def mc_psr_grad(v, runner, b_obs, rng, seed, n_sample):
    """MaxCut PSR gradient with a tunable n_sample (mc.ascend hardcodes 1)."""
    g = np.zeros(mc.NP)
    expfn = mc.cut_expfn(runner, b_obs, rng)
    for k in range(mc.NP):
        np.random.seed(seed + k)
        progs = observable_program_generator(
            mc.H_param_k(v, k), mc.T, n_sample=n_sample, n_repetition=1,
            diff_var="vk", value=float(v[k]))
        g[k] = combine_gradient_results(progs, expfn, mc.T)
    return g


def mc_ascend(method, eps, n_sample, v0, runner, b_obs, eta, n_epochs, seed):
    v = v0.copy(); cuts = [mc.true_cut(v)]
    rng = np.random.default_rng(seed)
    for ep in range(n_epochs):
        if method == "FD":
            g = mc.fd_grad(v, runner, b_obs, eps, rng)
        else:
            g = mc_psr_grad(v, runner, b_obs, rng, seed + 7 * ep, n_sample)
        v = v + eta * g; cuts.append(mc.true_cut(v))
    return np.array(cuts)


def h2_curves():
    runner = NoisyQuTiPRunner(2, noise=NoiseModel(n_qubits=2, T2=2.0))
    rng0 = np.random.RandomState(7)
    starts = [rng0.uniform(-1.0, 1.0, q.NP) for _ in range(4)]
    eta, n_epochs, seeds = 0.1, 40, 2
    P, F = [], []
    for i, v0 in enumerate(starts):
        for s in range(seeds):
            sd = 3000 + 17 * i + s
            P.append(f.descend("PSR", None, 4, v0, runner, eta, n_epochs,
                               seed=sd, b_obs=50))
            F.append(f.descend("FD", 0.1, 1, v0, runner, eta, n_epochs,
                               seed=sd, b_obs=200))
    return q.E0, np.array(P), np.array(F)


def maxcut_curves():
    # FAIR: PSR n=4 @ b_obs=25 vs FD ε=0.1 @ b_obs=100 (equal budget); SAME
    # start + seed per trial (paired); shot noise only.
    runner = NoisyQuTiPRunner(mc.N, noise=None)
    starts = [np.random.RandomState(2 + i).uniform(-0.5, 0.5, mc.NP)
              for i in range(3)]
    eta, n_epochs = 0.08, 40
    P, F = [], []
    for i, v0 in enumerate(starts):
        sd = 200 + i
        P.append(mc_ascend("PSR", None, 4, v0, runner, 25, eta, n_epochs, sd))
        F.append(mc_ascend("FD", 0.1, 1, v0, runner, 100, eta, n_epochs, sd))
    return mc.MAXCUT, np.array(P), np.array(F)


def main():
    E0, hP, hF = h2_curves()
    MX, mP, mF = maxcut_curves()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=150)

    eh = np.arange(hP.shape[1])
    for D, c, lbl in ((hP, "#1f77b4", "PSR (n=4)"), (hF, "#d62728", "FD (ε=0.1)")):
        mu, sd = D.mean(0), D.std(0)
        axA.plot(eh, mu, color=c, lw=2.2, label=lbl)
        axA.fill_between(eh, mu - sd, mu + sd, color=c, alpha=0.15)
    axA.axhline(E0, ls="--", color="k", lw=1, label=f"$E_0$={E0:.3f}")
    axA.set_xlabel("epoch"); axA.set_ylabel(r"energy $\langle H_{\mathrm{H_2}}\rangle$")
    axA.set_title("(A) H$_2$ VQE — energy vs epoch (paired, T2=2, equal budget)")
    axA.legend(frameon=False, fontsize=9, loc="upper right")

    em = np.arange(mP.shape[1])
    for D, c, lbl in ((mP, "#1f77b4", "PSR"), (mF, "#d62728", "FD (ε=0.1)")):
        mu, sd = D.mean(0), D.std(0)
        axB.plot(em, mu, color=c, lw=2.2, label=lbl)
        axB.fill_between(em, mu - sd, mu + sd, color=c, alpha=0.15)
    axB.axhline(MX, ls="--", color="k", lw=1, label=f"true max cut = {MX:.0f}")
    axB.set_xlabel("epoch"); axB.set_ylabel(r"cut $\langle C\rangle$")
    axB.set_title("(B) MaxCut QAOA — local-optima dominated (paired)")
    axB.legend(frameon=False, fontsize=9, loc="lower right")

    fig.suptitle("Convergence (fair paired): PSR wins the smooth H$_2$ basin "
                 "(gradient quality matters);\nMaxCut is non-convex — neither "
                 "reaches 4, and coarse gradients explore local optima better",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "figures", "convergence_curves.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"H2: PSR final {hP[:,-1].mean():.4f}, FD final {hF[:,-1].mean():.4f} "
          f"(E0={E0:.4f})")
    print(f"MaxCut: PSR final {mP[:,-1].mean():.4f}, FD final {mF[:,-1].mean():.4f} "
          f"(max cut={MX:.0f})")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
