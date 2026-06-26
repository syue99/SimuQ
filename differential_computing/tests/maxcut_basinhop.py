"""
maxcut_basinhop.py — basin-hopping with memory: clean gradient descent within each
basin, stochastic jumps to escape, keep the BEST optimum found.  The precise
gradient (PSR) should reach a better best-cut than the coarse one (FD).

Why this isolates gradient quality better than continuous annealing: the inner
gradient ascent is NOT corrupted by exploration noise, so a more precise gradient
estimate navigates each basin to a better local optimum.  The jumps (same seed for
both methods) provide controlled exploration; "keep best" means neither regresses.
Under FINITE shots, PSR (n=4) is a more precise estimate than FD (ε=0.1) at equal
budget → it finds better optima per basin → higher best cut.

MaxCut 4-cycle, shot noise (equal budget), same start+seed, several starts.
Track best-so-far cut vs basin-hop; plot.  True max cut = 4.

Run:  conda run -n qec_pg python differential_computing/tests/maxcut_basinhop.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import maxcut_psr_vs_fd as mc
from observable_program_generator import observable_program_generator
from combine_gradient import combine_gradient_results
from noisy_qutip import NoisyQuTiPRunner


def psr_grad(v, runner, b_obs, rng, seed, n_sample):
    g = np.zeros(mc.NP)
    expfn = mc.cut_expfn(runner, b_obs, rng)
    for k in range(mc.NP):
        np.random.seed(seed + k)
        progs = observable_program_generator(
            mc.H_param_k(v, k), mc.T, n_sample=n_sample, n_repetition=1,
            diff_var="vk", value=float(v[k]))
        g[k] = combine_gradient_results(progs, expfn, mc.T)
    return g


def basinhop(method, eps, n_sample, b_obs, v0, runner, eta, n_hops, inner,
             sigma, seed):
    rng = np.random.default_rng(seed)
    jrng = np.random.default_rng(seed + 99)        # jump stream (SAME for both)
    v = v0.copy()
    best_v, best_cut = v.copy(), mc.true_cut(v)
    hist = [best_cut]
    for hop in range(n_hops):
        for it in range(inner):                    # clean ascent to a local opt
            if method == "FD":
                g = mc.fd_grad(v, runner, b_obs, eps, rng)
            else:
                g = psr_grad(v, runner, b_obs, rng, seed + 7 * (hop * inner + it),
                             n_sample)
            v = v + eta * g
        c = mc.true_cut(v)
        if c > best_cut:
            best_cut, best_v = c, v.copy()
        hist.append(best_cut)
        v = best_v + sigma * jrng.standard_normal(mc.NP)   # jump from the best
    return np.array(hist)


def main():
    runner = NoisyQuTiPRunner(mc.N, noise=None)        # shot noise only
    starts = [np.random.RandomState(2 + i).uniform(-0.5, 0.5, mc.NP)
              for i in range(3)]
    eta, n_hops, inner, sigma = 0.1, 7, 10, 0.8

    P, F = [], []
    for i, v0 in enumerate(starts):
        sd = 400 + i
        P.append(basinhop("PSR", None, 4, 25, v0, runner, eta, n_hops, inner,
                          sigma, sd))
        F.append(basinhop("FD", 0.1, 1, 100, v0, runner, eta, n_hops, inner,
                          sigma, sd))
    P, F = np.array(P), np.array(F)

    print(f"MaxCut basin-hopping (keep best).  PSR n=4 @b25 vs FD ε=0.1 @b100, "
          f"equal budget, same start+seed jumps.")
    print(f"max cut={mc.MAXCUT:.0f}, {n_hops} hops × {inner} inner steps, "
          f"σ={sigma}, {len(starts)} starts.\n")
    print(f"{'method':>10}{'best cut':>11}{'deficit':>10}")
    print(f"{'PSR':>10}{P[:,-1].mean():>11.4f}{mc.MAXCUT - P[:,-1].mean():>10.4f}")
    print(f"{'FD':>10}{F[:,-1].mean():>11.4f}{mc.MAXCUT - F[:,-1].mean():>10.4f}")

    hops = np.arange(P.shape[1])
    fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=150)
    for D, c, lbl in ((P, "#1f77b4", "PSR (n=4)"), (F, "#d62728", "FD (ε=0.1)")):
        mu, sd = D.mean(0), D.std(0)
        ax.plot(hops, mu, "o-", color=c, lw=2.2, label=lbl)
        ax.fill_between(hops, mu - sd, mu + sd, color=c, alpha=0.15)
    ax.axhline(mc.MAXCUT, ls="--", color="k", lw=1, label="true max cut = 4")
    ax.set_xlabel("basin-hop"); ax.set_ylabel(r"best cut so far  $\langle C\rangle$")
    ax.set_title("MaxCut basin-hopping (clean descent + jumps, keep best):\n"
                 "precise PSR gradient finds better local optima → higher best cut")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "figures", "maxcut_basinhop.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
