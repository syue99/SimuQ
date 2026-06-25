"""
maxcut_free_shots.py — with FREE shots (exact expectations), does PSR reach the
correct MaxCut answer?  Isolates gradient BIAS (correctness) from shot VARIANCE.

Same MaxCut 4-cycle QAOA as maxcut_psr_vs_fd, but the cut is measured EXACTLY (no
shot sampling).  PSR is then a stochastic gradient only through its τ sampling
(n_sample); FD is exact at fine ε.  If PSR drives the cut to 4 (the optimum), its
gradient is correct/unbiased and any earlier shortfall was shot variance, not bias.

We run PSR at a few n_sample (1 = τ-stochastic, up to ~exact) and FD at fine ε,
gradient ascent, and report the cut reached.

Run:  conda run -n qec_pg python differential_computing/tests/maxcut_free_shots.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import qutip as qp

import maxcut_psr_vs_fd as m
from observable_program_generator import observable_program_generator
from combine_gradient import combine_gradient_results
from noisy_qutip import NoisyQuTiPRunner


def exact_expfn(runner):
    """Exact <C> on the branch final state — free (unlimited) shots."""
    def expfn(H_list):
        rho = runner.run_sequence(H_list, m.PLUS)
        return float(qp.expect(m.C, rho).real)
    return expfn


def psr_grad(v, runner, seed, n_sample):
    g = np.zeros(m.NP)
    expfn = exact_expfn(runner)
    for k in range(m.NP):
        np.random.seed(seed + k)
        progs = observable_program_generator(
            m.H_param_k(v, k), m.T, n_sample=n_sample, n_repetition=1,
            diff_var="vk", value=float(v[k]))
        g[k] = combine_gradient_results(progs, expfn, m.T)
    return g


def fd_grad(v, runner, eps):
    expfn = exact_expfn(runner)
    g = np.zeros(m.NP)
    for k in range(m.NP):
        vp = v.copy(); vp[k] += eps
        vm = v.copy(); vm[k] -= eps
        g[k] = (expfn([[m.H_of_v(vp), m.T]]) - expfn([[m.H_of_v(vm), m.T]])) / (2 * eps)
    return g


def ascend(method, eps, n_sample, v0, runner, eta, n_epochs, seed):
    v = v0.copy(); cuts = [m.true_cut(v)]
    for ep in range(n_epochs):
        if method == "FD":
            g = fd_grad(v, runner, eps)
        else:
            g = psr_grad(v, runner, seed + 7 * ep, n_sample)
        v = v + eta * g
        cuts.append(m.true_cut(v))
    return np.array(cuts)


def main():
    runner = NoisyQuTiPRunner(m.N, noise=None)        # noiseless + free shots
    v0 = np.random.RandomState(2).uniform(-0.5, 0.5, m.NP)
    eta, n_epochs, seeds = 0.08, 60, 3

    print(f"MaxCut QAOA, FREE shots (exact <C>), noiseless.  max cut={m.MAXCUT:.0f}, "
          f"{m.NP} params, T={m.T}, η={eta}, {n_epochs} epochs, {seeds} seeds.\n")
    print(f"{'config':>18}{'final cut':>11}{'deficit':>10}{'best cut':>11}")

    configs = [("FD ε=0.01", "FD", 0.01, 1)] + \
              [(f"PSR n_sample={k}", "PSR", None, k) for k in (1, 5, 20)]
    for label, method, eps, ns in configs:
        Cu = np.zeros((seeds, n_epochs + 1))
        for s in range(seeds):
            Cu[s] = ascend(method, eps or 0.01, ns, v0, runner, eta, n_epochs,
                           seed=100 + s)
        fc = Cu[:, -1].mean()
        # maximization → best = highest cut along the (mean) trajectory
        print(f"{label:>18}{fc:>11.4f}{m.MAXCUT - fc:>10.4f}{Cu.mean(0).max():>11.4f}")

    print(f"\nIf PSR (any n_sample) drives the cut to ≈{m.MAXCUT:.0f}, its gradient "
          f"is correct/unbiased\n— earlier shortfalls were shot VARIANCE, not bias. "
          f"n_sample=1 is τ-stochastic\n(noisier path) but should still reach the "
          f"optimum since the τ-noise is zero-mean.")


if __name__ == "__main__":
    main()
