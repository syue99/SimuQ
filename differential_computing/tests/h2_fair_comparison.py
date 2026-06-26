"""
h2_fair_comparison.py — PSR vs FD on H2 VQE with IDENTICAL starts and seeds.

Earlier H2 comparisons may have been muddied by different starts / seed handling
and by local-optima sensitivity.  Here PSR and FD use the SAME set of random
starts (and the same per-trial seed), so any difference is method, not setup.

Part 1 — free shots (exact <H_H2>, noiseless): per-start final energy.  If PSR
  (high n_sample) matches FD per start, PSR's gradient is correct and the path is
  comparable.
Part 2 — noisy, EQUAL shot budget (PSR n_sample=4 @ b_obs=50 vs FD ε=0.1 @
  b_obs=200), faithful model, same starts × seeds: averaged final gap.

Run:  conda run -n qec_pg python differential_computing/tests/h2_fair_comparison.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import qutip as qp

import vqe_noisy_comparison as q
from observable_program_generator import observable_program_generator
from combine_gradient import combine_gradient_results
from noisy_qutip import NoisyQuTiPRunner
from noise_model import NoiseModel

T = 1.0


def exact_expfn(runner):
    def expfn(H_list):
        rho = runner.run_sequence(H_list, q.PSI0)
        return float(qp.expect(q.H_H2, rho).real)
    return expfn


def psr_grad(v, runner, expfn, seed, n_sample):
    g = np.zeros(q.NP)
    for k in range(q.NP):
        np.random.seed(seed + k)
        progs = observable_program_generator(
            q.H_param_k(v, k), T, n_sample=n_sample, n_repetition=1,
            diff_var="vk", value=float(v[k]))
        g[k] = combine_gradient_results(progs, expfn, T)
    return g


def fd_grad(v, runner, expfn, eps):
    g = np.zeros(q.NP)
    for k in range(q.NP):
        vp = v.copy(); vp[k] += eps
        vm = v.copy(); vm[k] -= eps
        g[k] = (expfn([[q.H_of_v(vp), T]]) - expfn([[q.H_of_v(vm), T]])) / (2 * eps)
    return g


def descend(method, eps, n_sample, v0, runner, eta, n_epochs, seed, b_obs=None):
    v = v0.copy(); E = [q.true_energy(v, T)]
    rng = np.random.default_rng(seed)
    expfn = (exact_expfn(runner) if b_obs is None
             else q.energy_expfn(runner, b_obs, rng))
    for ep in range(n_epochs):
        if method == "FD":
            g = fd_grad(v, runner, expfn, eps)
        else:
            g = psr_grad(v, runner, expfn, seed + 7 * ep, n_sample)
        v = v - eta * g; E.append(q.true_energy(v, T))
    return np.array(E)


def main():
    rng0 = np.random.RandomState(7)
    STARTS = [rng0.uniform(-1.0, 1.0, q.NP) for _ in range(5)]   # shared starts
    eta, n_epochs = 0.1, 40

    # ── Part 1: free shots, noiseless, same starts ──
    clean = NoisyQuTiPRunner(2, noise=None)
    print(f"PART 1 — free shots (exact), noiseless.  Same 5 starts.  E0={q.E0:.4f}.")
    print(f"{'start':>7}{'PSR n=8':>11}{'FD ε=0.01':>11}{'|Δ|':>9}")
    for i, v0 in enumerate(STARTS):
        ep = descend("PSR", None, 8, v0, clean, eta, n_epochs, seed=1000 + i)[-1]
        ef = descend("FD", 0.01, 1, v0, clean, eta, n_epochs, seed=1000 + i)[-1]
        print(f"{i:>7}{ep:>11.4f}{ef:>11.4f}{abs(ep - ef):>9.4f}")

    # ── Part 2: noisy, equal budget, same starts × seeds ──
    noise = NoiseModel(n_qubits=2, T2=2.0)
    runner = NoisyQuTiPRunner(2, noise=noise)
    seeds = 3
    print(f"\nPART 2 — noisy (T2=2, faithful), EQUAL budget "
          f"(PSR n=4 @b50 vs FD ε=0.1 @b200).  Same {len(STARTS)} starts × "
          f"{seeds} seeds.")
    psr_all, fd_all = [], []
    for i, v0 in enumerate(STARTS):
        for s in range(seeds):
            sd = 2000 + 17 * i + s
            psr_all.append(descend("PSR", None, 4, v0, runner, eta, n_epochs,
                                   seed=sd, b_obs=50)[-1] - q.E0)
            fd_all.append(descend("FD", 0.1, 1, v0, runner, eta, n_epochs,
                                  seed=sd, b_obs=200)[-1] - q.E0)
    psr_all, fd_all = np.array(psr_all), np.array(fd_all)
    print(f"  PSR mean gap = {psr_all.mean():.4f} ± {psr_all.std():.4f}")
    print(f"  FD  mean gap = {fd_all.mean():.4f} ± {fd_all.std():.4f}")
    wins = int(np.sum(psr_all < fd_all))
    print(f"  PSR better on {wins}/{len(psr_all)} matched (start,seed) trials")
    print(f"\n(Same start+seed per trial → paired comparison; difference is method, "
          f"not setup.)")


if __name__ == "__main__":
    main()
