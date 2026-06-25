"""
kick_dephasing_correction.py — was PSR's "decoherence problem" a modeling artifact?

User's correction: the PSR kick is compiled to a single/two-qubit GATE (clock-
state rotation / Rydberg gate), NOT a dressed analog evolution.  So the dressing-
level T2* should NOT act during the kick — only the (separately modeled) gate
error.  My earlier model applied T2* dephasing to ALL segments including the kick,
which over-penalized PSR and made the long 7π/4 kick look catastrophic.

This re-runs the H2 VQE under T2 dephasing in two models:
  - kick_dephases=True  (my earlier, conservative model: T2 everywhere)
  - kick_dephases=False (physically faithful: T2 only on the dressed evolution)
for standard-kick PSR, short-kick PSR, and FD.  If PSR's gap shrinks with
kick_dephases=False, the "decoherence problem" was largely the mis-modeling — and
the short-kick benefit should also shrink (the kick no longer dephases).

Run:  conda run -n qec_pg python differential_computing/tests/kick_dephasing_correction.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import vqe_noisy_comparison as q
from observable_program_generator import observable_program_generator
from combine_gradient import combine_gradient_results
from noisy_qutip import NoisyQuTiPRunner
from noise_model import NoiseModel


def psr_grad(v, T, runner, b_obs, rng, seed, short):
    g = np.zeros(q.NP)
    expfn = q.energy_expfn(runner, b_obs, rng)
    for k in range(q.NP):
        np.random.seed(seed + k)
        programs = observable_program_generator(
            q.H_param_k(v, k), T, n_sample=1, n_repetition=1,
            diff_var="vk", value=float(v[k]), short_kick=short)
        g[k] = combine_gradient_results(programs, expfn, T)
    return g


def descend(method, eps, v0, T, runner, b_obs, eta, n_epochs, seed, short=False):
    v = v0.copy()
    E = [q.true_energy(v, T)]
    rng = np.random.default_rng(seed)
    for ep in range(n_epochs):
        if method == "FD":
            g = q.fd_grad(v, T, runner, b_obs, eps, rng)
        else:
            g = psr_grad(v, T, runner, b_obs, rng, seed + 7 * ep, short)
        v = v - eta * g
        E.append(q.true_energy(v, T))
    return np.array(E)


def run(kick_dephases):
    T2 = 5.0
    noise = NoiseModel(n_qubits=2, T2=T2)
    runner = NoisyQuTiPRunner(2, noise=noise, kick_dephases=kick_dephases)
    v0 = np.random.RandomState(3).uniform(-1.0, 1.0, q.NP)
    b_obs, eta, n_epochs, seeds = 100, 0.10, 40, 3
    thresh = q.E0 + 0.05
    out = {}
    for label, method, short in [("FD ε=0.1", "FD", False),
                                 ("PSR standard", "PSR", False),
                                 ("PSR short", "PSR", True)]:
        E = np.zeros((seeds, n_epochs + 1))
        for s in range(seeds):
            E[s] = descend(method, 0.1, v0, 1.0, runner, b_obs, eta, n_epochs,
                           seed=10 + s, short=short)
        reached = np.where(E.mean(0) <= thresh)[0]
        ep = int(reached[0]) if len(reached) else -1
        out[label] = (E[:, -1].mean() - q.E0, ep)
    return out


def main():
    print(f"H2 VQE under T2=5 dephasing.  E0={q.E0:.4f}.  gap(epoch≤E0+0.05).\n")
    print(f"{'model':>33}{'FD ε=0.1':>14}{'PSR standard':>16}{'PSR short':>14}")
    for kd, label in [(True, "kick_dephases=True (T2 on kick)"),
                      (False, "kick_dephases=False (kick=gate)")]:
        r = run(kd)
        def cell(x):
            ep = f"{x[1]}" if x[1] >= 0 else "never"
            return f"{x[0]:.4f}({ep})"
        print(f"{label:>33}{cell(r['FD ε=0.1']):>14}{cell(r['PSR standard']):>16}"
              f"{cell(r['PSR short']):>14}")

    print(f"\nReading: kick_dephases=False is the physically faithful model (kick "
          f"is a gate,\nnot dressed → no dressing-T2*).  If PSR-standard's gap "
          f"shrinks there, the earlier\n'decoherence problem' was mostly the "
          f"mis-modeling; and short vs standard should\nconverge (the kick no "
          f"longer dephases, so its duration stops mattering).")


if __name__ == "__main__":
    main()
