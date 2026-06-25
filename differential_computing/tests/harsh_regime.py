"""
harsh_regime.py — reproduce a dramatic FD failure in a noisy, low-shot regime.

The fair-budget dephasing sweep showed PSR overtakes FD as noise grows, but only
modestly — our clean dephasing-only sim is benign vs the paper's noisy IBM
hardware.  Here we push to a harsh regime (strong dephasing T2=1 + gate error +
VERY low shots b_obs) where variance dominates: FD's shot noise amplified by its ε
should make it fail at every ε, while PSR (no ε, lower variance) still descends.

H2 VQE, faithful model, EQUAL total shot budget: PSR n_sample=4 @ b_obs=B vs FD
swept over ε @ b_obs=4B.  Track the true energy → E0; plot energy vs epoch.

Run:  conda run -n qec_pg python differential_computing/tests/harsh_regime.py
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
from observable_program_generator import observable_program_generator
from combine_gradient import combine_gradient_results
from noisy_qutip import NoisyQuTiPRunner
from noise_model import NoiseModel


def psr_grad(v, T, runner, b_obs, rng, seed, n_sample):
    g = np.zeros(q.NP)
    expfn = q.energy_expfn(runner, b_obs, rng)
    for k in range(q.NP):
        np.random.seed(seed + k)
        progs = observable_program_generator(
            q.H_param_k(v, k), T, n_sample=n_sample, n_repetition=1,
            diff_var="vk", value=float(v[k]))
        g[k] = combine_gradient_results(progs, expfn, T)
    return g


def descend(method, eps, v0, T, runner, b_obs, eta, n_epochs, seed, n_sample=1):
    v = v0.copy(); E = [q.true_energy(v, T)]
    rng = np.random.default_rng(seed)
    for ep in range(n_epochs):
        if method == "FD":
            g = q.fd_grad(v, T, runner, b_obs, eps, rng)
        else:
            g = psr_grad(v, T, runner, b_obs, rng, seed + 7 * ep, n_sample)
        v = v - eta * g; E.append(q.true_energy(v, T))
    return np.array(E)


def main():
    T = 1.0
    B = 8                          # VERY low shots (PSR per eval); FD gets 4B=32
    noise = NoiseModel(n_qubits=2, T2=1.0, gate_error_1q=1e-4, gate_error_2q=1e-3,
                       gate_coherent_frac=0.5)
    runner = NoisyQuTiPRunner(2, noise=noise)         # faithful model
    v0 = np.random.RandomState(3).uniform(-1.0, 1.0, q.NP)
    eta, n_epochs, seeds = 0.10, 50, 5
    fd_eps = [0.05, 0.1, 0.3, 0.6]

    runs = [(f"PSR n=4 @b{B}", "PSR", None)] + \
           [(f"FD ε={e} @b{4*B}", "FD", e) for e in fd_eps]
    res = {}
    for label, method, eps in runs:
        E = np.zeros((seeds, n_epochs + 1))
        b_obs = B if method == "PSR" else 4 * B
        for s in range(seeds):
            E[s] = descend(method, eps or 0.1, v0, T, runner, b_obs, eta,
                           n_epochs, seed=10 + s, n_sample=4)
        res[label] = E

    print(f"HARSH regime — H2 VQE, faithful model.  T2=1 + gate err, equal budget "
          f"(PSR n=4 @b_obs={B} vs FD @b_obs={4*B}).\nE0={q.E0:.4f}, η={eta}, "
          f"{n_epochs} epochs, {seeds} seeds.\n")
    print(f"{'method':>16}{'final E':>11}{'gap to E0':>12}{'best reached':>14}")
    for label, E in res.items():
        fe = E[:, -1].mean()
        print(f"{label:>16}{fe:>11.4f}{fe - q.E0:>12.4f}{E.mean(0).min():>14.4f}")

    steps = np.arange(n_epochs + 1)
    fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=150)
    fdc = ["#d62728", "#ff7f0e", "#9467bd", "#8c564b"]
    for i, (label, E) in enumerate(res.items()):
        mu, sd = E.mean(0), E.std(0)
        c = "#1f77b4" if label.startswith("PSR") else fdc[(i - 1) % len(fdc)]
        lw = 2.6 if label.startswith("PSR") else 1.7
        ax.plot(steps, mu, label=label, color=c, lw=lw)
        ax.fill_between(steps, mu - sd, mu + sd, color=c, alpha=0.12)
    ax.axhline(q.E0, ls="--", color="k", lw=1, label=f"$E_0$={q.E0:.3f}")
    ax.set_xlabel("epoch"); ax.set_ylabel(r"energy $\langle H_{\mathrm{H_2}}\rangle$")
    ax.set_title("H$_2$ VQE, harsh regime (strong dephasing + very low shots):\n"
                 "PSR descends; FD at every ε is variance-wrecked")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "figures", "harsh_regime.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
