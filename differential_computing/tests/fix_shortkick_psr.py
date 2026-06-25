"""
fix_shortkick_psr.py — fix PSR's decoherence sensitivity by SHORTENING the kick.

Diagnosis (decompose_psr_error.py): PSR's residual gap under T2* is NOT the gate
error — it's the kick DURATION.  Algorithm 1 runs the +1 branch kick for 7π/4≈5.5,
longer than the whole evolution T=1, so that branch dephases ~6.5/T2 while the −1
branch dephases ~1.8/T2 → asymmetric corruption → biased gradient.

Fix (exact for Pauli generators, since exp(-iH_jθ) is 2π-periodic):
    exp(-i H_j · 7π/4) = exp(-i (-H_j) · π/4).
So replace the +1 branch kick [H_j, 7π/4] with [-H_j, π/4] — SAME unitary, 7×
shorter, and symmetric with the −1 branch (both π/4).  Noiselessly identical
gradient; under T2* far less dephasing.

We compare, on the H2 VQE under +T2 dephasing: standard-kick PSR vs short-kick
PSR vs FD, reporting the final gap to E0.  (Also a noiseless sanity check that the
two PSR variants give the same gradient.)

Run:  conda run -n qec_pg python differential_computing/tests/fix_shortkick_psr.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import vqe_noisy_comparison as q
from simuq.hamiltonian import productHamiltonian, TIHamiltonian
from combine_gradient import combine_gradient_results
from noisy_qutip import NoisyQuTiPRunner
from noise_model import NoiseModel


def shortkick_programs(parametrized_H, T, n_sample, diff_var, value):
    """Like observable_program_generator but with SHORT symmetric kicks.

    −1 branch: [Hj, π/4]   +1 branch: [-Hj, π/4]   (both short, symmetric).
    """
    u_grad_dict = parametrized_H.take_diff_coef(diff_var)
    evaluated_H = parametrized_H.set_parameterizedHam({diff_var: value})
    tau_list = np.random.rand(n_sample) * T
    s = np.pi / 4.0
    out = []
    for Hj_tuple, ugrad_raw in u_grad_dict.items():
        ug = float(ugrad_raw.subs(diff_var, value)) if hasattr(ugrad_raw, "subs") \
            else float(ugrad_raw)
        if ug == 0.0:
            continue
        prod = productHamiltonian(from_list=Hj_tuple)
        Hj = TIHamiltonian(parametrized_H.sites_type, parametrized_H.sites_name,
                           [(prod, 1)])
        Hj_neg = TIHamiltonian(parametrized_H.sites_type,
                               parametrized_H.sites_name, [(prod, -1)])
        H_tot = []
        for tau in tau_list:
            # −1 (p̃⁻): +Hj, π/4 ;  +1 (p̃⁺): −Hj, π/4  (unitary == 7π/4 of +Hj)
            H_tot.append([[evaluated_H, tau], [Hj, s], [evaluated_H, T - tau]])
            H_tot.append([[evaluated_H, tau], [Hj_neg, s], [evaluated_H, T - tau]])
        out.append([H_tot, ug, 1])
    return out


def psr_grad(v, T, runner, b_obs, rng, seed, short):
    g = np.zeros(q.NP)
    expfn = q.energy_expfn(runner, b_obs, rng)
    for k in range(q.NP):
        np.random.seed(seed + k)
        if short:
            programs = shortkick_programs(q.H_param_k(v, k), T, 1, "vk", float(v[k]))
        else:
            from observable_program_generator import observable_program_generator
            programs = observable_program_generator(
                q.H_param_k(v, k), T, n_sample=1, n_repetition=1,
                diff_var="vk", value=float(v[k]))
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


def main():
    # noiseless sanity: standard vs short-kick PSR give the same gradient
    clean = NoisyQuTiPRunner(2, noise=None)
    v = np.random.RandomState(1).uniform(-1, 1, q.NP)
    rng = np.random.default_rng(0)
    g_std = psr_grad(v, 1.0, clean, 100000, rng, 0, short=False)
    g_short = psr_grad(v, 1.0, clean, 100000, rng, 0, short=True)
    print(f"Noiseless sanity (≈exact): ||std − short|| = "
          f"{np.linalg.norm(g_std - g_short):.4f}  (should be ~0)\n")

    T2 = 5.0
    noise = NoiseModel(n_qubits=2, T2=T2)        # T2 only (the culprit channel)
    runner = NoisyQuTiPRunner(2, noise=noise)
    v0 = np.random.RandomState(3).uniform(-1.0, 1.0, q.NP)
    b_obs, eta, n_epochs, seeds = 100, 0.10, 40, 3

    print(f"H2 VQE under +T2 dephasing (T2={T2}).  E0={q.E0:.4f}, η={eta}, "
          f"{n_epochs} epochs, {seeds} seeds.")
    print(f"{'method':>22}{'final E':>11}{'gap to E0':>12}")
    configs = [("FD ε=0.1", "FD", 0.1, False),
               ("PSR standard kick", "PSR", None, False),
               ("PSR SHORT kick", "PSR", None, True)]
    for label, method, eps, short in configs:
        E = np.zeros((seeds, n_epochs + 1))
        for s in range(seeds):
            E[s] = descend(method, eps or 0.1, v0, T2 and 1.0 or 1.0, runner,
                           b_obs, eta, n_epochs, seed=10 + s, short=short)
        fe = E[:, -1].mean()
        print(f"{label:>22}{fe:>11.4f}{fe - q.E0:>12.4f}")

    print(f"\nIf SHORT-kick PSR's gap drops toward FD's, the long 7π/4 kick was "
          f"the cause —\nand shortening it (exact for Pauli kicks) makes PSR "
          f"decoherence-robust.")


if __name__ == "__main__":
    main()
