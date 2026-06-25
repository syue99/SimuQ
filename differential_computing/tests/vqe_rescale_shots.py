"""
vqe_rescale_shots.py — does rescaling PSR (1/λ) + more τ-samples make it match FD?

Two follow-up questions on the H2 VQE under decoherence:
  (1) PSR's gradient is attenuated by λ<1; rescale by 1/λ so its MEAN matches FD.
      Since the sign is (mostly) correct, does PSR then converge as fast as FD?
      Catch: rescaling amplifies the NOISE by 1/λ too — it fixes the mean, not the
      variance.  So this only helps if PSR's variance is already ≤ FD's.
  (2) PSR has two variance sources — shot noise (b_obs) and τ-sampling (n_sample).
      More τ-samples lowers variance → PSR should catch up, but at n_sample× the
      per-gradient circuit cost.  We report epochs-to-threshold AND cost-to-
      threshold (epochs × per-gradient evaluations) for the fair comparison.

λ is auto-estimated at the start (||PSR_exact|| / ||FD_exact|| under decoherence).

Run:  conda run -n qec_pg python differential_computing/tests/vqe_rescale_shots.py
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


def psr_grad_ns(v, T, runner, b_obs, rng, seed, n_sample, rescale):
    g = np.zeros(q.NP)
    expfn = q.energy_expfn(runner, b_obs, rng)
    for k in range(q.NP):
        np.random.seed(seed + k)
        programs = observable_program_generator(
            q.H_param_k(v, k), T, n_sample=n_sample, n_repetition=1,
            diff_var="vk", value=float(v[k]))
        g[k] = combine_gradient_results(programs, expfn, T)
    return rescale * g


def descend(method, eps, v0, T, runner, b_obs, eta, n_epochs, seed,
            n_sample=1, rescale=1.0):
    v = v0.copy()
    E = [q.true_energy(v, T)]
    rng = np.random.default_rng(seed)
    for ep in range(n_epochs):
        if method == "PSR":
            g = psr_grad_ns(v, T, runner, b_obs, rng, seed + 7 * ep, n_sample, rescale)
        else:
            g = q.fd_grad(v, T, runner, b_obs, eps, rng)
        v = v - eta * g
        E.append(q.true_energy(v, T))
    return np.array(E)


def estimate_lambda(v0, T, runner):
    """λ ≈ ||PSR_exact|| / ||FD_exact|| under decoherence (no shots, n_sample large)."""
    expfn_exact = lambda Hl: float(np.real(
        __import__("qutip").expect(q.H_H2, runner.run_sequence(Hl, q.PSI0))))
    gp = np.zeros(q.NP); gf = np.zeros(q.NP)
    for k in range(q.NP):
        np.random.seed(1)
        progs = observable_program_generator(
            q.H_param_k(v0, k), T, n_sample=40, n_repetition=1,
            diff_var="vk", value=float(v0[k]))
        gp[k] = combine_gradient_results(progs, expfn_exact, T)
        vp = v0.copy(); vp[k] += 1e-2
        vm = v0.copy(); vm[k] -= 1e-2
        gf[k] = (expfn_exact([[q.H_of_v(vp), T]]) - expfn_exact([[q.H_of_v(vm), T]])) / 2e-2
    return float(np.linalg.norm(gp) / (np.linalg.norm(gf) + 1e-12))


def main():
    T = 1.0
    noise = NoiseModel(n_qubits=2, T2=5.0, gate_error_1q=1e-4,
                       gate_error_2q=1e-3, gate_coherent_frac=0.5)
    runner = NoisyQuTiPRunner(2, noise=noise)
    v0 = np.random.RandomState(3).uniform(-1.0, 1.0, q.NP)
    b_obs, eta, n_epochs, seeds = 100, 0.10, 40, 3
    thresh = q.E0 + 0.05

    lam = estimate_lambda(v0, T, runner)
    rescale = 1.0 / lam
    print(f"H2 + decoherence (T=1, T2=5 + gate err).  Estimated λ={lam:.3f} → "
          f"PSR rescaled by 1/λ={rescale:.3f}.")
    print(f"E0={q.E0:.4f}, threshold={thresh:.4f}, η={eta}, {n_epochs} epochs, "
          f"{seeds} seeds, b_obs={b_obs}.\n")
    print(f"{'method':>18}{'cost/grad':>11}{'final E':>10}{'gap':>8}"
          f"{'epoch≤thr':>11}{'cost≤thr':>11}")

    def report(label, E, cost_per_grad):
        fe = E[:, -1].mean()
        reached = np.where(E.mean(0) <= thresh)[0]
        ep = int(reached[0]) if len(reached) else -1
        ep_tag = f"{ep}" if ep >= 0 else "never"
        cost_tag = f"{ep * cost_per_grad}" if ep >= 0 else "—"
        print(f"{label:>18}{cost_per_grad:>11}{fe:>10.4f}{fe - q.E0:>8.4f}"
              f"{ep_tag:>11}{cost_tag:>11}")

    # FD tuned ε (cost = 2 evals per param)
    E = np.zeros((seeds, n_epochs + 1))
    for s in range(seeds):
        E[s] = descend("FD", 0.1, v0, T, runner, b_obs, eta, n_epochs, seed=10 + s)
    report("FD ε=0.1", E, 2 * q.NP)

    # PSR rescaled, sweep n_sample (cost = 2·n_sample per param)
    for ns in (1, 2, 4):
        E = np.zeros((seeds, n_epochs + 1))
        for s in range(seeds):
            E[s] = descend("PSR", None, v0, T, runner, b_obs, eta, n_epochs,
                           seed=10 + s, n_sample=ns, rescale=rescale)
        report(f"PSR×1/λ n_samp={ns}", E, 2 * ns * q.NP)

    print(f"\nReading: 'cost/grad' = energy evaluations per gradient (PSR scales "
          f"with n_sample).\n'cost≤thr' = epochs×cost to reach the threshold "
          f"(the FAIR budget metric).  Rescale fixes\nPSR's magnitude; n_sample "
          f"lowers its variance — see if PSR catches FD on epochs, and\nat what "
          f"total cost.")


if __name__ == "__main__":
    main()
