"""
vqe_lr_check.py — is PSR's slower convergence under decoherence fundamental, or
just a learning-rate effect from its attenuated gradient?

Under decoherence PSR's gradient is attenuated (λ<1), acting like a smaller
effective learning rate → it descended slower than tuned FD in vqe_noisy_comparison.
The attenuation is a constant rescaling, so it should be absorbable into η.  Here
we give EACH method its own η sweep on the H2+decoherence landscape and compare
best-vs-best convergence speed.

Run:  conda run -n qec_pg python differential_computing/tests/vqe_lr_check.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import vqe_noisy_comparison as q
from noise_model import NoiseModel


def converge_epoch(E, thresh):
    reached = np.where(E.mean(0) <= thresh)[0]
    return int(reached[0]) if len(reached) else -1


def main():
    T = 1.0
    noise = NoiseModel(n_qubits=2, T2=5.0, gate_error_1q=1e-4,
                       gate_error_2q=1e-3, gate_coherent_frac=0.5)
    runner = q.NoisyQuTiPRunner(2, noise=noise)
    v0 = np.random.RandomState(3).uniform(-1.0, 1.0, q.NP)
    b_obs, n_epochs, seeds = 100, 40, 3
    thresh = q.E0 + 0.05

    print(f"H2 + decoherence (T=1, T2=5 + gate err).  Each method swept over η.")
    print(f"E0={q.E0:.4f}, threshold = E0+0.05 = {thresh:.4f}, {n_epochs} epochs, "
          f"{seeds} seeds, b_obs={b_obs}.\n")
    print(f"{'method':>12}{'η':>7}{'final E':>11}{'gap':>9}{'epoch≤thr':>11}")

    configs = [("PSR", None, e) for e in (0.10, 0.15, 0.20, 0.30)] \
        + [("FD", 0.1, e) for e in (0.05, 0.10, 0.15, 0.20)]
    best = {}
    for method, eps, eta in configs:
        E = np.zeros((seeds, n_epochs + 1))
        for s in range(seeds):
            E[s] = q.descend(method, eps or 0.1, v0, T, runner, b_obs, eta,
                             n_epochs, seed=10 + s)
        fe = E[:, -1].mean()
        ep = converge_epoch(E, thresh)
        label = "PSR" if method == "PSR" else f"FD ε={eps}"
        tag = f"{ep}" if ep >= 0 else "never"
        print(f"{label:>12}{eta:>7.2f}{fe:>11.4f}{fe - q.E0:>9.4f}{tag:>11}")
        # track best (earliest convergence; tie-break on final E)
        score = (ep if ep >= 0 else 999, fe)
        if label not in best or score < best[label][0]:
            best[label] = (score, eta)

    print(f"\nBest per method (earliest convergence):")
    for label, ((ep, fe), eta) in best.items():
        tag = f"epoch {ep}" if ep < 999 else "never"
        print(f"  {label:>10}: η={eta:.2f}  →  {tag}, final {fe:.4f}")
    print(f"\nIf PSR's best η reaches the threshold as fast as FD's best η, the "
          f"slowness was\njust the attenuation rescaling (fixable by η) — and PSR "
          f"still needs no ε.")


if __name__ == "__main__":
    main()
