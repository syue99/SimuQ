"""
decompose_psr_error.py — why can't PSR converge to the true ground under noise?

Candidates for PSR's residual gap (vs FD's smaller gap) on the H2 VQE:
  (a) kick GATE error — PSR inserts a kick gate each branch; imperfect gates bias it.
  (b) T2* dephasing variance/bias.
  (c) KICK DURATION asymmetry — Algorithm 1's kick runs for (1±3/4)π, i.e. π/4≈0.785
      AND 7π/4≈5.5.  The 7π/4 branch is LONGER than the evolution T=1, so under T2*
      it dephases ~7× more than the π/4 branch → asymmetric corruption → biased
      gradient.  FD has no kicks, so it never pays this.

We toggle noise sources independently and read PSR's final gap vs FD's:
  shot-only | +gate (perfect T2) | +T2 (perfect kicks) | +both.
If PSR's gap stays small with T2 OFF but blows up with T2 ON → it's dephasing
(candidate b/c), not the gate (a).  Then we test candidate (c) directly by also
reporting the per-branch kick durations and the mean evolution time PSR vs FD see.

Run:  conda run -n qec_pg python differential_computing/tests/decompose_psr_error.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import vqe_noisy_comparison as q
from noisy_qutip import NoisyQuTiPRunner
from noise_model import NoiseModel


def run(noise, T=1.0, b_obs=100, eta=0.10, n_epochs=40, seeds=3):
    runner = NoisyQuTiPRunner(2, noise=noise)
    v0 = np.random.RandomState(3).uniform(-1.0, 1.0, q.NP)
    out = {}
    for method, eps in (("PSR", None), ("FD", 0.1)):
        E = np.zeros((seeds, n_epochs + 1))
        for s in range(seeds):
            E[s] = q.descend(method, eps or 0.1, v0, T, runner, b_obs, eta,
                             n_epochs, seed=10 + s)
        out[method] = (E[:, -1].mean(), E[:, -1].std())
    return out


def main():
    T2 = 5.0
    GE = dict(gate_error_1q=1e-4, gate_error_2q=1e-3, gate_coherent_frac=0.5)
    conditions = [
        ("shot only",            None),
        ("+gate (perfect T2)",   NoiseModel(n_qubits=2, **GE)),
        ("+T2 (perfect kicks)",  NoiseModel(n_qubits=2, T2=T2)),
        ("+both",                NoiseModel(n_qubits=2, T2=T2, **GE)),
    ]

    print(f"H2 VQE — decomposing PSR's residual gap.  E0={q.E0:.4f}, T=1, "
          f"b_obs=100, 40 epochs, 3 seeds.\n")
    print(f"{'noise condition':>22} | {'PSR final gap':>16} | {'FD final gap':>16}"
          f" | {'PSR−FD':>8}")
    for label, noise in conditions:
        r = run(noise)
        pgap = r["PSR"][0] - q.E0
        fgap = r["FD"][0] - q.E0
        print(f"{label:>22} | {pgap:>9.4f}±{r['PSR'][1]:>5.3f} | "
              f"{fgap:>9.4f}±{r['FD'][1]:>5.3f} | {pgap - fgap:>+8.4f}")

    # candidate (c): how much evolution time does each estimator actually accrue?
    from observable_program_generator import observable_program_generator
    np.random.seed(0)
    progs = observable_program_generator(
        q.H_param_k(np.zeros(q.NP), 0), 1.0, n_sample=1, n_repetition=1,
        diff_var="vk", value=0.0)
    durs = []
    for H_tot, _, _ in progs:
        for H_list in H_tot:
            durs.append([d for _, d in H_list])     # [tau, kick, T-tau]
    print(f"\nKick-duration check (T=1): a PSR branch is [τ, kick, T−τ].")
    for d in durs:
        total = sum(d)
        print(f"  segments={np.round(d,3)}  total evolution time={total:.3f} "
              f"(kick={d[1]:.3f})")
    print(f"  FD sees only T=1.0 per evaluation (no kick).")
    print(f"\nReading: if PSR−FD gap is ~0 with T2 OFF and grows with T2 ON, the "
          f"residual is\ndephasing — and the long 7π/4≈5.5 kick (dephasing ~5.5/T2) "
          f"is the prime suspect,\nnot the gate error (toggle +gate to confirm it's "
          f"small).")


if __name__ == "__main__":
    main()
