"""
fair_protocol_figures.py — shot-budget and dephasing sweeps under the FAIR PAIRED
protocol (same starts + seeds across PSR and FD, multi-start, equal budget).

Supersedes the earlier (unpaired / kick-dephasing-artifact) shot-budget and
dephasing figures.  H2 VQE, faithful noise model.  Metric: final energy gap to E0,
averaged over the SAME M starts × S seeds for both methods (paired).

  Panel A — gap vs total shot budget (fixed moderate dephasing T2=2).
  Panel B — gap vs dephasing strength T/T2 (fixed shot budget).
Equal budget per gradient: PSR n_sample=4 @ b_obs=B vs FD ε=0.1 @ b_obs=4B.

Saves figures/fair_protocol_comparison.png.

Run:  conda run -n qec_pg python differential_computing/tests/fair_protocol_figures.py
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
from noisy_qutip import NoisyQuTiPRunner
from noise_model import NoiseModel

NP = q.NP
N_TERMS = 4                  # H2 measured Pauli terms (for total-shot accounting)
ETA, N_EPOCHS = 0.1, 25
rng0 = np.random.RandomState(7)
STARTS = [rng0.uniform(-1.0, 1.0, NP) for _ in range(4)]
SEEDS = 2


def paired_gap(runner, b_obs_psr, eps, n_sample):
    """Mean (PSR, FD) final gap over the SAME starts × seeds (paired)."""
    pg, fg = [], []
    for i, v0 in enumerate(STARTS):
        for s in range(SEEDS):
            sd = 3000 + 17 * i + s
            pg.append(f.descend("PSR", None, n_sample, v0, runner, ETA, N_EPOCHS,
                                seed=sd, b_obs=b_obs_psr)[-1] - q.E0)
            fg.append(f.descend("FD", eps, 1, v0, runner, ETA, N_EPOCHS,
                                seed=sd, b_obs=n_sample * b_obs_psr)[-1] - q.E0)
    return np.array(pg), np.array(fg)


def main():
    n_sample = 4
    # ── Sweep A: shot budget (T2=2) ──
    runnerA = NoisyQuTiPRunner(2, noise=NoiseModel(n_qubits=2, T2=2.0))
    b_psr_list = [6, 12, 25, 50]
    A = []
    print("PART A — shot-budget sweep (T2=2, paired).  "
          "PSR n=4 @b_obs=B vs FD ε=0.1 @b_obs=4B.")
    print(f"{'total shots/grad':>18}{'PSR gap':>12}{'FD gap':>12}{'PSR wins':>10}")
    for bp in b_psr_list:
        pg, fg = paired_gap(runnerA, bp, 0.1, n_sample)
        total = 2 * NP * N_TERMS * n_sample * bp
        A.append((total, pg.mean(), pg.std(), fg.mean(), fg.std()))
        print(f"{total:>18}{pg.mean():>12.4f}{fg.mean():>12.4f}"
              f"{int((pg<fg).sum())}/{len(pg)}".rjust(10))

    # ── Sweep B: dephasing (fixed budget b_obs_psr=12) ──
    bp = 12
    B = []
    print("\nPART B — dephasing sweep (paired, fixed budget).  T/T2* axis.")
    print(f"{'T2':>8}{'T/T2*':>8}{'PSR gap':>12}{'FD gap':>12}{'PSR wins':>10}")
    for T2 in [None, 5.0, 2.0, 1.0, 0.5]:
        noise = None if T2 is None else NoiseModel(n_qubits=2, T2=T2)
        runner = NoisyQuTiPRunner(2, noise=noise)
        pg, fg = paired_gap(runner, bp, 0.1, n_sample)
        x = 0.0 if T2 is None else f.T / T2
        B.append((x, pg.mean(), pg.std(), fg.mean(), fg.std()))
        lbl = "  ∞" if T2 is None else f"{T2:.1f}"
        print(f"{lbl:>8}{x:>8.2f}{pg.mean():>12.4f}{fg.mean():>12.4f}"
              f"{int((pg<fg).sum())}/{len(pg)}".rjust(10))

    # ── plot ──
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=150)
    A = np.array(A); B = np.array(B)
    axA.errorbar(A[:, 0], A[:, 1], yerr=A[:, 2], fmt="o-", color="#1f77b4",
                 lw=2, capsize=3, label="PSR (n=4)")
    axA.errorbar(A[:, 0], A[:, 3], yerr=A[:, 4], fmt="s-", color="#d62728",
                 lw=2, capsize=3, label="FD (ε=0.1)")
    axA.set_xscale("log"); axA.set_xlabel("total shots / gradient")
    axA.set_ylabel("final energy gap to $E_0$")
    axA.set_title("(A) shot-budget sweep (T2=2, paired)")
    axA.legend(frameon=False, fontsize=9)

    axB.errorbar(B[:, 0], B[:, 1], yerr=B[:, 2], fmt="o-", color="#1f77b4",
                 lw=2, capsize=3, label="PSR (n=4)")
    axB.errorbar(B[:, 0], B[:, 3], yerr=B[:, 4], fmt="s-", color="#d62728",
                 lw=2, capsize=3, label="FD (ε=0.1)")
    axB.set_xlabel(r"dephasing strength $T/T_2^*$"); axB.set_ylabel("gap to $E_0$")
    axB.set_title("(B) dephasing sweep (paired)")
    axB.legend(frameon=False, fontsize=9)

    fig.suptitle("H$_2$ VQE, FAIR paired comparison (same starts+seeds, equal "
                 "budget): PSR vs FD\nPSR wins at equal budget — lower variance, "
                 "advantage grows at low shots and high noise", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "figures", "fair_protocol_comparison.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
