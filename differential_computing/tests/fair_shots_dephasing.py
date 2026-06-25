"""
fair_shots_dephasing.py — (1) give PSR more shots at EQUAL budget vs FD; (2) sweep
dephasing strength to find where PSR wins (the 'noisy real-device' regime).

Two questions:
  (1) PSR n_sample=1 carries τ-sampling variance.  Give PSR more τ-samples and
      compare to FD at a SIMILAR TOTAL shot budget (PSR n_sample=k at b_obs=B uses
      the same total as FD at b_obs=k·B).  Does PSR reach the correct answer?
  (2) Our clean sim may be too benign for FD (the paper's FD failure was on noisy
      IBM hardware).  Sweep the dephasing strength (T2 large→small) under the
      faithful model and see whether PSR overtakes FD as noise grows: FD's
      gradient SNR collapses (shrinking signal / fixed shot variance), while PSR's
      lower variance survives.

H2 VQE (2 qubits, fast).  Track the true (noiseless) energy → E0.

Run:  conda run -n qec_pg python differential_computing/tests/fair_shots_dephasing.py
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


def descend(method, eps, v0, T, runner, b_obs, eta, n_epochs, seed,
            n_sample=1):
    v = v0.copy(); E = [q.true_energy(v, T)]
    rng = np.random.default_rng(seed)
    for ep in range(n_epochs):
        if method == "FD":
            g = q.fd_grad(v, T, runner, b_obs, eps, rng)
        else:
            g = psr_grad(v, T, runner, b_obs, rng, seed + 7 * ep, n_sample)
        v = v - eta * g; E.append(q.true_energy(v, T))
    return np.array(E)


def final_gap(method, eps, runner, T, b_obs, n_sample, v0, eta=0.1,
              n_epochs=40, seeds=3):
    fe = []
    for s in range(seeds):
        E = descend(method, eps or 0.1, v0, T, runner, b_obs, eta, n_epochs,
                    seed=10 + s, n_sample=n_sample)
        fe.append(E[-1])
    return float(np.mean(fe)) - q.E0


def main():
    T = 1.0
    v0 = np.random.RandomState(3).uniform(-1.0, 1.0, q.NP)

    # ── Part 1: fair-budget PSR (more τ-samples) vs FD, at moderate dephasing ──
    print("PART 1 — equal total shots/gradient.  PSR n_sample=k @ b_obs=B vs "
          "FD @ b_obs=k·B.\nModerate dephasing T2=2 (faithful model). E0="
          f"{q.E0:.4f}.\n")
    B = 50
    noise2 = NoiseModel(n_qubits=2, T2=2.0)
    runner2 = NoisyQuTiPRunner(2, noise=noise2)
    print(f"{'config':>26}{'shots/grad':>12}{'final gap':>11}")
    fd_gap = final_gap("FD", 0.1, runner2, T, 4 * B, 1, v0)  # FD @ b_obs=4B
    print(f"{'FD ε=0.1 @ b_obs=' + str(4*B):>26}{2*q.NP*4*B:>12}{fd_gap:>11.4f}")
    for k in (1, 2, 4):
        g = final_gap("PSR", None, runner2, T, B, k, v0)     # PSR n_sample=k @ B
        print(f"{f'PSR n_sample={k} @ b_obs={B}':>26}{2*q.NP*k*B:>12}{g:>11.4f}")

    # ── Part 2: dephasing sweep — does PSR overtake FD as noise grows? ──
    print("\nPART 2 — dephasing sweep (faithful model), EQUAL budget: "
          "PSR n_sample=4 @ b_obs=50 vs FD ε=0.1 @ b_obs=200.\n")
    T2_list = [None, 5.0, 2.0, 1.0, 0.5]
    print(f"{'T2':>8}{'T/T2*':>8}{'PSR gap':>11}{'FD gap':>11}{'winner':>9}")
    psr_gaps, fd_gaps, xs = [], [], []
    for T2 in T2_list:
        noise = None if T2 is None else NoiseModel(n_qubits=2, T2=T2)
        runner = NoisyQuTiPRunner(2, noise=noise)
        pg = final_gap("PSR", None, runner, T, 50, 4, v0)
        fg = final_gap("FD", 0.1, runner, T, 200, 1, v0)
        psr_gaps.append(pg); fd_gaps.append(fg)
        xs.append(0.0 if T2 is None else T / T2)
        lbl = f"{T2:.1f}" if T2 else "  ∞"
        tt = "0.00" if T2 is None else f"{T/T2:.2f}"
        win = "PSR" if pg < fg else "FD"
        print(f"{lbl:>8}{tt:>8}{pg:>11.4f}{fg:>11.4f}{win:>9}")

    # plot gap vs dephasing strength
    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=150)
    ax.plot(xs, psr_gaps, "o-", color="#1f77b4", lw=2, label="PSR (n_sample=4)")
    ax.plot(xs, fd_gaps, "s-", color="#d62728", lw=2, label="FD (ε=0.1)")
    ax.set_xlabel(r"dephasing strength  $T/T_2^*$  (0 = clean)")
    ax.set_ylabel("final energy gap to $E_0$")
    ax.set_title("H$_2$ VQE: PSR vs FD at equal shot budget, vs dephasing\n"
                 "FD's SNR collapses as noise grows; PSR's lower variance survives")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "figures", "fair_shots_dephasing.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
