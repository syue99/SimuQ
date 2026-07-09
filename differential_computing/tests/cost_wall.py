"""
cost_wall.py (Fig 8) — the toolchain runs where the simulator cannot.

Wall-clock cost vs n:
  1. EXACT NOISY SIMULATION of the compiled gradient program: measured
     per-branch mesolve time (3-segment PSR branch, density matrix, T/T2*=0.15
     conventions) × branch count 2·M·n_sample (C7 convention n_sample=100),
     n = 2..7 measured; exponential fit dashed beyond; intractable region shaded.
  2. COMPILATION: measured (compile_scaling_data.json cache, n to 12).
  3. O(1) CORRECTION: the light-cone slope on the m=5 subsystem — timed, flat,
     independent of n (plotted at nominal n out to 100).

Prose number for §5.4: the n where full-gradient simulation crosses 1 hour.

Run:  conda run -n qec_pg python differential_computing/tests/cost_wall.py
(run on a quiet machine — this is a timing benchmark)
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import qutip as qp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analytic_rescale as ar

T, T2 = 1.5, 10.0           # T/T2* = 0.15 headline conventions
N_SAMPLE = 100              # C7 convention → branches = 2·M·n_sample
NS_SIM = [2, 3, 4, 5, 6, 7]
REPS = 3                    # timing repetitions (median)

I2 = qp.qeye(2); X = qp.sigmax(); Z = qp.sigmaz()


def emb(op, i, n):
    return qp.tensor([op if k == i else I2 for k in range(n)])


def chain_H(th, n):
    H = 0
    for i in range(n - 1):
        H = H + th * emb(Z, i, n) * emb(Z, i + 1, n)
    for i in range(n):
        H = H + emb(X, i, n)
    return H


def time_branch_sim(n):
    """Median wall-clock of ONE noisy PSR branch (3-segment mesolve chain)."""
    H = chain_H(0.5, n)
    Hj = emb(Z, 0, n) * emb(Z, 1, n)
    psi0 = qp.tensor([qp.basis(2, 0)] * n)
    rho = psi0 * psi0.dag()
    c_ops = [np.sqrt(1.0 / (2 * T2)) * emb(Z, i, n) for i in range(n)]
    obs = emb(Z, 0, n) * emb(Z, 1, n)
    segs = [(H, T / 2), (Hj, np.pi / 4), (H, T / 2)]   # kick unitary, mid split
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        r = rho
        for k, (Hk, d) in enumerate(segs):
            if k == 1:   # kick = noiseless gate
                U = (-1j * Hk * d).expm()
                r = U * r * U.dag()
            else:
                r = qp.mesolve(Hk, r, [0.0, float(d)], c_ops=c_ops).states[-1]
        float(qp.expect(obs, r).real)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def time_correction():
    """Median wall-clock of the m=5 light-cone slope (n-independent)."""
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        ar.chain_slope(0.5, T, radius=4, n_grid=120)   # m = 5 subsystem
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    figdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    os.makedirs(figdir, exist_ok=True)
    cache = os.path.join(figdir, "cost_wall_data.json")

    if os.path.exists(cache):
        d = json.load(open(cache))
        print("loaded cache — replotting only")
    else:
        sim = []
        for n in NS_SIM:
            tb = time_branch_sim(n)
            M = n - 1
            branches = 2 * M * N_SAMPLE
            sim.append(dict(n=n, t_branch=tb, branches=branches,
                            t_gradient=tb * branches))
            print(f"  n={n}: branch {tb:.4f}s × {branches} branches "
                  f"= {tb * branches:.1f}s / gradient", flush=True)
        t_corr = time_correction()
        print(f"  correction (m=5 subsystem): {t_corr:.3f}s at ANY n")
        compile_rows = json.load(open(os.path.join(figdir,
                                                   "compile_scaling_data.json")))
        d = dict(T=T, T2=T2, n_sample=N_SAMPLE, sim=sim, t_corr=t_corr,
                 compile_rows=compile_rows)
        json.dump(d, open(cache, "w"), default=float)
        print(f"cached: {cache}")

    sim = d["sim"]
    ns = np.array([r["n"] for r in sim])
    tg = np.array([r["t_gradient"] for r in sim])
    # exponential fit on the density-matrix branch cost (dominates)
    coef = np.polyfit(ns[2:], np.log(tg[2:]), 1)      # slope per qubit
    n_ext = np.arange(ns[-1], 21)
    tg_ext = np.exp(coef[1] + coef[0] * n_ext)
    n_hour = (np.log(3600) - coef[1]) / coef[0]
    print(f"fit: gradient-sim time × {np.exp(coef[0]):.1f} per qubit; "
          f"crosses 1 hour at n ≈ {n_hour:.1f}")

    comp = [r for r in d["compile_rows"] if r["status"] == "ok"]
    cn = [r["n"] for r in comp]; ct = [r["compile_s"] for r in comp]

    fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=150)
    ax.semilogy(ns, tg, "o-", color="#d62728", lw=2.2,
                label="exact noisy simulation of the gradient program (measured)")
    ax.semilogy(n_ext, tg_ext, "--", color="#d62728", lw=1.4, alpha=0.7,
                label="exponential fit (extrapolated)")
    ax.semilogy(cn, ct, "s-", color="#7b1fa2", lw=2,
                label="compilation (measured, n to 12)")
    corr_ns = [2, 4, 7, 12, 20, 50, 100]
    ax.semilogy(corr_ns, [d["t_corr"]] * len(corr_ns), "^", color="#1f77b4",
                ms=7, ls=":", lw=1.4,
                label=f"O(1) correction, m=5 subsystem ({d['t_corr']:.2f}s at any n)")
    ax.axhline(3600, color="#888", lw=1, ls="--")
    ax.text(2.2, 4300, "1 hour", fontsize=8, color="#666")
    ax.axvspan(n_hour, 21, color="#f5f5f5", zorder=0)
    ax.text((n_hour + 21) / 2, tg[0], "exact simulation\nintractable",
            ha="center", fontsize=9, color="#999")
    ax.set_xlabel("qubits n")
    ax.set_ylabel("wall-clock (s, log)")
    ax.set_xlim(1.5, 21)
    ax.set_title("The cost wall: simulation of the gradient program is exponential;\n"
                 "compilation and the correction are not — the toolchain runs where "
                 "the simulator cannot")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    out = os.path.join(figdir, "cost_wall.png")
    fig.savefig(out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
