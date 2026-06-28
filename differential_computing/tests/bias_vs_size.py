"""
bias_vs_size.py — BIAS only (exact expectations, variance set aside): does PSR-
rescaled stay unbiased as the system grows, while FD's attenuation bias grows?

For TFIM chains n=2..6 (θ·ΣZ_iZ_{i+1}+ΣX_i, M=n-1 chain-rule terms, obs <Z0Z1>),
at fixed moderate dephasing, compute the BIAS = |estimate − ideal gradient| with
INFINITE shots (exact expectations):
  - FD best-ε → ε→0 → the NOISY-landscape gradient = λ'·g_ideal; bias=(1−λ')|g|
    (the ATTENUATION — uncorrectable by tuning ε; this is FD's BEST possible, with
    infinite shots and perfect step).
  - PSR raw → λ_PSR·g_ideal; bias=(1−λ_PSR)|g|.
  - PSR rescaled → g_ideal; bias≈ small residual (rescale removes the attenuation).

KEY: bias is INDEPENDENT of the chain-rule M (the rescale corrects the TOTAL
gradient) — so many terms don't hurt the bias, only the variance (set aside here).
The attenuation grows with system size (more dephasing channels) → FD's best-case
bias GROWS with n; PSR-rescaled stays near zero.  Argument: even for a large many-
term system, PSR-rescaled reaches the true gradient where FD cannot (and FD is then
made even worse by its 1/ε shot variance).

Run:  conda run -n qec_pg python differential_computing/tests/bias_vs_size.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sympy as sp
import qutip as qp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simuq import QSystem, Qubit
import analytic_rescale as ar
from observable_program_generator import observable_program_generator
from noisy_qutip import NoisyQuTiPRunner
from noise_model import NoiseModel

T, T2 = 1.5, 20.0
POOL = 150
I = qp.qeye(2); X, Z = qp.sigmax(), qp.sigmaz()


def emb(op, i, n):
    return qp.tensor([op if k == i else I for k in range(n)])


def zz(i, j, n): return emb(Z, i, n) * emb(Z, j, n)


def tfim(n):
    def sq():
        x = sp.Symbol("x"); qs = QSystem(); q = [Qubit(qs) for _ in range(n)]
        H = x * sum((q[i].Z*q[i+1].Z for i in range(n-1)), 0*q[0].Z)
        for i in range(n): H = H + q[i].X
        return H, "x"
    def hq(th):
        H = th*sum(zz(i, i+1, n) for i in range(n-1))
        for i in range(n): H = H + emb(X, i, n)
        return H
    return sq, hq, zz(0, 1, n)


def run(n):
    sq, hq, obs = tfim(n)
    PSIn = qp.tensor([qp.basis(2, 0)] * n)
    clean = NoisyQuTiPRunner(n, noise=None)
    noisy = NoisyQuTiPRunner(n, noise=NoiseModel(n_qubits=n, T2=T2))
    H, var = sq()

    def fc(th):
        return float(qp.expect(obs, (-1j*hq(th)*T).expm()*PSIn).real)
    fnz = noisy.make_expectation_fn(PSIn, obs)
    def fn(th):
        return fnz([[H.set_parameterizedHam({var: float(th)}), T]])

    xs = np.linspace(0.2, 2.0, 44)
    gi = np.array([(fc(x+1e-3)-fc(x-1e-3))/2e-3 for x in xs])
    # robust moderate-gradient point (same sign under noise, not at a feature)
    x_star, best = None, -1
    for k, x in enumerate(xs):
        if abs(gi[k]) > 0.3:
            lam = (fn(x+1e-2)-fn(x-1e-2))/2e-2 / gi[k]
            if 0.35 < lam < 0.95 and abs(gi[k]) > best:
                x_star, best = float(x), abs(gi[k])
    if x_star is None:
        x_star = float(xs[np.argmax(np.abs(gi))])
    g_ideal = (fc(x_star+1e-3)-fc(x_star-1e-3))/2e-3

    # FD best (infinite shots, ε→0) = noisy-landscape gradient
    g_fd = (fn(x_star+1e-3)-fn(x_star-1e-3))/2e-3
    # PSR exact mean (sum over M chain-rule terms)
    np.random.seed(7)
    progs = observable_program_generator(H, T, n_sample=POOL, n_repetition=1,
                                         diff_var=var, value=x_star)
    pexp = noisy.make_expectation_fn(PSIn, obs)
    g_psr = 0.0; M = 0
    for H_tot, ug, _ in progs:
        bj = len(H_tot)//2; M += 1
        e = np.array([pexp(H_tot[2*i]) for i in range(bj)])
        p = np.array([pexp(H_tot[2*i+1]) for i in range(bj)])
        g_psr += (T/bj)*float(ug)*np.sum(e - p)
    # analytic rescale (full-system slope; light-cone-computable for large n)
    s = ar.lambda_slope(hq, obs, PSIn, T, n, z_sites=range(n), theta=x_star,
                        n_grid=110)
    factor = ar.rescale_factor(s, T, T2)
    g_psr_res = g_psr * factor

    return dict(n=n, M=M, g_ideal=g_ideal,
                fd_bias=abs(g_fd - g_ideal),
                psr_bias=abs(g_psr - g_ideal),
                res_bias=abs(g_psr_res - g_ideal),
                lam_fd=g_fd/g_ideal, lam_psr=g_psr/g_ideal)


def main():
    ns = [2, 3, 4, 5, 6]
    res = [run(n) for n in ns]
    print(f"{'n':>3}{'M':>3}{'|g|':>8}{'λ_FD':>7}{'λ_PSR':>7}"
          f"{'FD bias':>9}{'PSR bias':>10}{'resc bias':>11}")
    for r in res:
        print(f"{r['n']:>3}{r['M']:>3}{abs(r['g_ideal']):>8.3f}{r['lam_fd']:>7.3f}"
              f"{r['lam_psr']:>7.3f}{r['fd_bias']:>9.4f}{r['psr_bias']:>10.4f}"
              f"{r['res_bias']:>11.4f}")

    nn = [r["n"] for r in res]
    fig, ax = plt.subplots(figsize=(7.8, 5.0), dpi=150)
    ax.plot(nn, [r["fd_bias"] for r in res], "s-", color="#7b1fa2", lw=2.4,
            label="FD best-ε  (= attenuation bias)")
    ax.plot(nn, [r["psr_bias"] for r in res], "o--", color="#9e9e9e", lw=2,
            label="PSR raw")
    ax.plot(nn, [r["res_bias"] for r in res], "o-", color="#00897b", lw=2.8,
            label="PSR rescaled")
    for r in res:
        ax.annotate(f"M={r['M']}", (r["n"], r["fd_bias"]), fontsize=7.5,
                    xytext=(0, 6), textcoords="offset points", ha="center",
                    color="#7b1fa2")
    ax.set_xlabel("chain length n  (chain-rule terms M = n−1)")
    ax.set_ylabel("gradient BIAS  |estimate − ideal|  (∞ shots)")
    ax.set_title("Bias only (variance set aside): FD's attenuation bias GROWS with\n"
                 "system size; PSR rescaled stays near zero for ANY M (rescale "
                 "corrects the total gradient)")
    ax.set_xticks(nn)
    ax.legend(frameon=False, fontsize=9)
    ax.text(0.02, 0.02, "FD's bias is its BEST case (∞ shots, ε→0).  Finite shots "
            "add variance ∝ 1/ε on top → FD worse still.", transform=ax.transAxes,
            fontsize=8, color="#444", va="bottom")
    fig.tight_layout()
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "figures", "bias_vs_size.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
