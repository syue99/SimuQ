"""
maxcut_psr_vs_fd.py — MaxCut QAOA from Leng et al. 2022 (Fig 2c,d): PSR vs FD.

4-cycle graph, maximize <C>, C = 2 − ½(Z0Z1+Z1Z2+Z2Z3+Z3Z0), max cut = 4.
Analog ansatz |ψ(v)> = exp(−iT Σ_k v_k G_k)|++++>, gens = 4 ZZ edges + 4 X mixers.
Cut measured with b_obs shots per ZZ term.  Per-parameter gradient: PSR (stochastic
n_sample=1) vs FD swept over ε.  Gradient ASCENT to maximize the cut; track the
true (noiseless) cut → 4.  Their finding (Fig 2d): FD does not converge under shot
noise; the parameter-shift method does.

Run:  conda run -n qec_pg python differential_computing/tests/maxcut_psr_vs_fd.py
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
from observable_program_generator import observable_program_generator
from combine_gradient import combine_gradient_results
from noisy_qutip import NoisyQuTiPRunner

I = qp.qeye(2); X, Z = qp.sigmax(), qp.sigmaz()
N = 4
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]


def _qp_op(p, i, j=None):
    ops = [I] * N
    ops[i] = p
    if j is not None:
        ops[j] = p
    return qp.tensor(ops)


C = 2.0 * qp.tensor([I] * N) - 0.5 * sum(_qp_op(Z, i, j) for i, j in EDGES)
MAXCUT = float(np.max(C.eigenenergies()))
ZZ_TERMS = [_qp_op(Z, i, j) for i, j in EDGES]      # for shot measurement
T = 3.0
PLUS = qp.tensor([(qp.basis(2, 0) + qp.basis(2, 1)).unit()] * N)

_qs = QSystem(); _q = [Qubit(_qs) for _ in range(N)]
GENS_SIMUQ = [_q[i].Z * _q[j].Z for i, j in EDGES] + [_q[i].X for i in range(N)]
GENS_QP = [_qp_op(Z, i, j) for i, j in EDGES] + [_qp_op(X, i) for i in range(N)]
NP = len(GENS_SIMUQ)


def H_of_v(v):
    H = float(v[0]) * GENS_SIMUQ[0]
    for k in range(1, NP):
        H = H + float(v[k]) * GENS_SIMUQ[k]
    return H


def H_param_k(v, k):
    sym = sp.Symbol("vk")
    H = sym * GENS_SIMUQ[k]
    for j in range(NP):
        if j != k:
            H = H + float(v[j]) * GENS_SIMUQ[j]
    return H


def true_cut(v):
    Hg = sum(vk * Gk for vk, Gk in zip(v, GENS_QP))
    s = (-1j * T * Hg).expm() * PLUS
    return float(qp.expect(C, s).real)


def cut_expfn(runner, b_obs, rng):
    def expfn(H_list):
        rho = runner.run_sequence(H_list, PLUS)
        c = 2.0
        for P in ZZ_TERMS:
            ev = min(1.0, max(-1.0, float(qp.expect(P, rho).real)))
            k = rng.binomial(b_obs, 0.5 * (1 + ev))
            c -= 0.5 * (2.0 * k / b_obs - 1.0)
        return c
    return expfn


def psr_grad(v, runner, b_obs, rng, seed):
    g = np.zeros(NP)
    expfn = cut_expfn(runner, b_obs, rng)
    for k in range(NP):
        np.random.seed(seed + k)
        progs = observable_program_generator(
            H_param_k(v, k), T, n_sample=1, n_repetition=1,
            diff_var="vk", value=float(v[k]))
        g[k] = combine_gradient_results(progs, expfn, T)
    return g


def fd_grad(v, runner, b_obs, eps, rng):
    expfn = cut_expfn(runner, b_obs, rng)
    g = np.zeros(NP)
    for k in range(NP):
        vp = v.copy(); vp[k] += eps
        vm = v.copy(); vm[k] -= eps
        g[k] = (expfn([[H_of_v(vp), T]]) - expfn([[H_of_v(vm), T]])) / (2 * eps)
    return g


def ascend(method, eps, v0, runner, b_obs, eta, n_epochs, seed):
    v = v0.copy()
    cuts = [true_cut(v)]
    rng = np.random.default_rng(seed)
    for ep in range(n_epochs):
        if method == "PSR":
            g = psr_grad(v, runner, b_obs, rng, seed + 7 * ep)
        else:
            g = fd_grad(v, runner, b_obs, eps, rng)
        v = v + eta * g            # ASCENT (maximize the cut)
        cuts.append(true_cut(v))
    return np.array(cuts)


def main():
    b_obs, eta, n_epochs, seeds = 100, 0.08, 40, 3
    fd_eps = [0.01, 0.1, 0.5]
    runner = NoisyQuTiPRunner(N, noise=None)        # shot noise only (Fig 2d)
    v0 = np.random.RandomState(2).uniform(-0.5, 0.5, NP)

    runs = [("PSR (no ε)", "PSR", None)] + [(f"FD ε={e}", "FD", e) for e in fd_eps]
    res = {}
    for label, method, eps in runs:
        Cu = np.zeros((seeds, n_epochs + 1))
        for s in range(seeds):
            Cu[s] = ascend(method, eps or 0.1, v0, runner, b_obs, eta, n_epochs,
                           seed=10 + s)
        res[label] = Cu

    print(f"MaxCut QAOA (4-cycle).  max cut={MAXCUT:.0f}.  {NP} params, T={T}, "
          f"b_obs={b_obs}/term, η={eta}, {n_epochs} epochs, {seeds} seeds.\n")
    print(f"{'method':>12}{'final cut':>11}{'deficit (4−cut)':>18}")
    for label, Cu in res.items():
        fc = Cu[:, -1].mean()
        print(f"{label:>12}{fc:>11.4f}{MAXCUT - fc:>18.4f}")

    steps = np.arange(n_epochs + 1)
    fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=150)
    fdc = ["#d62728", "#ff7f0e", "#9467bd"]
    for i, (label, Cu) in enumerate(res.items()):
        mu, sd = Cu.mean(0), Cu.std(0)
        c = "#1f77b4" if label.startswith("PSR") else fdc[(i - 1) % len(fdc)]
        lw = 2.6 if label.startswith("PSR") else 1.8
        ax.plot(steps, mu, label=label, color=c, lw=lw)
        ax.fill_between(steps, mu - sd, mu + sd, color=c, alpha=0.13)
    ax.axhline(MAXCUT, ls="--", color="k", lw=1, label=f"max cut = {MAXCUT:.0f}")
    ax.set_xlabel("epoch"); ax.set_ylabel(r"cut value $\langle C\rangle$")
    ax.set_title("MaxCut QAOA (4-cycle) under shot noise: PSR vs FD\n"
                 "PSR (no ε) maximizes the cut; FD at small ε stalls (shot noise)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.abspath(os.path.join(out_dir, "maxcut_psr_vs_fd.png"))
    fig.savefig(out)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
