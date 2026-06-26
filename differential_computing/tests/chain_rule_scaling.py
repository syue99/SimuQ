"""
chain_rule_scaling.py — does PSR's shot advantage erode when a parameter affects
MANY terms (the chain-rule sum)?

In H2/MaxCut each parameter multiplies ONE generator → PSR does 1 kick-pair per
gradient component (favorable).  In a realistic analog ansatz a parameter affects
M terms, so PSR sums M kick-pairs (cost ∝ M·n_sample), while FD is always 2
evaluations.  At equal shot budget, PSR's shots/eval are diluted by M·n_sample.

We build H(v) = v·(Σ_{j=1}^M G_j) + fixed mixer on 3 qubits, and compare the
single-point gradient RMSE (vs the exact gradient) of PSR vs FD at EQUAL total shot
budget, sweeping M.  PSR's RMSE should grow with M; FD's (at fixed ε) stays flat.
The crossover is where the chain-rule cost erases PSR's advantage.

Run:  conda run -n qec_pg python differential_computing/tests/chain_rule_scaling.py
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
from noisy_qutip import NoisyQuTiPRunner

I = qp.qeye(2); X, Z = qp.sigmax(), qp.sigmaz()
NQ = 3
T = 0.6
PSI0 = qp.tensor([qp.basis(2, 0)] * NQ)
OBS = qp.tensor(Z, Z, I)               # Z0 Z1


def emb(op, i, j=None):
    o = [I] * NQ; o[i] = op
    if j is not None:
        o[j] = op
    return qp.tensor(o)


# pool of generators a single parameter v can multiply (Z / ZZ terms)
POOL_QP = [emb(Z, 0), emb(Z, 1), emb(Z, 2), emb(Z, 0, 1), emb(Z, 1, 2), emb(Z, 0, 2)]
MIXER_QP = 0.5 * sum(emb(X, i) for i in range(NQ))


def build(M):
    """Parametrized H where v multiplies the first M pool generators."""
    v = sp.Symbol("v")
    qs = QSystem(); q = [Qubit(qs) for _ in range(NQ)]
    pool = [q[0].Z, q[1].Z, q[2].Z, q[0].Z * q[1].Z, q[1].Z * q[2].Z,
            q[0].Z * q[2].Z]
    H = v * pool[0]
    for j in range(1, M):
        H = H + v * pool[j]
    for i in range(NQ):
        H = H + 0.5 * q[i].X            # fixed mixer
    Hqp = lambda val: (val * sum(POOL_QP[:M]) + MIXER_QP)
    return H, "v", Hqp


def exact_grad(Hqp, theta, eps=1e-3):
    def f(val):
        s = (-1j * T * Hqp(val)).expm() * PSI0
        return float(qp.expect(OBS, s).real)
    return (f(theta + eps) - f(theta - eps)) / (2 * eps)


def psr_components(H, var, theta, runner):
    """exact per-branch <O> + weights for PSR (1 τ sample per term)."""
    np.random.seed(0)
    progs = observable_program_generator(H, T, n_sample=1, n_repetition=1,
                                         diff_var=var, value=theta)
    expfn = runner.make_expectation_fn(PSI0, OBS)
    w, e = [], []
    for H_tot, ug, _ in progs:
        # n_sample=1 → 2 branches (minus, plus)
        w.append(+float(ug) * T); e.append(expfn(H_tot[0]))
        w.append(-float(ug) * T); e.append(expfn(H_tot[1]))
    return np.array(w), np.array(e)


def fd_components(Hqp, theta, eps, runner):
    def ev(val):
        rho = runner.run_sequence([[None, 0]], PSI0)  # placeholder
        return 0.0
    # exact <O> at v±eps (qutip), then shot-sample
    def f(val):
        s = (-1j * T * Hqp(val)).expm() * PSI0
        return float(qp.expect(OBS, s).real)
    w = 1.0 / (2 * eps)
    return np.array([+w, -w]), np.array([f(theta + eps), f(theta - eps)])


def resample(w, e, n_per, R, rng):
    p = 0.5 * (1 + np.clip(e, -1, 1))
    k = rng.binomial(int(max(1, n_per)), p[None, :], size=(R, len(e)))
    return (2.0 * k / max(1, n_per) - 1.0) @ w


def main():
    runner = NoisyQuTiPRunner(NQ, noise=None)
    theta = 0.7
    S = 4000               # total shots / gradient (fixed, equal for PSR & FD)
    R, rng = 3000, np.random.default_rng(0)
    Ms = [1, 2, 3, 6]
    eps_list = [0.05, 0.2]

    print(f"Chain-rule scaling — H(v)=v·Σ_{{1..M}}G_j + mixer, 3 qubits, <Z0Z1>.")
    print(f"Equal total budget S={S} shots/grad.  Single-point gradient RMSE "
          f"vs M.\n")
    hdr = "PSR" + "".join(f"  FD ε={e}" for e in eps_list)
    print(f"{'M':>3}{'PSR RMSE':>11}" + "".join(f"{f'FD ε={e}':>11}" for e in eps_list))
    psr_rmse, fd_rmse = [], {e: [] for e in eps_list}
    for M in Ms:
        H, var, Hqp = build(M)
        truth = exact_grad(Hqp, theta)
        # PSR: 2M branch evals; shots/eval = S/(2M)
        wp, ep = psr_components(H, var, theta, runner)
        gp = resample(wp, ep, S / (2 * M), R, rng)
        pr = float(np.sqrt(np.mean((gp - truth) ** 2))); psr_rmse.append(pr)
        row = f"{M:>3}{pr:>11.4f}"
        for eps in eps_list:
            wf, ef = fd_components(Hqp, theta, eps, runner)
            gf = resample(wf, ef, S / 2, R, rng)   # FD: 2 evals, shots/eval=S/2
            fr = float(np.sqrt(np.mean((gf - truth) ** 2))); fd_rmse[eps].append(fr)
            row += f"{fr:>11.4f}"
        print(row)

    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=150)
    ax.plot(Ms, psr_rmse, "o-", color="#1f77b4", lw=2.2, label="PSR (n=1)")
    for eps, c in zip(eps_list, ["#d62728", "#ff7f0e"]):
        ax.plot(Ms, fd_rmse[eps], "s-", color=c, lw=2, label=f"FD ε={eps}")
    ax.set_xlabel("M  (terms one parameter affects — chain-rule sum)")
    ax.set_ylabel("single-point gradient RMSE")
    ax.set_title("PSR's shot advantage erodes as the chain-rule sum grows\n"
                 "(equal total budget; PSR cost ∝ M, FD always 2 evals)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "figures", "chain_rule_scaling.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
