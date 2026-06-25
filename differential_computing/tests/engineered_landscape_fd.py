"""
engineered_landscape_fd.py — engineer a landscape where FD cannot converge at ANY ε.

The H2 VQE has a large-gradient descent region that carries FD down regardless of
ε (the dilemma only bites near the minimum, too late to matter).  To expose the
dilemma we engineer a cost that is SHALLOW (small gradient) but SHARP (high
curvature) THROUGHOUT the descent, so FD is squeezed everywhere:
  - small ε → shot noise (1/(2ε)) drowns the weak gradient,
  - large ε → the secant spans the sharp curvature → bias dominates,
and no ε in between works.  PSR (exact shift, no ε) follows the weak gradient.

Engineered quantum cost (1 qubit): H(θ) = c·θ·Z + X, evolve time T, measure <Z>
from |0>.  Small c → weak θ-coupling → SHALLOW gradient; large T → rapid winding
of the dynamical phase → SHARP curvature.  We pick a start in a single descent
basin and run PSR vs FD over a wide ε sweep under shot noise.

Run:  conda run -n qec_pg python differential_computing/tests/engineered_landscape_fd.py
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

C_COUPLE = 0.35      # weak θ-coupling → shallow gradient
T = 5.0              # long evolution → sharp curvature
OBS = qp.sigmaz()
PSI0 = qp.basis(2, 0)


def H_param():
    th = sp.Symbol("th")
    qs = QSystem(); q = [Qubit(qs)]
    return C_COUPLE * th * q[0].Z + q[0].X, "th"


def ideal_cost(theta):
    H = (C_COUPLE * float(theta) * qp.sigmaz() + qp.sigmax())
    s = (-1j * T * H).expm() * PSI0
    return float(qp.expect(OBS, s).real)


def exact_grad(theta, eps=1e-3):
    return (ideal_cost(theta + eps) - ideal_cost(theta - eps)) / (2 * eps)


def shot_energy(runner, H_list, b_obs, rng):
    rho = runner.run_sequence(H_list, PSI0)
    e = min(1.0, max(-1.0, float(qp.expect(OBS, rho).real)))
    k = rng.binomial(b_obs, 0.5 * (1 + e))
    return 2.0 * k / b_obs - 1.0


def psr_grad(Hp, var, theta, runner, b_obs, rng, seed, n_sample=4):
    np.random.seed(seed)
    progs = observable_program_generator(Hp, T, n_sample=n_sample,
                                         n_repetition=1, diff_var=var, value=float(theta))
    expfn = lambda Hl: shot_energy(runner, Hl, b_obs, rng)
    return combine_gradient_results(progs, expfn, T)


def fd_grad(Hp, var, theta, eps, runner, b_obs, rng):
    Hp_p = Hp.set_parameterizedHam({var: theta + eps})
    Hp_m = Hp.set_parameterizedHam({var: theta - eps})
    fp = shot_energy(runner, [[Hp_p, T]], b_obs, rng)
    fm = shot_energy(runner, [[Hp_m, T]], b_obs, rng)
    return (fp - fm) / (2 * eps)


def descend(method, eps, Hp, var, theta0, runner, b_obs, eta, n_steps, seed):
    th = theta0
    costs = [ideal_cost(th)]
    rng = np.random.default_rng(seed)
    for st in range(n_steps):
        if method == "PSR":
            g = psr_grad(Hp, var, th, runner, b_obs, rng, seed + 7 * st)
        else:
            g = fd_grad(Hp, var, th, eps, runner, b_obs, rng)
        th = th - eta * g
        th = float(np.clip(th, THETA_LO, THETA_HI))
        costs.append(ideal_cost(th))
    return np.array(costs)


# pick a single descent basin (scan)
_grid = np.linspace(-3, 3, 601)
_c = np.array([ideal_cost(float(t)) for t in _grid])
# choose a window around a local min with a shallow approach
THETA_LO, THETA_HI = -2.0, 2.0
_w = (_grid >= THETA_LO) & (_grid <= THETA_HI)
THETA_STAR = float(_grid[_w][np.argmin(_c[_w])])
C_STAR = float(_c[_w].min())


def main():
    Hp, var = H_param()
    # start on the shallow shoulder of the basin (small gradient throughout)
    theta0 = THETA_STAR + 1.3
    g0 = exact_grad(theta0)
    b_obs, eta, n_steps, seeds = 50, 0.8, 50, 5
    fd_eps = [0.02, 0.05, 0.1, 0.3, 0.8]
    runner = NoisyQuTiPRunner(1, noise=None)   # shot noise only (clean test)

    print(f"Engineered shallow+sharp cost: H=cθZ+X, c={C_COUPLE}, T={T}.  "
          f"<Z> from |0>.\n θ*={THETA_STAR:.3f}, C*={C_STAR:.4f}, start θ0={theta0:.3f}"
          f" (|grad|≈{abs(g0):.4f} — shallow), b_obs={b_obs}, η={eta}, "
          f"{n_steps} steps, {seeds} seeds.\n")

    runs = [("PSR (no ε)", "PSR", None)] + [(f"FD ε={e}", "FD", e) for e in fd_eps]
    res = {}
    for label, method, eps in runs:
        Cs = np.zeros((seeds, n_steps + 1))
        for s in range(seeds):
            Cs[s] = descend(method, eps or 0.1, Hp, var, theta0, runner, b_obs,
                            eta, n_steps, seed=100 + s)
        res[label] = Cs

    print(f"{'method':>14}{'final C':>11}{'gap to C*':>12}{'best reached':>14}")
    for label, Cs in res.items():
        fe = Cs[:, -1].mean()
        print(f"{label:>14}{fe:>11.4f}{fe - C_STAR:>12.4f}{Cs.mean(0).min():>14.4f}")

    steps = np.arange(n_steps + 1)
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=150)
    fd_colors = ["#d62728", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]
    for i, (label, Cs) in enumerate(res.items()):
        mu, sd = Cs.mean(0), Cs.std(0)
        c = "#1f77b4" if label.startswith("PSR") else fd_colors[(i - 1) % len(fd_colors)]
        lw = 2.6 if label.startswith("PSR") else 1.7
        ax.plot(steps, mu, label=label, color=c, lw=lw)
        ax.fill_between(steps, mu - sd, mu + sd, color=c, alpha=0.12)
    ax.axhline(C_STAR, ls="--", color="k", lw=1, label="basin min $C^*$")
    ax.set_xlabel("descent step"); ax.set_ylabel(r"cost $\langle Z\rangle(\theta)$")
    ax.set_title("Engineered shallow+sharp landscape (no large-gradient highway):\n"
                 "PSR follows the weak gradient; every FD ε fails (variance / bias)")
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.abspath(os.path.join(out_dir, "engineered_landscape_fd.png"))
    fig.savefig(out)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
