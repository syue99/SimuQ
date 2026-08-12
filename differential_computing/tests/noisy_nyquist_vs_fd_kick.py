"""
noisy_nyquist_vs_fd_kick.py — the three-way comparison UNDER NOISE, using the
same noise model as the device-target figures: dephasing at T/T2*=0.5 and a
control-resolution setpoint error δ ~ N(0, r), r=0.02, plus finite shots.

Estimand: ∇C_noisy(θ0) — the gradient of the deployed program's own DEPHASED
landscape (the device gradient the kick-exactness theorem targets).  The question
this answers empirically: does that exactness TRANSFER to the Nyquist waveform
shift?  If Nyquist also converges to ∇C_noisy (shot-limited, δ-robust), the
device-noise lemma covers both strategies.

Noise entry (physical, and the crux):
  * FD programs TWO shifted setpoints θ0±ε, each with resolution error δ, so δ
    enters as δ/ε — a shot-independent variance that blows up as ε→0 (and large
    ε truncates).  Oracle-FD picks the best ε; it still floors.
  * kick and Nyquist program the BASE θ0 once (plus, resp., a Pauli kick or a
    large waveform shift s≳1/4K), so δ is only an operating-point shift d·C''(θ0)
    — ε-free.  They floor at the tiny r·|C''| instead of the δ/ε minimum.

For the linear generator H=θZ+X, FD and Nyquist executions are landscape samples
C_noisy(θ'), so C_noisy(θ) is precomputed once on a grid (dephased mesolve) and
interpolated; kick's 3-segment branch values are precomputed at θ0.  Trials then
reduce to interpolation + binomial sampling.  Caches figures/noisy_nyquist_vs_fd_kick.{json,png}.
Run: conda run -n qec_pg python differential_computing/tests/noisy_nyquist_vs_fd_kick.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sympy as sp
import qutip as qp
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simuq import QSystem, Qubit
from noisy_qutip import NoisyQuTiPRunner
from noise_model import NoiseModel
from observable_program_generator import observable_program_generator
from nyquist_shift import nyquist_program_generator, tangent_hamiltonian, bandwidth_K

T, T2, X0, R_CTRL = 2.5, 5.0, 0.7, 0.02          # T/T2*=0.5, δ~N(0,0.02)  (as before)
N_KICK, N_NYQ = 16, 16                            # matched 32 base executions
BUDGETS = [1600, 6400, 25600, 102400, 409600]
R_TRIAL = 400
OBS = qp.tensor(qp.sigmaz(), qp.qeye(2))
PSI0 = qp.tensor(qp.basis(2, 0), qp.basis(2, 0))
FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))


def Hsys():
    x = sp.Symbol("x"); qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    return x * q[0].Z + q[0].X


def shots(val, n, rng):
    return 2.0 * rng.binomial(int(max(1, n)), 0.5 * (1 + np.clip(val, -1, 1))) / max(1, n) - 1.0


def precompute():
    H = Hsys()
    noisy = NoisyQuTiPRunner(2, noise=NoiseModel(n_qubits=2, T2=T2))
    expfn = noisy.make_expectation_fn(PSI0, OBS)
    C = lambda th: expfn([[H.set_parameterizedHam({"x": float(th)}), T]])

    _, A = tangent_hamiltonian(H, "x", X0); K = bandwidth_K(A, T)
    s_max = (max(N_KICK, N_NYQ) + 0.5) / (2 * K)
    grid = np.linspace(X0 - s_max - 0.3, X0 + s_max + 0.3, 900)
    Cg = np.array([C(th) for th in grid])
    Cint = interp1d(grid, Cg, kind="cubic")
    print(f"K={K:.3f}  s_max={s_max:.2f}  grid[{grid[0]:.1f},{grid[-1]:.1f}] ({len(grid)} pts)")

    h = 1e-2
    grad_true = (C(X0 + h) - C(X0 - h)) / (2 * h)
    C2 = (C(X0 + h) - 2 * C(X0) + C(X0 - h)) / h ** 2        # C''(θ0): δ operating-pt slope
    print(f"∇C_noisy(θ0)={grad_true:+.5f}  C''={C2:+.4f}  δ-floor r·|C''|={R_CTRL*abs(C2):.4f}")

    # kick branch exact values at θ0 (deterministic midpoint τ)
    tau = (np.arange(N_KICK) + 0.5) / N_KICK * T
    progs = observable_program_generator(H, T, N_KICK, 1, "x", X0, tau_list=tau)
    H_tot, ugrad, _ = progs[0]
    fm = np.array([expfn(H_tot[2 * i]) for i in range(N_KICK)])      # f⁻
    fp = np.array([expfn(H_tot[2 * i + 1]) for i in range(N_KICK)])  # f⁺
    kick = {"fm": fm, "fp": fp, "ugrad": float(ugrad)}

    # Nyquist shift lists + weights (base θ0), for "none" and "lanczos".
    # For H=θZ+X the shifted single-segment H = (θ0+s)Z + X, so the effective
    # landscape sample point is θ0 + s; the generator returns the shift list.
    nyq = {}
    for win in ("none", "lanczos"):
        pr, info = nyquist_program_generator(H, T, "x", X0, N=N_NYQ, window=win)
        nyq[win] = {"theta": X0 + np.array(info["shifts"]),
                    "w": np.array([p["weight"] for p in pr])}
    return dict(Cint=Cint, grad_true=grad_true, C2=C2, K=K, kick=kick, nyq=nyq)


def fd_rmse(P, eps, Ntot, rng):
    Cint, gt = P["Cint"], P["grad_true"]; n = Ntot // 2
    errs = []
    for _ in range(R_TRIAL):
        dp, dm = rng.normal(0, R_CTRL), rng.normal(0, R_CTRL)
        fp = shots(Cint(X0 + eps + dp), n, rng)
        fm = shots(Cint(X0 - eps + dm), n, rng)
        errs.append((fp - fm) / (2 * eps) - gt)
    return float(np.sqrt(np.mean(np.square(errs))))


def kick_rmse(P, Ntot, rng):
    # PSR is ε-free: no δ/ε. The base-setpoint δ is only a 2nd-order operating-
    # point shift (∇C_noisy(θ0+δ)=∇C_noisy(θ0)+O(δ)), so PSR converges to the
    # device gradient shot-limited — matching the prior device-target result.
    k, gt = P["kick"], P["grad_true"]; n = Ntot // (2 * N_KICK)
    errs = []
    for _ in range(R_TRIAL):
        sm = shots(k["fm"], n, rng); sp_ = shots(k["fp"], n, rng)
        g = (T / N_KICK) * k["ugrad"] * np.sum(sm - sp_)
        errs.append(g - gt)
    return float(np.sqrt(np.mean(np.square(errs))))


def nyq_rmse(P, win, Ntot, rng):
    d = P["nyq"][win]; Cint, gt = P["Cint"], P["grad_true"]
    n = Ntot // len(d["theta"])
    vals_exact = Cint(d["theta"])                       # noiseless dephased samples
    errs = []
    for _ in range(R_TRIAL):
        s = shots(vals_exact, n, rng)
        errs.append(float(np.sum(d["w"] * s)) - gt)
    return float(np.sqrt(np.mean(np.square(errs))))


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    P = precompute()
    rng = np.random.default_rng(0)
    eps_grid = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    res = {"fd_best": [], "fd_by_eps": {e: [] for e in eps_grid},
           "kick": [], "nyq_none": [], "nyq_lanczos": [],
           "budgets": BUDGETS, "grad_true": P["grad_true"],
           "delta_floor": R_CTRL * abs(P["C2"]), "T_over_T2": T / T2, "r": R_CTRL}
    for Ntot in BUDGETS:
        fds = {e: fd_rmse(P, e, Ntot, rng) for e in eps_grid}
        for e in eps_grid:
            res["fd_by_eps"][e].append(fds[e])
        res["fd_best"].append(min(fds.values()))
        res["kick"].append(kick_rmse(P, Ntot, rng))
        res["nyq_none"].append(nyq_rmse(P, "none", Ntot, rng))
        res["nyq_lanczos"].append(nyq_rmse(P, "lanczos", Ntot, rng))
        print(f"  N={Ntot:>7d}: FD-best={res['fd_best'][-1]:.4f}  kick={res['kick'][-1]:.4f}  "
              f"nyq={res['nyq_none'][-1]:.4f}  nyq-lanczos={res['nyq_lanczos'][-1]:.4f}")

    json.dump(res, open(os.path.join(FIGDIR, "noisy_nyquist_vs_fd_kick.json"), "w"),
              indent=2, default=float)
    figure(res)


def figure(res):
    N = np.array(res["budgets"])
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    ax.loglog(N, res["fd_best"], "s--", color="#D55E00", ms=6, label="oracle-FD (best ε)")
    ax.loglog(N, res["kick"], "o-", color="#009E73", ms=6, label="kick-PSR (det-τ)")
    ax.loglog(N, res["nyq_none"], "^-", color="#0072B2", ms=6, label="Nyquist det")
    ax.loglog(N, res["nyq_lanczos"], "v-", color="#56B4E9", ms=6, label="Nyquist det+Lanczos")
    ax.loglog(N, res["kick"][0] * (N / N[0]) ** -0.5, ":", color="#999", lw=1, label="$N^{-1/2}$")
    fd_floor = min(res["fd_best"])
    ax.axhline(fd_floor, color="#D55E00", lw=0.8, ls="-.")
    ax.text(N[-1] * 0.4, fd_floor * 1.2, r"FD $\delta/\varepsilon$ floor", fontsize=7,
            color="#a0451a", ha="right")
    ax.set_xlabel("total shots $N$"); ax.set_ylabel(r"RMSE vs $\nabla C_{\rm noisy}$")
    ax.set_title(f"$T/T_2^*$={res['T_over_T2']:.2f}, control $r$={res['r']}", fontsize=9)
    ax.legend(fontsize=7.5); ax.grid(True, which="both", alpha=0.15)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "noisy_nyquist_vs_fd_kick.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"figure: {out}")


if __name__ == "__main__":
    main()
