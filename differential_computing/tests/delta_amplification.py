"""
delta_amplification.py — show how the control error δ enters FD vs PSR as the FD
step ε approaches δ.

FD differences two shifted setpoints θ±ε, each carrying a setpoint error δ, so the
δ contribution is (δ/ε)·C' — AMPLIFIED by 1/ε.  As ε→δ this reaches O(∇C): the
estimate is as large as the gradient itself (sign unreliable).  Kick / Nyquist are
ε-free: they program the base once, so δ is only an operating-point offset δ·C''
— O(δ), FLAT in ε, never amplified.

Same dephased landscape as before (H=θZ+X, T/T2*=0.5).  Isolates δ (infinite-shot
limit) to expose the amplification cleanly.  Saves figures/delta_amplification.{png,json}.
Run: conda run -n qec_pg python differential_computing/tests/delta_amplification.py
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

T, T2, DELTA = 2.5, 5.0, 0.02
OBS = qp.tensor(qp.sigmaz(), qp.qeye(2)); PSI0 = qp.tensor(qp.basis(2, 0), qp.basis(2, 0))
FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    x = sp.Symbol("x"); qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    H = x * q[0].Z + q[0].X
    noisy = NoisyQuTiPRunner(2, noise=NoiseModel(n_qubits=2, T2=T2))
    ex = noisy.make_expectation_fn(PSI0, OBS)
    C = lambda th: ex([[H.set_parameterizedHam({"x": float(th)}), T]])

    # pick θ0 with a clear gradient and modest curvature (clean contrast)
    scan = np.linspace(0.3, 2.0, 60)
    h = 1e-2
    cp = np.array([C(t) for t in scan])
    x0 = float(scan[np.argmax(np.abs(np.gradient(cp, scan)))])
    Cp = (C(x0 + h) - C(x0 - h)) / (2 * h)              # ∇C(θ0)
    Cpp = (C(x0 + h) - 2 * C(x0) + C(x0 - h)) / h ** 2   # C''(θ0)
    print(f"θ0={x0:.3f}  ∇C={Cp:+.4f}  C''={Cpp:+.4f}  δ={DELTA}")

    grid = np.linspace(x0 - 1.2, x0 + 1.2, 400)
    Cint = interp1d(grid, [C(t) for t in grid], kind="cubic")

    rng = np.random.default_rng(0)
    eps = np.geomspace(DELTA, 0.6, 30)
    R = 20000
    fd_err, fd_wrong = [], []
    for e in eps:                                        # infinite-shot: isolate δ
        dp = rng.normal(0, DELTA, R); dm = rng.normal(0, DELTA, R)
        est = (Cint(np.clip(x0 + e + dp, grid[0], grid[-1]))
               - Cint(np.clip(x0 - e + dm, grid[0], grid[-1]))) / (2 * e)
        fd_err.append(float(np.sqrt(np.mean((est - Cp) ** 2))))
        fd_wrong.append(float(np.mean(np.sign(est) != np.sign(Cp))))
    psr_err = DELTA * abs(Cpp)                            # operating-point δ, ε-free

    print(f"  PSR δ-error (flat) = {psr_err:.4f}")
    for e, fe, fw in list(zip(eps, fd_err, fd_wrong))[::6]:
        print(f"  ε/δ={e/DELTA:5.1f}  FD err={fe:.3f}  wrong-sign={fw:.0%}")

    fig, ax = plt.subplots(figsize=(5.0, 3.7))
    ax.loglog(eps / DELTA, fd_err, "s-", color="#D55E00", ms=4, label=r"FD $\sim\delta/\varepsilon$")
    ax.axhline(psr_err, color="#0072B2", lw=2.2, label=r"kick / Nyquist $\sim\delta$ (flat)")
    ax.axhline(abs(Cp), color="#444", lw=0.9, ls=":", label=r"$|\nabla C|$")
    ax.axvline(1.0, color="#999", lw=0.8, ls="--")
    ax.text(1.05, fd_err[0] * 0.6, r"$\varepsilon=\delta$", fontsize=7, color="#666")
    ax.set_xlabel(r"$\varepsilon/\delta$"); ax.set_ylabel(r"gradient error (RMSE)")
    ax.set_title(rf"$T/T_2^*={T/T2:.2f}$, $\delta={DELTA}$: FD amplifies $\delta$ by $1/\varepsilon$",
                 fontsize=8.5)
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.15)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "delta_amplification.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    json.dump(dict(x0=x0, grad=Cp, Cpp=Cpp, delta=DELTA, eps=list(eps),
                   fd_err=fd_err, fd_wrong=fd_wrong, psr_err=psr_err),
              open(os.path.join(FIGDIR, "delta_amplification.json"), "w"), indent=2, default=float)
    print(f"figure: {out}")


if __name__ == "__main__":
    main()
