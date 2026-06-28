"""
plot_landscape.py — visualize the sharp landscape and WHY FD fails there.

Plots <Z0>(x) for H(x)=x·Z0+X0 evolved time T (ideal and under dephasing), marks
the small-gradient sharp point x*, and overlays:
  - the TRUE tangent at x* (small slope = the gradient we want),
  - the FD SECANT through x*±ε (its slope = FD's estimate),
showing the secant has the WRONG sign because ε spans the curvature.

Saves figures/landscape.png.

Run:  conda run -n qec_pg python differential_computing/tests/plot_landscape.py
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
from noisy_qutip import NoisyQuTiPRunner
from noise_model import NoiseModel

T = 4.0
OBS = qp.tensor(qp.sigmaz(), qp.qeye(2))
PSI0 = qp.tensor(qp.basis(2, 0), qp.basis(2, 0))
X_STAR = 0.506          # the small-gradient sharp point (from sharp_small_gradient)
EPS = 0.30


def H_eval(x):
    xs = sp.Symbol("x")
    qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    H = xs * q[0].Z + q[0].X
    return H.set_parameterizedHam({"x": float(x)})


def landscape(runner, xs):
    expfn = runner.make_expectation_fn(PSI0, OBS)
    return np.array([expfn([[H_eval(x), T]]) for x in xs])


def main():
    xs = np.linspace(0.0, 1.3, 140)
    clean = NoisyQuTiPRunner(2, noise=None)
    noisy = NoisyQuTiPRunner(2, noise=NoiseModel(n_qubits=2, T2=2.0))
    Zc = landscape(clean, xs)
    Zn = landscape(noisy, xs)

    # gradient (slope) and FD secant at x*, on the noisy landscape
    fn = lambda x: noisy.make_expectation_fn(PSI0, OBS)([[H_eval(x), T]])
    g_true = (fn(X_STAR + 1e-3) - fn(X_STAR - 1e-3)) / 2e-3
    fp, fm = fn(X_STAR + EPS), fn(X_STAR - EPS)
    g_fd = (fp - fm) / (2 * EPS)
    z_star = fn(X_STAR)

    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=150)
    ax.plot(xs, Zc, color="#999999", lw=1.8, ls="--", label="ideal landscape")
    ax.plot(xs, Zn, color="#2c3e50", lw=2.4, label="noisy landscape (T2=2)")

    # true tangent at x* (small slope)
    tx = np.array([X_STAR - 0.22, X_STAR + 0.22])
    ax.plot(tx, z_star + g_true * (tx - X_STAR), color="#1f77b4", lw=2.4,
            label=f"TRUE tangent (slope {g_true:+.2f})")
    # FD secant through x*±ε (wrong slope)
    ax.plot([X_STAR - EPS, X_STAR + EPS], [fm, fp], "s-", color="#d62728", lw=2.2,
            markersize=7, label=f"FD secant ε={EPS} (slope {g_fd:+.2f})")
    ax.axvline(X_STAR, color="gray", ls=":", lw=1)
    ax.plot([X_STAR], [z_star], "o", color="black", ms=7)
    ax.annotate(f"x* = {X_STAR}\nsmall gradient,\nsharp curvature",
                xy=(X_STAR, z_star), xytext=(X_STAR + 0.12, z_star + 0.35),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="black"))

    ax.set_xlabel("parameter  x"); ax.set_ylabel(r"$\langle Z_0\rangle(x)$")
    ax.set_title(f"Sharp landscape (H=x·Z0+X0, T={T}): the FD secant (ε={EPS}) "
                 f"has the WRONG sign\nbecause ε spans the curvature; the true "
                 f"tangent is small & opposite")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "figures", "landscape.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"x*={X_STAR}: true slope {g_true:+.4f}, FD secant slope {g_fd:+.4f} "
          f"(ε={EPS})  →  {'SAME sign' if g_true*g_fd>0 else 'OPPOSITE sign'}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
