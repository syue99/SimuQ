"""
landscape_and_distance.py — sharp landscape with multiple FD secants, and the
distance |estimate − REAL (ideal) gradient| for FD(ε), PSR raw, PSR rescaled.

Regime: sharp landscape from a long evolution (T=4) but LOW dephasing (T2=16 →
T/T2*=0.25), the small-T/T2* regime where the analytic rescale works well.  The
landscape sharpness (→ FD step bias) comes from the coherent dynamics; the
dephasing (→ PSR attenuation, correctable) is a separate, mild rate.

Panel A — <Z0>(x): ideal and noisy, the small-gradient point x*, the TRUE ideal
  tangent, and FD secants at several ε (different & wrong slopes).
Panel B — distance to the REAL (ideal) gradient: FD at each ε (large, often wrong
  sign), PSR raw (small attenuation error), PSR rescaled (smallest — recovered).

Run:  conda run -n qec_pg python differential_computing/tests/landscape_and_distance.py
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
import scaling_universality as su
from noisy_qutip import NoisyQuTiPRunner
from noise_model import NoiseModel

T = 4.0
T2 = 16.0                 # T/T2* = 0.25 (small-noise regime, rescale accurate)
OBS = qp.tensor(qp.sigmaz(), qp.qeye(2))
PSI0 = qp.tensor(qp.basis(2, 0), qp.basis(2, 0))
I = qp.qeye(2); X, Z = qp.sigmax(), qp.sigmaz()
FD_EPS = [0.3, 0.6, 0.9]


def Hsq(theta):                                   # qutip H(theta)
    return theta * qp.tensor(Z, I) + qp.tensor(X, I)


def Hsimuq():
    x = sp.Symbol("x")
    qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    return x * q[0].Z + q[0].X, "x"


def f_of(runner):
    expfn = runner.make_expectation_fn(PSI0, OBS)
    return lambda x: expfn([[Hsimuq()[0].set_parameterizedHam({"x": float(x)}), T]])


def main():
    clean = NoisyQuTiPRunner(2, noise=None)
    noisy = NoisyQuTiPRunner(2, noise=NoiseModel(n_qubits=2, T2=T2))
    fc, fn = f_of(clean), f_of(noisy)

    # find small-gradient sharp point on the ideal landscape
    xs_scan = np.linspace(0.2, 2.2, 300)
    g = np.array([(fc(x + 1e-3) - fc(x - 1e-3)) / 2e-3 for x in xs_scan])
    h = 0.05
    f3 = np.array([(fc(x+2*h)-2*fc(x+h)+2*fc(x-h)-fc(x-2*h))/(2*h**3) for x in xs_scan])
    score = np.abs(f3) / (np.abs(g) + 0.05); score[np.abs(g) < 0.05] = -1
    x_star = float(xs_scan[np.argmax(score)])

    # the REAL gradient (ideal) at x*
    g_real = (fc(x_star + 1e-3) - fc(x_star - 1e-3)) / 2e-3

    # PSR raw (under noise) and rescaled
    H, var = Hsimuq()
    g_psr_raw = su.psr_grad(H, var, x_star, T, noisy, 2, OBS, n_sample=300)
    s = ar.lambda_slope(Hsq, OBS, PSI0, T, 2, z_sites=[0], theta=x_star)
    factor = ar.rescale_factor(s, T, T2)
    g_psr_resc = g_psr_raw * factor

    # FD estimates on the noisy landscape at several ε
    fd = {eps: (fn(x_star + eps) - fn(x_star - eps)) / (2 * eps) for eps in FD_EPS}

    print(f"Sharp landscape T={T}, T2={T2} (T/T2*={T/T2:.2f}).  x*={x_star:.3f}.")
    print(f"  REAL (ideal) gradient = {g_real:+.4f}")
    print(f"  PSR raw   = {g_psr_raw:+.4f}  (dist {abs(g_psr_raw-g_real):.4f})")
    print(f"  PSR resc  = {g_psr_resc:+.4f}  (dist {abs(g_psr_resc-g_real):.4f})  "
          f"[slope s={s:+.3f}, factor 1/λ={factor:.3f}]")
    for eps in FD_EPS:
        print(f"  FD ε={eps} = {fd[eps]:+.4f}  (dist {abs(fd[eps]-g_real):.4f})"
              f"{'  WRONG sign' if np.sign(fd[eps])!=np.sign(g_real) else ''}")

    # ── plot ──
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=150)
    xs = np.linspace(0.0, 1.4, 140)
    Zc = np.array([fc(x) for x in xs]); Zn = np.array([fn(x) for x in xs])
    axA.plot(xs, Zc, "--", color="#999999", lw=1.8, label="ideal landscape")
    axA.plot(xs, Zn, color="#2c3e50", lw=2.2, label=f"noisy (T2={T2:.0f})")
    z0 = fc(x_star)
    tx = np.array([x_star - 0.25, x_star + 0.25])
    axA.plot(tx, z0 + g_real * (tx - x_star), color="#1f77b4", lw=2.6,
             label=f"TRUE tangent ({g_real:+.2f})")
    fdc = ["#d62728", "#ff7f0e", "#9467bd"]
    for eps, c in zip(FD_EPS, fdc):
        axA.plot([x_star-eps, x_star+eps], [fn(x_star-eps), fn(x_star+eps)],
                 "s-", color=c, lw=1.8, ms=5, label=f"FD ε={eps} ({fd[eps]:+.2f})")
    axA.axhline(0, color="gray", lw=0.8); axA.axvline(x_star, color="gray", ls=":", lw=1)
    axA.plot([x_star], [z0], "ko", ms=6)
    axA.set_xlabel("parameter x"); axA.set_ylabel(r"$\langle Z_0\rangle(x)$")
    axA.set_title(f"(A) sharp landscape, FD secants at several ε  (x*={x_star:.2f})")
    axA.legend(frameon=False, fontsize=8, loc="lower left")

    # Panel B: distance to the real gradient
    labels = [f"FD ε={e}" for e in FD_EPS] + ["PSR raw", "PSR rescaled"]
    dists = [abs(fd[e] - g_real) for e in FD_EPS] + \
            [abs(g_psr_raw - g_real), abs(g_psr_resc - g_real)]
    cols = fdc + ["#7f7f7f", "#1f77b4"]
    bars = axB.bar(range(len(labels)), dists, color=cols)
    axB.axhline(abs(g_real), color="k", ls=":", lw=1.2,
                label=f"|real gradient| = {abs(g_real):.3f}")
    for b, e in zip(bars[:len(FD_EPS)], FD_EPS):
        if np.sign(fd[e]) != np.sign(g_real):
            axB.text(b.get_x()+b.get_width()/2, b.get_height(), "wrong\nsign",
                     ha="center", va="bottom", fontsize=7, color="#d62728")
    axB.set_xticks(range(len(labels))); axB.set_xticklabels(labels, fontsize=8, rotation=15)
    axB.set_ylabel("distance  |estimate − real gradient|")
    axB.set_title("(B) error vs the REAL (ideal) gradient")
    axB.legend(frameon=False, fontsize=8.5)

    fig.suptitle(f"Small-T/T2* regime (T/T2*={T/T2:.2f}): FD's secant has the WRONG "
                 f"sign at every ε; PSR (raw & rescaled) keep the\ndirection and "
                 f"sit closest to the real gradient (rescale partially corrects the "
                 f"magnitude near this sharp feature)", fontsize=9.0)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "figures", "landscape_and_distance.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
