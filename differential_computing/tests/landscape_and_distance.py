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

T = 2.5
T2 = 25.0                 # T/T2* = 0.10 (realistic operating regime)
OBS = qp.tensor(qp.sigmaz(), qp.qeye(2))
PSI0 = qp.tensor(qp.basis(2, 0), qp.basis(2, 0))
I = qp.qeye(2); X, Z = qp.sigmax(), qp.sigmaz()
FD_EPS = [0.3, 0.6, 0.9]     # secants drawn on the landscape (panel A)
N_SHOTS = 40000              # finite shot budget / gradient (panel B)
R_TRIALS = 4000
N_SAMPLE = 256               # PSR tau samples


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

    # PSR raw (under noise) + analytic rescale factor
    H, var = Hsimuq()
    s = ar.lambda_slope(Hsq, OBS, PSI0, T, 2, z_sites=[0], theta=x_star)
    factor = ar.rescale_factor(s, T, T2)

    # ── shot-noise machinery (RMSE distance to the REAL gradient) ──
    rng = np.random.default_rng(0)

    def shots(exact, n):
        p = 0.5 * (1 + np.clip(exact, -1, 1))
        return 2.0 * rng.binomial(int(max(1, n)), p, size=R_TRIALS) / max(1, n) - 1

    def fd_rmse(eps):
        n = N_SHOTS // 2
        est = (shots(fn(x_star + eps), n) - shots(fn(x_star - eps), n)) / (2 * eps)
        return float(np.sqrt(np.mean((est - g_real) ** 2)))

    # PSR pool (noisy branch expectations), subsample + shots per trial
    np.random.seed(123)
    progs = su.observable_program_generator(H, T, n_sample=800, n_repetition=1,
                                            diff_var=var, value=x_star)
    pexp = noisy.make_expectation_fn(PSI0, OBS)
    H_tot, ug, _ = progs[0]; b = len(H_tot) // 2
    em = np.array([pexp(H_tot[2 * i]) for i in range(b)])
    ep = np.array([pexp(H_tot[2 * i + 1]) for i in range(b)])

    def psr_est():
        n_per = int(max(1, round(N_SHOTS / (2 * N_SAMPLE))))
        idx = rng.integers(0, len(em), size=(R_TRIALS, N_SAMPLE))
        fm = 2.0 * rng.binomial(n_per, 0.5 * (1 + np.clip(em[idx], -1, 1))) / n_per - 1
        fp = 2.0 * rng.binomial(n_per, 0.5 * (1 + np.clip(ep[idx], -1, 1))) / n_per - 1
        return (T / N_SAMPLE) * float(ug) * np.sum(fm - fp, axis=1)

    psr_raw = psr_est()
    psr_raw_rmse = float(np.sqrt(np.mean((psr_raw - g_real) ** 2)))
    psr_resc_rmse = float(np.sqrt(np.mean((psr_raw * factor - g_real) ** 2)))

    eps_grid = np.geomspace(0.02, 1.5, 22)
    fd_rmses = [fd_rmse(float(e)) for e in eps_grid]

    print(f"Sharp landscape T={T}, T2={T2} (T/T2*={T/T2:.2f}).  x*={x_star:.3f}, "
          f"real grad={g_real:+.4f}, N={N_SHOTS} shots.")
    print(f"  PSR raw RMSE      = {psr_raw_rmse:.4f}")
    print(f"  PSR rescaled RMSE = {psr_resc_rmse:.4f}  [factor 1/λ={factor:.3f}]")
    for e, r in zip(eps_grid, fd_rmses):
        print(f"  FD ε={e:5.3f}  RMSE={r:.4f}")

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
    for eps, c in zip(FD_EPS, ["#d62728", "#ff7f0e", "#9467bd"]):
        sl = (fn(x_star + eps) - fn(x_star - eps)) / (2 * eps)
        axA.plot([x_star-eps, x_star+eps], [fn(x_star-eps), fn(x_star+eps)],
                 "s-", color=c, lw=1.8, ms=5, label=f"FD ε={eps} ({sl:+.2f})")
    axA.axhline(0, color="gray", lw=0.8); axA.axvline(x_star, color="gray", ls=":", lw=1)
    axA.plot([x_star], [z0], "ko", ms=6)
    axA.set_xlabel("parameter x"); axA.set_ylabel(r"$\langle Z_0\rangle(x)$")
    axA.set_title(f"(A) sharp landscape, FD secants at several ε  (x*={x_star:.2f})")
    axA.legend(frameon=False, fontsize=8, loc="lower left")

    # Panel B: log-log RMSE distance vs ε (U-shape) + PSR lines
    axB.loglog(eps_grid, fd_rmses, "s-", color="#d62728", lw=2, label="FD (shots)")
    axB.axhline(psr_raw_rmse, color="#7f7f7f", lw=2.2, label="PSR raw")
    axB.axhline(psr_resc_rmse, color="#1f77b4", lw=2.6, label="PSR rescaled")
    axB.axhline(abs(g_real), color="k", ls=":", lw=1.2,
                label=f"|real grad| = {abs(g_real):.3f}")
    # annotate the two failure arms
    axB.text(eps_grid[1], fd_rmses[1], "small ε:\nshot noise\nblows up",
             fontsize=8, color="#d62728", va="center")
    axB.text(eps_grid[-3], fd_rmses[-3], "large ε:\nbias (wrong\ndirection)",
             fontsize=8, color="#d62728", ha="right", va="center")
    axB.set_xlabel(r"FD step size $\varepsilon$")
    axB.set_ylabel("distance to real gradient  (RMSE)")
    axB.set_title(f"(B) error vs ε, N={N_SHOTS} shots — FD's U-shape")
    axB.legend(frameon=False, fontsize=8, loc="upper center")

    fig.suptitle(f"Realistic regime T/T2*={T/T2:.2f}, N={N_SHOTS} shots: FD is "
                 f"trapped — small ε → shot noise dominates, large ε → bias\n"
                 f"(wrong direction); no ε reaches the real gradient. PSR rescaled "
                 f"(analytic 1/λ) sits well below the whole U.", fontsize=8.8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "figures", "landscape_and_distance.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
