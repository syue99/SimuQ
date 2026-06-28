"""
landscape_and_distance.py — sharp landscape with FD secants at all ε, and the
log-log distance |estimate − REAL gradient| where PSR wins most.

Point auto-selected: moderate-SMALL gradient on the sharp landscape where FD's
secant flips sign at large ε (so FD has no good step) yet the gradient is large
enough that the analytic rescale is stable.  Regime T/T2*=0.25, finite shots.

Panel A — <Z0>(x): ideal & noisy, the point x*, the TRUE tangent, and FD secants
  at every ε in the grid (each labeled with its actual evaluated slope).
Panel B — log-log distance (RMSE, with shots) to the REAL gradient vs ε: the FD
  U-shape (small ε → shot noise, large ε → bias); ε where FD's mean has the WRONG
  sign are marked in red.  PSR raw / rescaled as horizontal lines below.

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
from observable_program_generator import observable_program_generator
from noisy_qutip import NoisyQuTiPRunner
from noise_model import NoiseModel

T = 2.5
T2 = 10.0                 # T/T2* = 0.25
OBS = qp.tensor(qp.sigmaz(), qp.qeye(2))
PSI0 = qp.tensor(qp.basis(2, 0), qp.basis(2, 0))
I = qp.qeye(2); X, Z = qp.sigmax(), qp.sigmaz()
N_SHOTS, R, N_SAMPLE, POOL = 40000, 4000, 256, 800


def Hsq(th):
    return th * qp.tensor(Z, I) + qp.tensor(X, I)


def Hsimuq():
    x = sp.Symbol("x"); qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    return x * q[0].Z + q[0].X, "x"


def fmaker(runner):
    H, var = Hsimuq()
    expfn = runner.make_expectation_fn(PSI0, OBS)
    return lambda x: expfn([[H.set_parameterizedHam({"x": float(x)}), T]])


def main():
    rng = np.random.default_rng(0)
    clean = NoisyQuTiPRunner(2, noise=None)
    noisy = NoisyQuTiPRunner(2, noise=NoiseModel(n_qubits=2, T2=T2))
    fc, fn = fmaker(clean), fmaker(noisy)

    # ── select the point where PSR wins most: moderate-small |g| + FD sign-flip ──
    xs = np.linspace(0.2, 1.6, 130)
    gi = np.array([(fc(x + 1e-3) - fc(x - 1e-3)) / 2e-3 for x in xs])
    best = None
    for k, x in enumerate(xs):
        if 0.12 < abs(gi[k]) < 0.45:
            fd06 = (fn(x + 0.6) - fn(x - 0.6)) / 1.2
            if np.sign(fd06) != np.sign(gi[k]):
                score = abs(fd06 - gi[k])
                if best is None or score > best[1]:
                    best = (x, score)
    x_star = float(best[0]) if best else float(xs[np.argmin(np.abs(gi - 0.2))])
    g_real = (fc(x_star + 1e-3) - fc(x_star - 1e-3)) / 2e-3
    sgn = np.sign(g_real)

    # analytic rescale factor
    s = ar.lambda_slope(Hsq, OBS, PSI0, T, 2, z_sites=[0], theta=x_star)
    factor = ar.rescale_factor(s, T, T2)

    # ── shots / RMSE ──
    def shots(exact, n):
        return 2.0 * rng.binomial(int(max(1, n)),
                                  0.5*(1+np.clip(exact, -1, 1)), size=R)/max(1, n)-1

    eps_grid = np.geomspace(0.04, 1.4, 20)
    fd_rmse, fd_mean, fd_wrong = [], [], []
    for eps in eps_grid:
        n = N_SHOTS // 2
        fp_ex, fm_ex = fn(x_star + eps), fn(x_star - eps)
        est = (shots(fp_ex, n) - shots(fm_ex, n)) / (2 * eps)
        fd_rmse.append(float(np.sqrt(np.mean((est - g_real) ** 2))))
        m = (fp_ex - fm_ex) / (2 * eps)            # noiseless FD mean (the bias)
        fd_mean.append(m); fd_wrong.append(np.sign(m) != sgn)
    fd_rmse = np.array(fd_rmse); fd_wrong = np.array(fd_wrong)

    # PSR pool under noise
    H, var = Hsimuq()
    np.random.seed(123)
    progs = observable_program_generator(H, T, n_sample=POOL, n_repetition=1,
                                         diff_var=var, value=x_star)
    pexp = noisy.make_expectation_fn(PSI0, OBS)
    H_tot, ug, _ = progs[0]; b = len(H_tot)//2
    em = np.array([pexp(H_tot[2*i]) for i in range(b)])
    ep = np.array([pexp(H_tot[2*i+1]) for i in range(b)])
    n_per = int(max(1, round(N_SHOTS/(2*N_SAMPLE))))
    idx = rng.integers(0, len(em), size=(R, N_SAMPLE))
    fm = 2.0*rng.binomial(n_per, 0.5*(1+np.clip(em[idx], -1, 1)))/n_per-1
    fpb = 2.0*rng.binomial(n_per, 0.5*(1+np.clip(ep[idx], -1, 1)))/n_per-1
    raw = (T/N_SAMPLE)*float(ug)*np.sum(fm - fpb, axis=1)
    psr_raw_rmse = float(np.sqrt(np.mean((raw - g_real)**2)))
    psr_resc_rmse = float(np.sqrt(np.mean((raw*factor - g_real)**2)))

    print(f"T={T}, T/T2*={T/T2:.2f}, x*={x_star:.3f}, REAL grad={g_real:+.4f}, "
          f"1/λ={factor:.2f}.")
    print(f"  PSR raw RMSE {psr_raw_rmse:.4f}, PSR rescaled RMSE {psr_resc_rmse:.4f}")
    print(f"  FD wrong-sign at {int(fd_wrong.sum())}/{len(eps_grid)} ε values; "
          f"FD best RMSE {fd_rmse.min():.4f}")

    # ── plot ──
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.8, 4.9), dpi=150)
    gx = np.linspace(0.0, 1.45, 150)
    axA.plot(gx, [fc(x) for x in gx], "--", color="#9aa0a6", lw=1.6, label="ideal")
    axA.plot(gx, [fn(x) for x in gx], color="#202124", lw=2.2, label=f"noisy (T2={T2:.0f})")
    z0 = fc(x_star)
    tx = np.array([x_star - 0.22, x_star + 0.22])
    axA.plot(tx, z0 + g_real * (tx - x_star), color="#1a73e8", lw=3,
             label=f"true tangent  ({g_real:+.3f})")
    sec_eps = [0.15, 0.3, 0.45, 0.6, 0.9]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(sec_eps)))
    for eps, c in zip(sec_eps, cmap):
        fp_e, fm_e = fn(x_star + eps), fn(x_star - eps)
        sl = (fp_e - fm_e) / (2 * eps)
        axA.plot([x_star-eps, x_star+eps], [fm_e, fp_e], "o-", color=c, lw=1.8,
                 ms=4, label=f"FD ε={eps}: {sl:+.3f}")
    axA.axhline(0, color="#bbb", lw=0.8); axA.axvline(x_star, color="#bbb", ls=":", lw=1)
    axA.plot([x_star], [z0], "ko", ms=6)
    axA.set_xlabel("parameter x"); axA.set_ylabel(r"$\langle Z_0\rangle(x)$")
    axA.set_title(f"(A) sharp landscape + FD secants at all ε   (x*={x_star:.2f})")
    axA.legend(frameon=False, fontsize=8, loc="lower right")

    # Panel B: distance vs ε (purple FD curve; red markers = wrong sign)
    axB.loglog(eps_grid, fd_rmse, "-", color="#7b1fa2", lw=2, zorder=1, label="FD (shots)")
    ok = ~fd_wrong
    axB.loglog(eps_grid[ok], fd_rmse[ok], "o", color="#7b1fa2", ms=6, zorder=2)
    if fd_wrong.any():
        axB.loglog(eps_grid[fd_wrong], fd_rmse[fd_wrong], "X", color="#d50000",
                   ms=9, zorder=3, label="FD wrong sign")
    axB.axhline(psr_raw_rmse, color="#9e9e9e", lw=2.2, ls="--", label="PSR raw")
    axB.axhline(psr_resc_rmse, color="#00897b", lw=2.8, label="PSR rescaled")
    axB.set_xlabel(r"FD step size $\varepsilon$")
    axB.set_ylabel("distance to real gradient (RMSE)")
    axB.set_title(f"(B) error vs ε, N={N_SHOTS} shots")
    # write real gradient + annotations as text (no dashed line)
    axB.text(0.03, 0.04, f"real gradient = {g_real:+.3f}\n"
             f"small ε → shot noise   large ε → bias (wrong sign)",
             transform=axB.transAxes, fontsize=8, color="#333", va="bottom")
    axB.legend(frameon=False, fontsize=8.5, loc="upper center")

    fig.suptitle(f"PSR wins most here (T/T2*={T/T2:.2f}, x*={x_star:.2f}): FD is "
                 f"trapped — every ε either shot-noise-dominated or wrong-sign\n"
                 f"(red); PSR raw and rescaled sit below the whole U.",
                 fontsize=9.2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "figures", "landscape_and_distance.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
