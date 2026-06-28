"""
psr_vs_fd_noise_sweep.py — does PSR win MORE or LESS as T/T2* grows?

Fixed sharp landscape (T=2.5, small-gradient point), finite shots, swept dephasing
T/T2* ∈ {0.1,0.25,0.5,0.75,1.0}.  Distance = RMSE to the REAL (ideal, noise-
independent) gradient.  Compare FD (best ε over a grid), PSR raw, PSR rescaled
(analytic 1/λ from the ideal trajectory).

Physics: more dephasing attenuates the gradient.  FD's attenuation bias is
UNCORRECTABLE (it measures the noisy landscape); PSR's attenuation IS correctable
by the analytic rescale (until it breaks at high noise).  So PSR-rescaled's edge
over FD should GROW with T/T2* in the operating regime.

Run:  conda run -n qec_pg python differential_computing/tests/psr_vs_fd_noise_sweep.py
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
OBS = qp.tensor(qp.sigmaz(), qp.qeye(2))
PSI0 = qp.tensor(qp.basis(2, 0), qp.basis(2, 0))
I = qp.qeye(2); X, Z = qp.sigmax(), qp.sigmaz()
N_SHOTS, R, N_SAMPLE, POOL = 40000, 3000, 128, 400


def Hsq(th):
    return th * qp.tensor(Z, I) + qp.tensor(X, I)


def Hsimuq():
    x = sp.Symbol("x"); qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    return x * q[0].Z + q[0].X, "x"


def f_exact(runner, x):
    H, var = Hsimuq()
    return runner.make_expectation_fn(PSI0, OBS)(
        [[H.set_parameterizedHam({"x": float(x)}), T]])


def main():
    rng = np.random.default_rng(0)
    clean = NoisyQuTiPRunner(2, noise=None)
    # small-gradient sharp point on the ideal landscape
    xs = np.linspace(0.2, 2.2, 250)
    g = np.array([(f_exact(clean, x + 1e-3) - f_exact(clean, x - 1e-3)) / 2e-3
                  for x in xs])
    # MODERATE-gradient point (steepest region): the rescale is stable here
    # (small-gradient points make the analytic slope ∝1/g_ideal explode — a
    # separate instability shown elsewhere).  This isolates the noise-sweep.
    x_star = float(xs[np.argmax(np.abs(g))])
    g_real = (f_exact(clean, x_star + 1e-3) - f_exact(clean, x_star - 1e-3)) / 2e-3
    s = ar.lambda_slope(Hsq, OBS, PSI0, T, 2, z_sites=[0], theta=x_star)
    H, var = Hsimuq()
    print(f"T={T}, x*={x_star:.3f}, real grad={g_real:+.4f}, analytic slope s={s:+.3f}.\n")

    def shots(exact, n):
        return 2.0 * rng.binomial(int(max(1, n)),
                                  0.5*(1+np.clip(exact, -1, 1)), size=R)/max(1, n)-1

    eps_grid = np.geomspace(0.03, 1.3, 18)
    tt = [0.1, 0.25, 0.5, 0.75, 1.0]
    fd_best, psr_raw, psr_resc = [], [], []
    print(f"{'T/T2*':>7}{'FD best':>10}{'PSR raw':>10}{'PSR resc':>10}{'1/λ':>7}")
    for x in tt:
        T2 = T / x
        runner = NoisyQuTiPRunner(2, noise=NoiseModel(n_qubits=2, T2=T2))
        # FD U-shape min vs g_real
        fdrm = []
        for eps in eps_grid:
            n = N_SHOTS // 2
            est = (shots(f_exact(runner, x_star+eps), n)
                   - shots(f_exact(runner, x_star-eps), n)) / (2*eps)
            fdrm.append(np.sqrt(np.mean((est - g_real)**2)))
        fdmin = float(np.min(fdrm))
        # PSR pool under this noise
        np.random.seed(123)
        progs = observable_program_generator(H, T, n_sample=POOL, n_repetition=1,
                                             diff_var=var, value=x_star)
        pexp = runner.make_expectation_fn(PSI0, OBS)
        H_tot, ug, _ = progs[0]; b = len(H_tot)//2
        em = np.array([pexp(H_tot[2*i]) for i in range(b)])
        ep = np.array([pexp(H_tot[2*i+1]) for i in range(b)])
        n_per = int(max(1, round(N_SHOTS/(2*N_SAMPLE))))
        idx = rng.integers(0, len(em), size=(R, N_SAMPLE))
        fm = 2.0*rng.binomial(n_per, 0.5*(1+np.clip(em[idx], -1, 1)))/n_per-1
        fpb = 2.0*rng.binomial(n_per, 0.5*(1+np.clip(ep[idx], -1, 1)))/n_per-1
        raw = (T/N_SAMPLE)*float(ug)*np.sum(fm - fpb, axis=1)
        factor = ar.rescale_factor(s, T, T2)
        praw = float(np.sqrt(np.mean((raw - g_real)**2)))
        presc = float(np.sqrt(np.mean((raw*factor - g_real)**2)))
        fd_best.append(fdmin); psr_raw.append(praw); psr_resc.append(presc)
        print(f"{x:>7.2f}{fdmin:>10.4f}{praw:>10.4f}{presc:>10.4f}{factor:>7.2f}")

    fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=150)
    ax.plot(tt, fd_best, "s-", color="#d62728", lw=2.2, label="FD (best ε)")
    ax.plot(tt, psr_raw, "o--", color="#7f7f7f", lw=2, label="PSR raw")
    ax.plot(tt, psr_resc, "o-", color="#1f77b4", lw=2.6, label="PSR rescaled")
    ax.axhline(abs(g_real), color="k", ls=":", lw=1.2,
               label=f"|real grad| = {abs(g_real):.3f}")
    ax.set_xlabel(r"dephasing strength  $T/T_2^*$")
    ax.set_ylabel("distance to real gradient  (RMSE)")
    ax.set_title(f"Does PSR win more as noise grows?  (sharp T={T}, N={N_SHOTS} "
                 f"shots)\nFD's attenuation bias is uncorrectable; PSR rescaled "
                 f"corrects it → edge grows with T/T2*")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "figures", "psr_vs_fd_noise_sweep.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
