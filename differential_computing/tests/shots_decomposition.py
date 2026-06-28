"""
shots_decomposition.py — separate BIAS and VARIANCE in the shots-scaling distance.

The "distance" in shots_scaling.py is RMSE over shot realizations =
sqrt(bias² + variance).  This splits the two so it's unambiguous:
  - bias  = |mean(est) − true|   (CONSTANT in N — the attenuation/residual offset)
  - std   = sqrt(Var(est))       (shrinks ~N^{-1/2})
  - RMSE  = sqrt(bias² + std²)   (what was plotted; floors at the bias)
for PSR raw, PSR rescaled, and FD (fixed ε).

Run:  conda run -n qec_pg python differential_computing/tests/shots_decomposition.py
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

T, T2 = 2.5, 5.0          # T/T2* = 0.50
OBS = qp.tensor(qp.sigmaz(), qp.qeye(2))
PSI0 = qp.tensor(qp.basis(2, 0), qp.basis(2, 0))
I = qp.qeye(2); X, Z = qp.sigmax(), qp.sigmaz()
R, POOL, NS_CAP, FD_EPS = 4000, 1200, 1500, 0.3


def Hsq(th): return th * qp.tensor(Z, I) + qp.tensor(X, I)


def Hsimuq():
    x = sp.Symbol("x"); qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    return x * q[0].Z + q[0].X, "x"


def fmaker(r):
    H, var = Hsimuq()
    e = r.make_expectation_fn(PSI0, OBS)
    return lambda x: e([[H.set_parameterizedHam({"x": float(x)}), T]])


def stats(est, true):
    bias = abs(float(np.mean(est)) - true)
    std = float(np.std(est))
    rmse = float(np.sqrt(np.mean((est - true) ** 2)))
    return bias, std, rmse


def main():
    rng = np.random.default_rng(0)
    clean = NoisyQuTiPRunner(2, noise=None)
    noisy = NoisyQuTiPRunner(2, noise=NoiseModel(n_qubits=2, T2=T2))
    fc, fn = fmaker(clean), fmaker(noisy)
    xs = np.linspace(0.2, 1.6, 90)
    gi = np.array([(fc(x+1e-3)-fc(x-1e-3))/2e-3 for x in xs])
    x_star = float(xs[np.argmax(np.abs(gi))])
    g_real = (fc(x_star+1e-3)-fc(x_star-1e-3))/2e-3
    s = ar.lambda_slope(Hsq, OBS, PSI0, T, 2, z_sites=[0], theta=x_star)
    factor = ar.rescale_factor(s, T, T2)

    H, var = Hsimuq()
    np.random.seed(123)
    progs = observable_program_generator(H, T, n_sample=POOL, n_repetition=1,
                                         diff_var=var, value=x_star)
    pexp = noisy.make_expectation_fn(PSI0, OBS)
    H_tot, ug, _ = progs[0]; b = len(H_tot)//2
    em = np.array([pexp(H_tot[2*i]) for i in range(b)])
    ep = np.array([pexp(H_tot[2*i+1]) for i in range(b)]); ug = float(ug)
    fp_ex, fm_ex = fn(x_star+FD_EPS), fn(x_star-FD_EPS)

    budgets = np.array([200, 600, 2000, 6000, 20000, 60000, 200000])
    D = {k: {"bias": [], "std": [], "rmse": []}
         for k in ("PSR raw", "PSR rescaled", f"FD ε={FD_EPS}")}
    for N in budgets:
        ns = int(min(N//2, NS_CAP)); npr = int(max(1, round(N/(2*ns))))
        idx = rng.integers(0, len(em), size=(R, ns))
        fm = 2.0*rng.binomial(npr, 0.5*(1+np.clip(em[idx], -1, 1)))/npr-1
        fp = 2.0*rng.binomial(npr, 0.5*(1+np.clip(ep[idx], -1, 1)))/npr-1
        raw = (T/ns)*ug*np.sum(fm-fp, axis=1)
        for nm, e in (("PSR raw", raw), ("PSR rescaled", raw*factor)):
            b_, s_, r_ = stats(e, g_real)
            D[nm]["bias"].append(b_); D[nm]["std"].append(s_); D[nm]["rmse"].append(r_)
        nfd = N//2
        fpb = 2.0*rng.binomial(nfd, 0.5*(1+np.clip(fp_ex, -1, 1)), size=R)/nfd-1
        fmb = 2.0*rng.binomial(nfd, 0.5*(1+np.clip(fm_ex, -1, 1)), size=R)/nfd-1
        fd = (fpb-fmb)/(2*FD_EPS)
        b_, s_, r_ = stats(fd, g_real)
        k = f"FD ε={FD_EPS}"
        D[k]["bias"].append(b_); D[k]["std"].append(s_); D[k]["rmse"].append(r_)

    print(f"x*={x_star:.3f}, real grad={g_real:+.4f}, λ={1/factor:.3f}, "
          f"attenuation bias (1-λ)|g|={(1-1/factor)*abs(g_real):.4f}\n")
    for nm in D:
        print(f"{nm}:  bias(@maxN)={D[nm]['bias'][-1]:.4f}  "
              f"std: {D[nm]['std'][0]:.3f}->{D[nm]['std'][-1]:.4f}  "
              f"rmse(@maxN)={D[nm]['rmse'][-1]:.4f}")

    colors = {"PSR raw": "#9e9e9e", "PSR rescaled": "#00897b", f"FD ε={FD_EPS}": "#7b1fa2"}
    fig, (axB, axR) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150, sharey=True)
    for nm in D:
        c = colors[nm]
        axB.loglog(budgets, D[nm]["bias"], "o-", color=c, lw=2.2, label=f"{nm} bias")
        axB.loglog(budgets, D[nm]["std"], "^--", color=c, lw=1.4, alpha=0.7,
                   label=f"{nm} std")
        axR.loglog(budgets, D[nm]["rmse"], "o-", color=c, lw=2.4, label=nm)
        axR.loglog(budgets, D[nm]["bias"], ":", color=c, lw=1.4)
    axB.set_xlabel("total shots N"); axB.set_ylabel("error component")
    axB.set_title("(A) BIAS (solid, flat) vs STD (dashed, ~N$^{-1/2}$)")
    axB.legend(frameon=False, fontsize=7.5, ncol=1)
    axR.set_xlabel("total shots N")
    axR.set_title(r"(B) RMSE = $\sqrt{bias^2+var}$  (→ its bias floor, dotted)")
    axR.legend(frameon=False, fontsize=8)
    fig.suptitle("Splitting the 'distance': RMSE = bias² + variance.  More shots "
                 "kill VARIANCE (std→0); only the\nrescale kills the BIAS.  PSR raw "
                 "RMSE floors at its constant attenuation bias; rescaled keeps falling.",
                 fontsize=9.2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "figures", "shots_decomposition.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
