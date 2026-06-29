"""
fd_correctable.py — CAN the same analytic rescale fix FD too?

Honest test of a subtle point.  FD-best (ε→0, ∞ shots) returns exactly the noisy-
landscape gradient λ'·g_ideal — the SAME attenuated value PSR raw returns (we
measured λ_PSR=λ_FD).  So the SAME 1/λ rescale must also recover g_ideal from FD —
the ATTENUATION is correctable for FD too.  BUT:
  - at finite ε, FD also carries TRUNCATION bias ε²f'''/6 → rescale multiplies it by
    1/λ (AMPLIFIES it).  Only vanishes as ε→0.
  - at finite shots, FD variance ∝1/ε² → rescale multiplies it by (1/λ)² (AMPLIFIES).
So FD's attenuation is correctable, but the rescale does NOT rescue FD's ε-dilemma
(truncation + variance).  PSR is ε-free → it has ONLY the attenuation → rescale lands
it on the true gradient.

Panel A (∞ shots, BIAS): rescaled-FD bias → 0 as ε→0 (attenuation correctable) but
grows with ε (amplified truncation).  PSR rescaled = flat ~0.
Panel B (finite shots, RMSE): rescaled-FD keeps the U-shape (variance amplified);
PSR rescaled = flat low.

Run:  conda run -n qec_pg python differential_computing/tests/fd_correctable.py
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

NQ, T, T2 = 3, 1.5, 20.0
N_SHOTS, R = 20000, 3000
I = qp.qeye(2); X, Z = qp.sigmax(), qp.sigmaz()


def emb(op, i, n): return qp.tensor([op if k == i else I for k in range(n)])
def zz(i, j, n): return emb(Z, i, n) * emb(Z, j, n)


def hq(th, n):
    H = th*sum(zz(i, i+1, n) for i in range(n-1))
    for i in range(n): H = H + emb(X, i, n)
    return H


def hsimuq(n):
    x = sp.Symbol("x"); qs = QSystem(); q = [Qubit(qs) for _ in range(n)]
    H = x*sum((q[i].Z*q[i+1].Z for i in range(n-1)), 0*q[0].Z)
    for i in range(n): H = H + q[i].X
    return H, "x"


def main():
    OBS = zz(0, 1, NQ); PSIn = qp.tensor([qp.basis(2, 0)] * NQ)
    clean = NoisyQuTiPRunner(NQ, noise=None)
    noisy = NoisyQuTiPRunner(NQ, noise=NoiseModel(n_qubits=NQ, T2=T2))
    H, var = hsimuq(NQ)
    fc = lambda th: float(qp.expect(OBS, (-1j*hq(th, NQ)*T).expm()*PSIn).real)
    fnz = noisy.make_expectation_fn(PSIn, OBS)
    fn = lambda th: fnz([[H.set_parameterizedHam({var: float(th)}), T]])

    xs = np.linspace(0.3, 1.8, 40)
    gi = np.array([(fc(x+1e-3)-fc(x-1e-3))/2e-3 for x in xs])
    x_star = float(xs[np.argmax(np.abs(gi))])
    g_ideal = (fc(x_star+1e-3)-fc(x_star-1e-3))/2e-3

    # analytic rescale factor (1/λ), computed from the IDEAL trajectory
    slope = ar.lambda_slope(lambda th: hq(th, NQ), OBS, PSIn, T, NQ,
                            z_sites=range(NQ), theta=x_star, n_grid=110)
    inv_lambda = ar.rescale_factor(slope, T, T2)        # = 1/λ_pred

    # PSR rescaled (ε-free), deterministic τ pool
    orig = np.random.rand
    np.random.rand = lambda k: (np.arange(k)+0.5)/k
    try:
        progs = observable_program_generator(H, T, n_sample=200, n_repetition=1,
                                             diff_var=var, value=x_star)
    finally:
        np.random.rand = orig
    pexp = noisy.make_expectation_fn(PSIn, OBS)
    g_psr = 0.0
    for H_tot, ug, _ in progs:
        bj = len(H_tot)//2
        e = np.array([pexp(H_tot[2*i]) for i in range(bj)])
        p = np.array([pexp(H_tot[2*i+1]) for i in range(bj)])
        g_psr += (T/bj)*float(ug)*np.sum(e-p)
    psr_res_bias = abs(g_psr*inv_lambda - g_ideal)

    # FD across ε: raw and rescaled, BIAS (∞ shots) and RMSE (finite shots)
    eps = np.geomspace(0.01, 1.0, 18)
    fd_bias, fd_res_bias, fd_res_rmse = [], [], []
    rng = np.random.default_rng(0)
    for e in eps:
        fp, fm = fn(x_star+e), fn(x_star-e)
        g_fd = (fp-fm)/(2*e)
        fd_bias.append(abs(g_fd - g_ideal))
        fd_res_bias.append(abs(g_fd*inv_lambda - g_ideal))
        # finite shots → RMSE of the RESCALED FD estimate
        nfd = N_SHOTS//2
        a = 2.0*rng.binomial(nfd, 0.5*(1+np.clip(fp, -1, 1)), size=R)/nfd-1
        b = 2.0*rng.binomial(nfd, 0.5*(1+np.clip(fm, -1, 1)), size=R)/nfd-1
        est = (a-b)/(2*e)*inv_lambda
        fd_res_rmse.append(float(np.sqrt(np.mean((est-g_ideal)**2))))
    fd_bias = np.array(fd_bias); fd_res_bias = np.array(fd_res_bias)
    fd_res_rmse = np.array(fd_res_rmse)

    print(f"{NQ}q TFIM x*={x_star:.3f}  g_ideal={g_ideal:+.4f}  1/λ_pred={inv_lambda:.3f}")
    print(f"PSR rescaled bias = {psr_res_bias:.4f}\n")
    print(f"{'ε':>7}{'FD raw bias':>13}{'FD RESCALED bias':>18}"
          f"{'FD resc RMSE@20k':>18}")
    for i, e in enumerate(eps):
        print(f"{e:>7.3f}{fd_bias[i]:>13.4f}{fd_res_bias[i]:>18.4f}"
              f"{fd_res_rmse[i]:>18.4f}")
    print(f"\n→ FD rescaled bias at smallest ε = {fd_res_bias[0]:.4f}  "
          f"(attenuation IS removed; only finite-ε truncation remains)")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    axA.loglog(eps, np.clip(fd_bias, 1e-5, None), "s-", color="#7b1fa2", lw=2.2,
               label="FD raw (attenuation + truncation)")
    axA.loglog(eps, np.clip(fd_res_bias, 1e-5, None), "^-", color="#ef6c00", lw=2.2,
               label="FD RESCALED (attenuation removed; truncation×1/λ remains)")
    axA.axhline(psr_res_bias, color="#00897b", ls="-", lw=2.4,
                label=f"PSR rescaled (ε-free) = {psr_res_bias:.4f}")
    axA.set_xlabel(r"FD step size $\varepsilon$")
    axA.set_ylabel("gradient bias |est − ideal| (∞ shots)")
    axA.set_title("(A) BIAS: rescale REMOVES FD's attenuation (→0 as ε→0),\n"
                  "but cannot remove finite-ε truncation (it amplifies it ×1/λ)")
    axA.legend(frameon=False, fontsize=8, loc="lower right")
    axA.grid(True, which="both", alpha=0.12)

    axB.loglog(eps, fd_res_rmse, "^-", color="#ef6c00", lw=2.2,
               label=f"FD rescaled, {N_SHOTS//1000}k shots (U-shape: variance×(1/λ)²)")
    axB.axhline(psr_res_bias, color="#00897b", ls="-", lw=2.4,
                label="PSR rescaled bias floor (≈0)")
    axB.set_xlabel(r"FD step size $\varepsilon$")
    axB.set_ylabel("RMSE to ideal gradient")
    axB.set_title("(B) FINITE SHOTS: rescale AMPLIFIES FD's variance — the ε-dilemma\n"
                  "(small ε: variance; large ε: truncation) survives the rescale")
    axB.legend(frameon=False, fontsize=8, loc="lower left")
    axB.grid(True, which="both", alpha=0.12)

    fig.suptitle("Is FD correctable?  Its ATTENUATION is (same 1/λ as PSR, since "
                 "FD-best=PSR-raw=λ·g_ideal) — but the rescale does NOT rescue FD's "
                 "ε-dilemma.\nPSR is ε-free, so the SAME rescale lands it on the true "
                 "gradient.", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "figures", "fd_correctable.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
