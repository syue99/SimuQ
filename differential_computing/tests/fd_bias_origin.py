"""
fd_bias_origin.py — where does FD's bias come from: finite ε, or the noise?

FD (infinite shots) = central difference of the landscape it measures.  Two
independent biases vs the IDEAL gradient:
  - TRUNCATION (finite ε):  ~ε²·f'''/6  → vanishes as ε→0.
  - ATTENUATION (noise):    (1−λ')·|g_ideal|  → does NOT vanish as ε→0, because the
    noisy landscape's gradient λ'·g ≠ the ideal gradient.

We sweep ε and compute FD's bias vs g_ideal on (a) the NOISELESS landscape and
(b) the NOISY landscape.  Noiseless → 0 as ε→0 (pure truncation, slope 2).  Noisy
→ FLOORS at the attenuation as ε→0.  The floor (the gap at small ε) is the noise-
induced bias — that is what the bias_vs_size figure plots (evaluated at ε→0).

Run:  conda run -n qec_pg python differential_computing/tests/fd_bias_origin.py
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

    # a moderate-gradient point
    xs = np.linspace(0.3, 1.8, 40)
    gi = np.array([(fc(x+1e-3)-fc(x-1e-3))/2e-3 for x in xs])
    x_star = float(xs[np.argmax(np.abs(gi))])
    g_ideal = (fc(x_star+1e-3)-fc(x_star-1e-3))/2e-3
    g_noisy0 = (fn(x_star+1e-3)-fn(x_star-1e-3))/2e-3      # noisy gradient, ε→0
    atten = abs(g_noisy0 - g_ideal)
    lam = g_noisy0 / g_ideal

    eps = np.geomspace(0.01, 1.2, 22)
    bias_clean, bias_noisy = [], []
    for e in eps:
        fd_c = (fc(x_star+e)-fc(x_star-e))/(2*e)
        fd_n = (fn(x_star+e)-fn(x_star-e))/(2*e)
        bias_clean.append(abs(fd_c - g_ideal))
        bias_noisy.append(abs(fd_n - g_ideal))
    bias_clean = np.array(bias_clean); bias_noisy = np.array(bias_noisy)

    # PSR is ε-INDEPENDENT (kick-based) → flat lines.  Exact ∞-shot pool, summed
    # over the M chain-rule programs; then analytic light-cone rescale.
    np.random.seed(7)
    progs = observable_program_generator(H, T, n_sample=150, n_repetition=1,
                                         diff_var=var, value=x_star)
    pexp = noisy.make_expectation_fn(PSIn, OBS)
    g_psr = 0.0
    for H_tot, ug, _ in progs:
        bj = len(H_tot)//2
        e = np.array([pexp(H_tot[2*i]) for i in range(bj)])
        p = np.array([pexp(H_tot[2*i+1]) for i in range(bj)])
        g_psr += (T/bj)*float(ug)*np.sum(e - p)
    slope = ar.lambda_slope(lambda th: hq(th, NQ), OBS, PSIn, T, NQ,
                            z_sites=range(NQ), theta=x_star, n_grid=110)
    g_psr_res = g_psr * ar.rescale_factor(slope, T, T2)
    psr_bias = abs(g_psr - g_ideal)
    res_bias = abs(g_psr_res - g_ideal)
    lam_psr = g_psr / g_ideal

    print(f"{NQ}q TFIM, x*={x_star:.3f}, g_ideal={g_ideal:+.4f}, "
          f"g_noisy(ε→0)={g_noisy0:+.4f}, λ'={lam:.3f}")
    print(f"ATTENUATION bias (1−λ')|g| = {atten:.4f}  (the noise-induced shift, "
          f"independent of ε)")
    print(f"PSR raw  λ_PSR={lam_psr:.3f}  bias={psr_bias:.4f}  (flat: ε-independent)")
    print(f"PSR rescaled                  bias={res_bias:.4f}  (flat: ε-independent)\n")
    print(f"{'ε':>7}{'noiseless FD bias':>18}{'noisy FD bias':>15}")
    for i, e in enumerate(eps):
        print(f"{e:>7.3f}{bias_clean[i]:>18.4f}{bias_noisy[i]:>15.4f}")

    fig, ax = plt.subplots(figsize=(7.8, 5.2), dpi=150)
    ax.loglog(eps, bias_clean, "o-", color="#1a73e8", lw=2.4,
              label="FD on NOISELESS landscape  (pure truncation)")
    ax.loglog(eps, bias_noisy, "s-", color="#7b1fa2", lw=2.4,
              label="FD on NOISY landscape  (truncation + attenuation)")
    ax.axhline(atten, color="#d50000", ls="--", lw=1.8,
               label=f"FD attenuation floor (1−λ')|g| = {atten:.3f}")
    ax.axhline(psr_bias, color="#9e9e9e", ls="-.", lw=1.8,
               label=f"PSR raw bias (ε-indep) = {psr_bias:.3f}")
    ax.axhline(res_bias, color="#00897b", ls="-", lw=2.2,
               label=f"PSR rescaled bias (ε-indep) = {res_bias:.3f}")
    ax.loglog(eps, bias_clean[0]*(eps/eps[0])**2, ":", color="#1a73e8", lw=1,
              label=r"$\propto \varepsilon^2$ (truncation)")
    ax.set_xlabel(r"FD step size $\varepsilon$")
    ax.set_ylabel("gradient bias  |FD − ideal|  (∞ shots)")
    ax.set_title("Where FD's bias comes from: finite ε (truncation, →0) vs NOISE\n"
                 "(attenuation, a FLOOR that ε cannot remove).  At ε→0 only the "
                 "noise bias remains.")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.annotate("noise bias\n(does NOT vanish\nas ε→0)", xy=(eps[0], atten),
                xytext=(eps[0]*1.3, atten*2.2), fontsize=8, color="#d50000",
                arrowprops=dict(arrowstyle="->", color="#d50000"))
    fig.tight_layout()
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "figures", "fd_bias_origin.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
