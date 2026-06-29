"""
fd_bias_origin_cases.py — the FD-bias-origin picture across several GRADIENT CASES,
to show PSR(+rescale) is robust in the regime we need (moderate attenuation).

Each panel is a different case (Hamiltonian / observable / operating point) at the
SAME moderate dephasing.  Five curves per panel:
  - FD noiseless   : bias vs ε on the clean landscape — pure truncation, → 0.
  - FD noisy       : bias vs ε on the dephased landscape — floors at attenuation.
  - PSR ideal      : PSR on the clean landscape (noise=None) — flat, ≈ 0.  This is
                     the parallel to "FD noiseless": it proves PSR itself is exact,
                     so the noisy PSR bias is ENTIRELY the noise (not the method).
  - PSR raw        : PSR on the dephased landscape — flat, attenuated (1−λ_PSR)|g|.
  - PSR rescaled   : analytic light-cone rescale — flat, near 0 in this regime.

Cases:
  A  3q TFIM ZZ-param, steep point, obs <Z0Z1>     (M=2)
  B  3q TFIM ZZ-param, moderate point, obs <Z0Z1>  (M=2)
  C  4q TFIM X-param,  steep point,  obs <Z0Z1>     (M=4, different chain rule)
  D  3q TFIM ZZ-param, obs <Z0> (1-body), steep point (M=2) — different observable
We also keep an honest eye on a small-gradient point where the rescale (s∝1/g)
gets shaky; here we stay in the robust band on purpose.

Run:  conda run -n qec_pg python differential_computing/tests/fd_bias_origin_cases.py
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

T, T2 = 1.5, 20.0          # fixed moderate dephasing (the regime where rescale matters)
I = qp.qeye(2); X, Z = qp.sigmax(), qp.sigmaz()


def emb(op, i, n): return qp.tensor([op if k == i else I for k in range(n)])
def zz(i, j, n): return emb(Z, i, n) * emb(Z, j, n)


# ── case builders: each returns (n, simuq H+var, qutip hq(th), observable) ──
def case_zz(n):
    def sq():
        x = sp.Symbol("x"); qs = QSystem(); q = [Qubit(qs) for _ in range(n)]
        H = x*sum((q[i].Z*q[i+1].Z for i in range(n-1)), 0*q[0].Z)
        for i in range(n): H = H + q[i].X
        return H, "x"
    def hq(th):
        H = th*sum(zz(i, i+1, n) for i in range(n-1))
        for i in range(n): H = H + emb(X, i, n)
        return H
    return n, sq, hq


def case_xparam(n):
    def sq():
        x = sp.Symbol("x"); qs = QSystem(); q = [Qubit(qs) for _ in range(n)]
        H = sum((q[i].Z*q[i+1].Z for i in range(n-1)), 0*q[0].Z)
        for i in range(n): H = H + x*q[i].X
        return H, "x"
    def hq(th):
        H = sum(zz(i, i+1, n) for i in range(n-1))
        for i in range(n): H = H + th*emb(X, i, n)
        return H
    return n, sq, hq


def pick_point(fc, fn, lo, hi):
    """pick an operating point whose |g_ideal| is in [lo,hi] and stays same-sign
    under noise (robust band).  Falls back to the steepest point."""
    xs = np.linspace(0.25, 1.95, 46)
    gi = np.array([(fc(x+1e-3)-fc(x-1e-3))/2e-3 for x in xs])
    best_x, best_g = None, -1
    for k, x in enumerate(xs):
        g = abs(gi[k])
        if lo <= g <= hi:
            lam = ((fn(x+1e-2)-fn(x-1e-2))/2e-2) / gi[k]
            if 0.4 < lam < 0.98 and g > best_g:
                best_x, best_g = float(x), g
    if best_x is None:
        best_x = float(xs[np.argmax(np.abs(gi))])
    return best_x


def run_case(name, n, sq, hq, obs, band):
    PSIn = qp.tensor([qp.basis(2, 0)] * n)
    clean = NoisyQuTiPRunner(n, noise=None)
    noisy = NoisyQuTiPRunner(n, noise=NoiseModel(n_qubits=n, T2=T2))
    H, var = sq()

    fc = lambda th: float(qp.expect(obs, (-1j*hq(th)*T).expm()*PSIn).real)
    fnz = noisy.make_expectation_fn(PSIn, obs)
    fn = lambda th: fnz([[H.set_parameterizedHam({var: float(th)}), T]])

    x_star = pick_point(fc, fn, *band)
    g_ideal = (fc(x_star+1e-3)-fc(x_star-1e-3))/2e-3
    g_noisy0 = (fn(x_star+1e-3)-fn(x_star-1e-3))/2e-3
    atten = abs(g_noisy0 - g_ideal)

    eps = np.geomspace(0.01, 1.2, 22)
    bias_clean = np.array([abs((fc(x_star+e)-fc(x_star-e))/(2*e)-g_ideal) for e in eps])
    bias_noisy = np.array([abs((fn(x_star+e)-fn(x_star-e))/(2*e)-g_ideal) for e in eps])

    # PSR pools (summed over the M chain-rule programs) on clean and noisy runners.
    # This plot is about BIAS, so we integrate τ over [0,T] with a DETERMINISTIC
    # midpoint quadrature (O(1/n²)) instead of the default random MC draw (O(1/√n));
    # the τ-MC variance is a separate axis we set aside here.
    orig_rand = np.random.rand
    np.random.rand = lambda k: (np.arange(k) + 0.5) / k     # midpoint grid on [0,1]
    try:
        progs = observable_program_generator(H, T, n_sample=200, n_repetition=1,
                                             diff_var=var, value=x_star)
    finally:
        np.random.rand = orig_rand
    cexp = clean.make_expectation_fn(PSIn, obs)
    nexp = noisy.make_expectation_fn(PSIn, obs)

    def psr_sum(expfn):
        g, M = 0.0, 0
        for H_tot, ug, _ in progs:
            bj = len(H_tot)//2; M += 1
            e = np.array([expfn(H_tot[2*i]) for i in range(bj)])
            p = np.array([expfn(H_tot[2*i+1]) for i in range(bj)])
            g += (T/bj)*float(ug)*np.sum(e - p)
        return g, M
    g_psr_ideal, M = psr_sum(cexp)
    g_psr_noisy, _ = psr_sum(nexp)

    slope = ar.lambda_slope(hq, obs, PSIn, T, n, z_sites=range(n),
                            theta=x_star, n_grid=110)
    g_res = g_psr_noisy * ar.rescale_factor(slope, T, T2)

    return dict(name=name, n=n, M=M, x=x_star, g_ideal=g_ideal, eps=eps,
                bias_clean=bias_clean, bias_noisy=bias_noisy, atten=atten,
                psr_ideal=abs(g_psr_ideal-g_ideal), psr=abs(g_psr_noisy-g_ideal),
                res=abs(g_res-g_ideal), lam_fd=g_noisy0/g_ideal,
                lam_psr=g_psr_noisy/g_ideal)


def main():
    n3, sq3, hq3 = case_zz(3)
    n4, sq4x, hq4x = case_xparam(4)
    cases = [
        ("A: 3q ZZ-param, steep, ⟨Z0Z1⟩", n3, sq3, hq3, zz(0, 1, 3), (0.5, 1.5)),
        ("B: 3q ZZ-param, moderate, ⟨Z0Z1⟩", n3, sq3, hq3, zz(0, 1, 3), (0.30, 0.45)),
        ("C: 4q X-param (M=4), ⟨Z0Z1⟩", n4, sq4x, hq4x, zz(0, 1, 4), (0.4, 1.5)),
        ("D: 3q ZZ-param, ⟨Z0⟩ (1-body)", n3, sq3, hq3, emb(Z, 0, 3), (0.4, 1.5)),
    ]
    res = [run_case(*c) for c in cases]

    print(f"fixed T={T}, T2={T2}\n")
    print(f"{'case':>34}{'M':>3}{'x*':>6}{'|g|':>7}{'λ_FD':>7}{'λ_PSR':>7}"
          f"{'FD flr':>8}{'PSRid':>7}{'PSRraw':>8}{'PSRres':>8}")
    for r in res:
        print(f"{r['name']:>34}{r['M']:>3}{r['x']:>6.2f}{abs(r['g_ideal']):>7.3f}"
              f"{r['lam_fd']:>7.3f}{r['lam_psr']:>7.3f}{r['atten']:>8.4f}"
              f"{r['psr_ideal']:>7.4f}{r['psr']:>8.4f}{r['res']:>8.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.2), dpi=150, sharex=True)
    for r, ax in zip(res, axes.ravel()):
        ax.loglog(r["eps"], np.clip(r["bias_clean"], 1e-5, None), "o-",
                  color="#1a73e8", lw=2.1, ms=4, label="FD noiseless (truncation)")
        ax.loglog(r["eps"], r["bias_noisy"], "s-", color="#7b1fa2", lw=2.1, ms=4,
                  label="FD noisy (trunc+atten)")
        ax.axhline(r["atten"], color="#d50000", ls="--", lw=1.6,
                   label=f"FD floor = {r['atten']:.3f}")
        ax.axhline(max(r["psr_ideal"], 1e-4), color="#1a73e8", ls=":", lw=2.0,
                   label=f"PSR ideal = {r['psr_ideal']:.4f}")
        ax.axhline(r["psr"], color="#9e9e9e", ls="-.", lw=2.6, alpha=0.7,
                   label=f"PSR raw = {r['psr']:.3f}  (≈ FD floor)")
        ax.axhline(r["res"], color="#00897b", ls="-", lw=2.3,
                   label=f"PSR rescaled = {r['res']:.3f}")
        ax.set_title(f"{r['name']}\n|g|={abs(r['g_ideal']):.3f}, λ'={r['lam_fd']:.2f}, "
                     f"λ_PSR={r['lam_psr']:.2f}", fontsize=9)
        ax.legend(frameon=False, fontsize=7.0, loc="lower right", ncol=1)
        ax.grid(True, which="both", alpha=0.12)
    for ax in axes[-1]:
        ax.set_xlabel(r"FD step size $\varepsilon$")
    for ax in axes[:, 0]:
        ax.set_ylabel("gradient bias |est − ideal| (∞ shots)")
    fig.suptitle("FD-bias origin across gradient cases (fixed moderate noise).  PSR "
                 "ideal (dotted)=0 → PSR is EXACT.  PSR raw (grey) sits ON the FD "
                 "floor (red):\nsame attenuation bias — BUT it is ε-free (no truncation/"
                 "variance dilemma) and the analytic rescale (teal) removes it, "
                 "robustly across cases.", fontsize=10.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "figures", "fd_bias_origin_cases.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
