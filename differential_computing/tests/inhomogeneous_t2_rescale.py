"""
inhomogeneous_t2_rescale.py — per-qubit T2*: does the rescale still recover
the gradient, and does modeling the inhomogeneity matter?

Physics: dephasing rates Γ_i = 1/(2·T2*_i) differ per qubit.  The first-order
response decomposes per qubit: g_noisy ≈ g_ideal + Σ_i Γ_i·X_iℓ with
X_iℓ = ∂(d⟨O⟩/dΓ_i)/∂θ_ℓ computed from the IDEAL trajectory (ar.dO_dGamma
with z_sites=[i]).  Predicted attenuation per parameter:

    λ_ℓ = exp( Σ_i Γ_i · X_iℓ / g_ℓ )        (inhomogeneity-AWARE)
vs  λ_ℓ = exp( Γ̄ · Σ_i X_iℓ / g_ℓ )          (NAIVE homogeneous, mean rate)

For a swap-symmetric system+observable X_0ℓ = X_1ℓ and the naive mean-rate
model is exact at first order — so this test uses the ASYMMETRIC C3 system
(θ1·Z0 + θ2·Z0Z1 + X-drive, obs Z0): qubit 0 carries the observable, the
responses differ, and mis-modeling the rate distribution bites.

Sweep the rate ratio r = Γ_0/Γ_1 at FIXED arithmetic-mean rate Γ̄ = 1/(2·10)
(nominal T/T2* = 0.15): r ∈ {1, 1.5, 2, 3, 4}.  Compare λ_exact (fine-ε on
the noisy landscape with per-qubit c_ops) against both predictions, and the
rescaled-gradient bias per model.  Lemma check (PSR pools) at r=3 — the
Lindblad-PSR lemma covers any θ-independent time-local noise, inhomogeneous
rates included; validated here explicitly.

Run:  conda run -n qec_pg python differential_computing/tests/inhomogeneous_t2_rescale.py
"""

import json
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

T = 1.5
GBAR = 1.0 / (2.0 * 10.0)          # mean rate: nominal T2*=10 → T/T2* = 0.15
RATIOS = [1.0, 1.5, 2.0, 3.0, 4.0]  # r = Γ_0/Γ_1 at fixed mean
POINT = (0.35, 0.5)                 # C3 moderate-|g| operating point
H_FD = 1e-3
POOL = 200

I2 = qp.qeye(2); X = qp.sigmax(); Z = qp.sigmaz()
Z0 = qp.tensor(Z, I2); ZZ = qp.tensor(Z, Z)
XD = qp.tensor(X, I2) + qp.tensor(I2, X)
OBS = Z0
PSI0 = qp.tensor(qp.basis(2, 0), qp.basis(2, 0))


def Hq(t1, t2):
    return t1 * Z0 + t2 * ZZ + 1.0 * XD


def rates(r):
    """(Γ_0, Γ_1) with arithmetic mean GBAR and ratio r."""
    g1 = 2.0 * GBAR / (1.0 + r)
    return r * g1, g1


def fclean(t1, t2):
    return float(qp.expect(OBS, (-1j * Hq(t1, t2) * T).expm() * PSI0).real)


def fnoisy(t1, t2, g0, g1):
    c_ops = [np.sqrt(g0) * qp.tensor(Z, I2), np.sqrt(g1) * qp.tensor(I2, Z)]
    rho = PSI0 * PSI0.dag()
    res = qp.mesolve(Hq(t1, t2), rho, [0.0, T], c_ops=c_ops)
    return float(qp.expect(OBS, res.states[-1]).real)


def grad2(f, t1, t2, h=H_FD):
    return np.array([(f(t1 + h, t2) - f(t1 - h, t2)) / (2 * h),
                     (f(t1, t2 + h) - f(t1, t2 - h)) / (2 * h)])


def per_qubit_X(t1, t2):
    """X[i][ell] = ∂(d<O>/dΓ_i)/∂θ_ell from the ideal trajectory."""
    Xm = np.zeros((2, 2))
    for i in range(2):
        def D(a, b, i=i):
            return ar.dO_dGamma(Hq(a, b), OBS, PSI0, T, 2, z_sites=[i],
                                n_grid=120)
        Xm[i, 0] = (D(t1 + H_FD, t2) - D(t1 - H_FD, t2)) / (2 * H_FD)
        Xm[i, 1] = (D(t1, t2 + H_FD) - D(t1, t2 - H_FD)) / (2 * H_FD)
    return Xm


def psr_pool_vector(g0, g1, t1, t2):
    """Raw PSR gradient via per-parameter pools under inhomogeneous noise."""
    c_ops = [np.sqrt(g0) * qp.tensor(Z, I2), np.sqrt(g1) * qp.tensor(I2, Z)]

    def expfn(H_list):
        rho = PSI0 * PSI0.dag()
        for k, (Hs, dur) in enumerate(H_list):
            Hqo = Hs.to_qutip_qobj()
            if k == 1:                      # kick = noiseless gate
                U = (-1j * Hqo * float(dur)).expm()
                rho = U * rho * U.dag()
            else:
                res = qp.mesolve(Hqo, rho, [0.0, float(dur)], c_ops=c_ops)
                rho = res.states[-1]
        return float(qp.expect(OBS, rho).real)

    grad = []
    for ell in range(2):
        qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
        s = sp.Symbol("v")
        if ell == 0:
            Hp = s * (q[0].Z * 1.0) + float(t2) * (q[0].Z * q[1].Z) \
                + 1.0 * (q[0].X + q[1].X)
            val = t1
        else:
            Hp = float(t1) * (q[0].Z * 1.0) + s * (q[0].Z * q[1].Z) \
                + 1.0 * (q[0].X + q[1].X)
            val = t2
        orig = np.random.rand
        np.random.rand = lambda k: (np.arange(k) + 0.5) / k
        try:
            progs = observable_program_generator(Hp, T, n_sample=POOL,
                                                 n_repetition=1, diff_var="v",
                                                 value=float(val))
        finally:
            np.random.rand = orig
        g = 0.0
        for H_tot, ug, _ in progs:
            b = len(H_tot) // 2
            em = np.array([expfn(H_tot[2 * i]) for i in range(b)])
            ep = np.array([expfn(H_tot[2 * i + 1]) for i in range(b)])
            g += (T / b) * float(ug) * np.sum(em - ep)
        grad.append(g)
    return np.array(grad)


def main():
    figdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    os.makedirs(figdir, exist_ok=True)
    cache = os.path.join(figdir, "inhomogeneous_t2_data.json")

    if os.path.exists(cache):
        d = json.load(open(cache))
        print("loaded cache — replotting only")
    else:
        t1, t2 = POINT
        g_true = grad2(fclean, t1, t2)
        Xm = per_qubit_X(t1, t2)
        print(f"point {POINT}: g_true = {g_true}")
        print(f"per-qubit responses X[i][ℓ]:\n{Xm}   "
              f"(asymmetry ratio X0/X1 per param: "
              f"{Xm[0, 0] / Xm[1, 0]:.2f}, {Xm[0, 1] / Xm[1, 1]:.2f})")

        rows = []
        for r in RATIOS:
            g0, g1 = rates(r)
            g_noisy = grad2(lambda a, b: fnoisy(a, b, g0, g1), t1, t2)
            lam_exact = g_noisy / g_true
            lam_aware = np.exp((g0 * Xm[0] + g1 * Xm[1]) / g_true)
            lam_naive = np.exp(GBAR * (Xm[0] + Xm[1]) / g_true)
            res_aware = g_noisy / lam_aware
            res_naive = g_noisy / lam_naive
            rel = lambda v: float(np.linalg.norm(v - g_true)
                                  / np.linalg.norm(g_true))
            rows.append(dict(
                r=r, g0=g0, g1=g1,
                lam_exact=list(lam_exact), lam_aware=list(lam_aware),
                lam_naive=list(lam_naive),
                bias_raw=rel(g_noisy), bias_aware=rel(res_aware),
                bias_naive=rel(res_naive)))
            print(f"  r={r}: λ_exact={lam_exact}  aware={lam_aware}  "
                  f"naive={lam_naive} | rel bias raw {rows[-1]['bias_raw']:.4f} "
                  f"aware {rows[-1]['bias_aware']:.4f} "
                  f"naive {rows[-1]['bias_naive']:.4f}", flush=True)

        # lemma check under inhomogeneous noise (r=3)
        g0, g1 = rates(3.0)
        g_pool = psr_pool_vector(g0, g1, t1, t2)
        g_fine = grad2(lambda a, b: fnoisy(a, b, g0, g1), t1, t2)
        print(f"lemma check r=3: pool {g_pool} vs fine-ε {g_fine}  "
              f"(max diff {np.max(np.abs(g_pool - g_fine)):.5f})")

        d = dict(point=list(POINT), g_true=list(map(float, g_true)),
                 X=[list(map(float, row)) for row in Xm], rows=rows,
                 lemma=dict(pool=list(map(float, g_pool)),
                            fine=list(map(float, g_fine))))
        json.dump(d, open(cache, "w"), default=float)
        print(f"cached: {cache}")

    rows = d["rows"]
    rs = [r["r"] for r in rows]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.2, 4.4), dpi=150)
    for ell, ls in ((0, "-"), (1, "--")):
        axA.plot(rs, [r["lam_exact"][ell] for r in rows], "ko",
                 ls="none", ms=6, mfc="none",
                 label="exact" if ell == 0 else None)
        axA.plot(rs, [r["lam_aware"][ell] for r in rows], ls, color="#00897b",
                 lw=2, label=f"aware θ{ell + 1}")
        axA.plot(rs, [r["lam_naive"][ell] for r in rows], ls, color="#d62728",
                 lw=1.6, alpha=0.8, label=f"naive (mean-rate) θ{ell + 1}")
    axA.set_xlabel(r"rate ratio  $r = \Gamma_0/\Gamma_1$  (mean fixed)")
    axA.set_ylabel(r"attenuation $\lambda_\ell$")
    axA.set_title("(A) predicted vs exact attenuation per parameter")
    axA.legend(frameon=False, fontsize=8)

    axB.semilogy(rs, [r["bias_raw"] for r in rows], "o--", color="#9e9e9e",
                 lw=1.8, label="raw PSR")
    axB.semilogy(rs, [r["bias_naive"] for r in rows], "s-", color="#d62728",
                 lw=2, label="rescaled, naive homogeneous model")
    axB.semilogy(rs, [r["bias_aware"] for r in rows], "o-", color="#00897b",
                 lw=2.4, label="rescaled, per-qubit aware")
    axB.set_xlabel(r"rate ratio  $r = \Gamma_0/\Gamma_1$  (mean fixed)")
    axB.set_ylabel("gradient-vector relative bias")
    axB.set_title("(B) rescaled bias vs T2* inhomogeneity")
    axB.legend(frameon=False, fontsize=8)
    fig.suptitle("Inhomogeneous per-qubit T2* (asymmetric 2q system, nominal "
                 "T/T2*=0.15): the per-qubit-aware rescale tracks the exact\n"
                 "attenuation at any spread; the homogeneous mean-rate model "
                 "degrades as the rate ratio grows", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = os.path.join(figdir, "inhomogeneous_t2.png")
    fig.savefig(out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
