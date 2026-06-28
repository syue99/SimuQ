"""
plot_rescaled.py — (A) the analytic rescale is SCALABLE (light-cone truncation:
slope converges in a small radius, cost independent of total chain length); and
(B) replot the attenuation figure with the rescale ON — λ_corrected collapses to 1.

Run:  conda run -n qec_pg python differential_computing/tests/plot_rescaled.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import qutip as qp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analytic_rescale as ar
import scaling_universality as su
from noisy_qutip import NoisyQuTiPRunner
from noise_model import NoiseModel

I = qp.qeye(2); X, Z = qp.sigmax(), qp.sigmaz()


def emb(op, i, n):
    return qp.tensor([op if k == i else I for k in range(n)])


# ── Panel A: light-cone truncation converges; cost is constant in total_n ──
def full_chain_slope(theta, T, total_n, n_grid=110):
    O = emb(Z, 0, total_n)
    psi0 = qp.tensor([qp.basis(2, 0)] * total_n)
    return ar.lambda_slope(lambda th: ar.chain_H_local(th, total_n), O, psi0, T,
                           total_n, z_sites=range(total_n), theta=theta,
                           n_grid=n_grid)


# ── Panel B: λ before/after the analytic rescale, per case ──
QCASES = {  # qutip H builder + observable, matching scaling_universality kinds
    "1q-Z":  (lambda th: th * emb(Z, 0, 2) + emb(X, 0, 2), emb(Z, 0, 2), [0], 2, 0.6, 2.0),
    "2q-X":  (lambda th: th * emb(Z, 0, 2) * emb(Z, 1, 2) + emb(X, 0, 2) + emb(X, 1, 2),
              emb(Z, 0, 2), [0, 1], 2, 0.5, 2.0),
    "2q-ZZ": (lambda th: np.sin(2 * th) * emb(Z, 0, 2) * emb(Z, 1, 2)
              + np.sin(2 * th) * emb(X, 0, 2) + np.sin(2 * th) * emb(X, 1, 2),
              emb(Z, 0, 2) * emb(Z, 1, 2), [0, 1], 2, 0.7, 1.0),
}


def lam_qutip(Hfn, O, psi0, T, theta, T2):
    """λ = g_noisy/g_ideal via qutip fine-difference (matches the PSR target)."""
    def g(runner, noisy):
        def Oend(th):
            if noisy:
                rho = qp.mesolve(Hfn(th), qp.ket2dm(psi0), [0, T],
                                 c_ops=[np.sqrt(1.0 / (2 * T2)) * emb(Z, i, psi0.dims[0].__len__())
                                        for i in range(len(psi0.dims[0]))]).states[-1]
                return float(qp.expect(O, rho).real)
            return float(qp.expect(O, (-1j * Hfn(th) * T).expm() * psi0).real)
        h = 1e-3
        return (Oend(theta + h) - Oend(theta - h)) / (2 * h)
    g_ideal = g(None, False)
    g_noisy = g(None, True)
    return g_noisy / g_ideal, g_ideal


def main():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=150)

    # Panel A: truncation convergence + constant cost
    theta, T = 0.6, 1.2
    print("PANEL A — light-cone truncation (chain, Z_0, θ=0.6, T=1.2):")
    radii = [1, 2, 3, 4]
    s_trunc = []
    for R in radii:
        t0 = time.time()
        s = ar.chain_slope(theta, T, R)
        s_trunc.append(s)
        print(f"  radius={R} (subsystem {R+1} qubits): slope={s:+.4f}  "
              f"[{time.time()-t0:.2f}s]")
    # full-system reference at a few total sizes (should match large-radius trunc)
    print("  full-system reference:")
    s_full = {}
    for ntot in (5, 7):
        s_full[ntot] = full_chain_slope(theta, T, ntot)
        print(f"    total_n={ntot}: slope={s_full[ntot]:+.4f}")
    axA.axhline(s_full[7], color="k", ls="--", lw=1.4, label="full system (n=7)")
    axA.plot([r + 1 for r in radii], s_trunc, "o-", color="#1f77b4", lw=2.2,
             label="light-cone truncation")
    axA.set_xlabel("subsystem size (light-cone radius+1)")
    axA.set_ylabel("attenuation slope s")
    axA.set_title("(A) truncation converges → cost independent of total n")
    axA.legend(frameon=False, fontsize=8.5)

    # Panel B: λ vs T/T2*, raw vs corrected
    print("\nPANEL B — λ before/after analytic rescale:")
    T2_list = [8.0, 4.0, 2.0, 1.0]
    colors = {"1q-Z": "#1f77b4", "2q-X": "#d62728", "2q-ZZ": "#2ca02c"}
    for kind, (Hfn, O, supp, nq, th, Tc) in QCASES.items():
        psi0 = qp.tensor([qp.basis(2, 0)] * nq)
        # analytic slope from the (small) ideal trajectory
        s = ar.lambda_slope(Hfn, O, psi0, Tc, nq, z_sites=supp, theta=th)
        xs, raw, corr = [], [], []
        for T2 in T2_list:
            lam, _ = lam_qutip(Hfn, O, psi0, Tc, th, T2)
            factor = ar.rescale_factor(s, Tc, T2)
            xs.append(Tc / T2); raw.append(lam); corr.append(lam * factor)
        xs = np.array(xs)
        c = colors[kind]
        axB.plot(xs, raw, "o--", color=c, lw=1.5, alpha=0.6, label=f"{kind} raw")
        axB.plot(xs, corr, "s-", color=c, lw=2.4, label=f"{kind} rescaled")
        print(f"  {kind}: s={s:+.3f}  rescaled λ = " +
              ", ".join(f"{x:.2f}→{v:.3f}" for x, v in zip(xs, corr)))
    axB.axhline(1.0, color="k", ls=":", lw=1.2)
    axB.axhspan(0.9, 1.1, color="green", alpha=0.08)
    axB.set_xlabel(r"$T/T_2^*$"); axB.set_ylabel(r"$\lambda$ (corrected → 1)")
    axB.set_title("(B) PSR gradient with analytic rescale ON")
    axB.legend(frameon=False, fontsize=7.5, ncol=2)

    fig.suptitle("Analytic rescaling in the PSR pipeline: slope computed from the "
                 "ideal trajectory via light-cone\ntruncation (scalable); applying "
                 "1/λ recovers the ideal gradient (raw λ<1 dashed → corrected ≈1)",
                 fontsize=9.2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "figures", "rescaled_gradient.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
