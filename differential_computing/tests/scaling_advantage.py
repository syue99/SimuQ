"""
scaling_advantage.py (C6) — does the PSR(+rescale)-vs-FD gradient advantage SCALE?

Honest scaling claim for the intractable-regime argument.  For TFIM chains
θ·ΣZ_iZ_{i+1}+ΣX_i, n=2.., at fixed moderate dephasing, we plot the BIAS
(|estimate − ideal|, ∞ shots, deterministic τ) of FD-best-ε vs PSR-rescaled for
TWO observable families:

  LOCAL   O = Z0Z1            → light-cone bounded → FD floor SATURATES → the
                                advantage is SIZE-ROBUST (persists, doesn't grow).
  EXTENSIVE O = Σ Z_iZ_{i+1}  → light cone / #dephasing terms grows with n → FD
                                floor GROWS → the advantage GROWS with n.  (This is
                                the VQE-energy-gradient shape: a sum of local terms.)

Message: PSR+rescale bias stays ~flat and near zero for BOTH; FD's attenuation bias
is size-robust for local observables and GROWS for extensive ones.  So the gradient-
estimation advantage does not wash out with size — it persists or grows — which,
combined with the light-cone locality proof (lightcone_slope.py) that small n is
provably representative, is the scalable-advantage claim.

Run:  conda run -n qec_pg python differential_computing/tests/scaling_advantage.py
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

T, T2 = 1.5, 20.0
POOL = 200
NS = [2, 3, 4, 5, 6, 7]        # 7 is heavy (density-matrix mesolve); cap if it stalls
I = qp.qeye(2); X, Z = qp.sigmax(), qp.sigmaz()


def emb(op, i, n): return qp.tensor([op if k == i else I for k in range(n)])
def zz(i, j, n): return emb(Z, i, n) * emb(Z, j, n)


def obs_local(n):
    return zz(0, 1, n)


def obs_extensive(n):
    return sum((zz(i, i+1, n) for i in range(n-1)), 0*emb(Z, 0, n))


def sq_builder(n):
    def sq():
        x = sp.Symbol("x"); qs = QSystem(); q = [Qubit(qs) for _ in range(n)]
        H = x*sum((q[i].Z*q[i+1].Z for i in range(n-1)), 0*q[0].Z)
        for i in range(n): H = H + q[i].X
        return H, "x"
    return sq


def hq_builder(n):
    def hq(th):
        H = th*sum(zz(i, i+1, n) for i in range(n-1))
        for i in range(n): H = H + emb(X, i, n)
        return H
    return hq


def run(n, obs_fn, label):
    obs = obs_fn(n)
    PSIn = qp.tensor([qp.basis(2, 0)] * n)
    noisy = NoisyQuTiPRunner(n, noise=NoiseModel(n_qubits=n, T2=T2))
    hq = hq_builder(n); sq = sq_builder(n)
    H, var = sq()

    def fc(th):
        return float(qp.expect(obs, (-1j*hq(th)*T).expm()*PSIn).real)
    fnz = noisy.make_expectation_fn(PSIn, obs)
    def fn(th):
        return fnz([[H.set_parameterizedHam({var: float(th)}), T]])

    xs = np.linspace(0.2, 2.0, 40)
    gi = np.array([(fc(x+1e-3)-fc(x-1e-3))/2e-3 for x in xs])
    x_star, best = None, -1
    for k, x in enumerate(xs):
        if abs(gi[k]) > 0.25:
            lam = (fn(x+1e-2)-fn(x-1e-2))/2e-2 / gi[k]
            if 0.3 < lam < 0.98 and abs(gi[k]) > best:
                x_star, best = float(x), abs(gi[k])
    if x_star is None:
        x_star = float(xs[np.argmax(np.abs(gi))])
    g_ideal = (fc(x_star+1e-3)-fc(x_star-1e-3))/2e-3
    g_fd = (fn(x_star+1e-3)-fn(x_star-1e-3))/2e-3          # FD best (ε→0)

    orig = np.random.rand
    np.random.rand = lambda k: (np.arange(k)+0.5)/k       # deterministic τ (bias study)
    try:
        progs = observable_program_generator(H, T, n_sample=POOL, n_repetition=1,
                                             diff_var=var, value=x_star)
    finally:
        np.random.rand = orig
    pexp = noisy.make_expectation_fn(PSIn, obs)
    g_psr = 0.0
    for H_tot, ug, _ in progs:
        bj = len(H_tot)//2
        e = np.array([pexp(H_tot[2*i]) for i in range(bj)])
        p = np.array([pexp(H_tot[2*i+1]) for i in range(bj)])
        g_psr += (T/bj)*float(ug)*np.sum(e - p)
    slope = ar.lambda_slope(hq, obs, PSIn, T, n, z_sites=range(n),
                            theta=x_star, n_grid=100)
    g_res = g_psr * ar.rescale_factor(slope, T, T2)

    r = dict(n=n, label=label, g=abs(g_ideal), fd=abs(g_fd-g_ideal),
             psr=abs(g_psr-g_ideal), res=abs(g_res-g_ideal), lam=g_fd/g_ideal)
    print(f"  [{label:>9}] n={n} |g|={r['g']:.3f} λ={r['lam']:.3f} "
          f"FD={r['fd']:.4f} PSRraw={r['psr']:.4f} PSRres={r['res']:.4f}", flush=True)
    return r


def main():
    loc, ext = [], []
    for n in NS:
        try:
            loc.append(run(n, obs_local, "local Z0Z1"))
            ext.append(run(n, obs_extensive, "ext ΣZZ"))
        except MemoryError:
            print(f"  n={n}: MemoryError — stopping (density matrix too large)")
            break

    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=150)
    for res, c, mk, name in [(loc, "#7b1fa2", "s", "LOCAL ⟨Z0Z1⟩ (light-cone bounded)"),
                             (ext, "#e65100", "^", "EXTENSIVE ⟨ΣZ_iZ_{i+1}⟩")]:
        nn = [r["n"] for r in res]
        ax.plot(nn, [r["fd"] for r in res], mk+"-", color=c, lw=2.4,
                label=f"FD best-ε — {name}")
        ax.plot(nn, [r["res"] for r in res], mk+"--", color=c, lw=1.8, alpha=0.65,
                mfc="white", label=f"PSR rescaled — {name}")
    ax.set_xlabel("chain length n")
    ax.set_ylabel("gradient BIAS |estimate − ideal|  (∞ shots)")
    ax.set_title("Does the advantage scale?  FD's attenuation bias SATURATES for a "
                 "local\nobservable (size-robust gap) and GROWS for an extensive one; "
                 "PSR-rescaled stays ~0 for both")
    ax.set_xticks(NS)
    ax.legend(frameon=False, fontsize=8.2, loc="upper left")
    ax.text(0.98, 0.55, "small-n is provably representative via\nlight-cone locality "
            "(lightcone_slope.py): the\nlocal-observable gap at n=4-7 is the n→∞ gap.",
            transform=ax.transAxes, fontsize=7.6, color="#444", va="top", ha="right")
    fig.tight_layout()
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "figures", "scaling_advantage.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
