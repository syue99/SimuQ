"""
bias_scaling_relative.py — RELATIVE gradient bias vs system size at T/T2* = 0.15
(the P1 "PSR's small bias scales with more qubits" figure; user-approved design
2026-07-01).

TFIM chain θ·ΣZ_iZ_{i+1} + ΣX_i, n=2..7, differentiate the GLOBAL coupling θ
(M = n-1 chain-rule terms).  Observables: local <Z0Z1> and extensive <ΣZZ>.
T = 1.5, T2 = 10  →  T/T2* = 0.15: double the C6 attenuation (FD floor ~20%
relative) while the first-order rescale stays quantitatively valid (~0.5%).
Same landscape family as C6 (scaling_advantage.py) and F1 (lightcone_collapse.py).

METRIC: relative bias |estimate − g_ideal| / |g_ideal|, INFINITE shots (exact
expectations), deterministic midpoint τ-quadrature — isolates BIAS; the λ-jitter
of operating-point selection largely divides out.

OPERATING POINT (engineered, no per-n jitter): fixed θ* = 0.5 for every n when
|g_ideal(θ*)| ≥ 0.25, else the nearest scanned θ that clears it.  θ*=0.5 is the
C6/F1 operating point (g ≈ −0.72 for n ≥ 4).

CURVES: FD best-ε (ε→0 on the noisy landscape) as a line, PSR raw as OPEN
markers overlaying it — visually identical BY THE LINDBLAD-PSR LEMMA (noiseless
Pauli kick + θ-independent dephasing + exact τ integral ⇒ raw analog PSR is an
unbiased estimator of the NOISY-landscape gradient, so its bias = FD's floor
exactly) — and PSR rescaled (analytic first-order correction, full-system slope).

Run:  conda run -n qec_pg python differential_computing/tests/bias_scaling_relative.py
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
from observable_program_generator import observable_program_generator
from noisy_qutip import NoisyQuTiPRunner
from noise_model import NoiseModel
import analytic_rescale as ar

T, T2 = 1.5, 10.0                 # T/T2* = 0.15 (user-approved regime)
POOL = 200                        # deterministic midpoint τ-quadrature points
NS = [2, 3, 4, 5, 6, 7]
THETA_STAR = 0.5                  # C6/F1 operating point; fallback scan if |g|<0.25
G_MIN = 0.25

I = qp.qeye(2); X, Z = qp.sigmax(), qp.sigmaz()


def emb(op, i, n): return qp.tensor([op if k == i else I for k in range(n)])
def zz(i, j, n): return emb(Z, i, n) * emb(Z, j, n)


def obs_local(n):
    return zz(0, 1, n)


def obs_extensive(n):
    return sum((zz(i, i + 1, n) for i in range(n - 1)), 0 * emb(Z, 0, n))


def sq_builder(n):
    def sq():
        x = sp.Symbol("x"); qs = QSystem(); q = [Qubit(qs) for _ in range(n)]
        H = x * sum((q[i].Z * q[i + 1].Z for i in range(n - 1)), 0 * q[0].Z)
        for i in range(n):
            H = H + q[i].X
        return H, "x"
    return sq


def hq_builder(n):
    def hq(th):
        H = th * sum(zz(i, i + 1, n) for i in range(n - 1))
        for i in range(n):
            H = H + emb(X, i, n)
        return H
    return hq


def pick_theta(n, obs, hq, psi0):
    """Fixed θ*=0.5; fall back to the nearest scanned θ with |g_ideal| ≥ G_MIN."""
    def g_ideal(th):
        f = lambda t: float(qp.expect(obs, (-1j * hq(t) * T).expm() * psi0).real)
        return (f(th + 1e-3) - f(th - 1e-3)) / 2e-3

    if abs(g_ideal(THETA_STAR)) >= G_MIN:
        return THETA_STAR
    cands = sorted(np.arange(0.2, 2.01, 0.05), key=lambda t: abs(t - THETA_STAR))
    for th in cands:
        if abs(g_ideal(float(th))) >= G_MIN:
            return float(th)
    return THETA_STAR


def run(n, obs_fn, label, gate_error=False):
    """gate_error adds a kick gate-error channel (Z-type, on the kicked pair) —
    a PSR-specific cost: FD runs no kick, so its landscape is untouched.  The
    rescale corrects dephasing attenuation only, NOT gate error.
    False = ideal kick; True = 99.9% 2q reference (eps=1e-3); a float = the 2q
    infidelity directly (e.g. 5e-4 for the cryo platform's 99.95%).  1q kicks
    use 1e-4 (99.99%) whenever gate error is on."""
    obs = obs_fn(n)
    psi0 = qp.tensor([qp.basis(2, 0)] * n)
    eps2 = None
    if gate_error:
        eps2 = 1e-3 if gate_error is True else float(gate_error)
    gate_kw = dict(gate_error_1q=1e-4, gate_error_2q=eps2) if eps2 else {}
    noisy = NoisyQuTiPRunner(n, noise=NoiseModel(n_qubits=n, T2=T2, **gate_kw))
    hq = hq_builder(n)
    H, var = sq_builder(n)()

    x_star = pick_theta(n, obs, hq, psi0)

    def fc(th):
        return float(qp.expect(obs, (-1j * hq(th) * T).expm() * psi0).real)
    fnz = noisy.make_expectation_fn(psi0, obs)
    def fn(th):
        return fnz([[H.set_parameterizedHam({var: float(th)}), T]])

    g_ideal = (fc(x_star + 1e-3) - fc(x_star - 1e-3)) / 2e-3
    g_fd = (fn(x_star + 1e-3) - fn(x_star - 1e-3)) / 2e-3      # FD best (ε→0)

    orig = np.random.rand
    np.random.rand = lambda k: (np.arange(k) + 0.5) / k        # deterministic τ
    try:
        progs = observable_program_generator(H, T, n_sample=POOL, n_repetition=1,
                                             diff_var=var, value=x_star)
    finally:
        np.random.rand = orig
    g_psr = 0.0
    for H_tot, ug, _ in progs:
        bj = len(H_tot) // 2
        e = np.array([fnz(H_tot[2 * i]) for i in range(bj)])
        p = np.array([fnz(H_tot[2 * i + 1]) for i in range(bj)])
        g_psr += (T / bj) * float(ug) * np.sum(e - p)

    slope = ar.lambda_slope(hq, obs, psi0, T, n, z_sites=range(n),
                            theta=x_star, n_grid=100)
    g_res = g_psr * ar.rescale_factor(slope, T, T2)

    ag = abs(g_ideal)
    r = dict(n=n, label=label, theta=x_star, g=g_ideal, lam=g_fd / g_ideal,
             fd_rel=abs(g_fd - g_ideal) / ag,
             psr_rel=abs(g_psr - g_ideal) / ag,
             res_rel=abs(g_res - g_ideal) / ag)
    print(f"  [{label:>9}] n={n} θ*={x_star:.2f} g={g_ideal:+.3f} λ={r['lam']:.3f} "
          f"FDrel={r['fd_rel']:.4f} PSRrawRel={r['psr_rel']:.4f} "
          f"PSRresRel={r['res_rel']:.4f}", flush=True)
    return r


def main():
    figdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    os.makedirs(figdir, exist_ok=True)
    cache = os.path.join(figdir, "bias_scaling_relative_data.json")

    # incremental cache: each completed (n, observable) is saved immediately,
    # so a killed run keeps its finished sizes and resumes where it stopped
    if os.path.exists(cache):
        d = json.load(open(cache)); loc, ext = d["loc"], d["ext"]
        print(f"loaded cache: local n={[r['n'] for r in loc]}, "
              f"extensive n={[r['n'] for r in ext]}")
    else:
        loc, ext = [], []

    def save():
        json.dump({"loc": loc, "ext": ext}, open(cache, "w"), default=float)

    for n in NS:
        if n not in [r["n"] for r in loc]:
            loc.append(run(n, obs_local, "local")); save()
        if n not in [r["n"] for r in ext]:
            ext.append(run(n, obs_extensive, "extensive")); save()
    print(f"cached: {cache}")

    # figure starts at n=3: the n=2 row sits at a fallback operating point
    # (θ*=1.10, λ=0.65 — |g(0.5)| below floor) and extensive ≡ local there;
    # it stays in the cache but not in the P1 plot (user decision 2026-07-02)
    N_PLOT_MIN = 3

    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=150)
    for res, c, mk, name in [(loc, "#7b1fa2", "s", "local ⟨Z0Z1⟩"),
                             (ext, "#e65100", "^", "extensive ⟨ΣZZ⟩")]:
        res = [r for r in res if r["n"] >= N_PLOT_MIN]
        nn = [r["n"] for r in res]
        ax.semilogy(nn, [r["fd_rel"] for r in res], mk + "-", color=c, lw=2.4,
                    label=f"FD best-ε — {name}")
        ax.semilogy(nn, [r["psr_rel"] for r in res], mk, color=c, ms=11,
                    mfc="none", mew=1.5, ls="none",
                    label=f"PSR raw — {name} (= FD, lemma)")
        ax.semilogy(nn, [r["res_rel"] for r in res], mk + "--", color=c, lw=1.8,
                    alpha=0.75, mfc="white",
                    label=f"PSR rescaled — {name}")
    ax.set_xlabel("chain length n")
    ax.set_ylabel("relative gradient bias  |estimate − ideal| / |ideal|   (∞ shots)")
    ax.set_title("Corrected PSR stays near-unbiased at every size (T/T2* = 0.15):\n"
                 "FD's best case floors at the attenuation; raw PSR shares that floor "
                 "exactly;\nonly the O(1)-cost analytic correction removes it",
                 fontsize=10.5)
    ax.set_xticks([n for n in NS if n >= N_PLOT_MIN])
    ax.grid(True, which="both", axis="y", alpha=0.15)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = os.path.join(figdir, "bias_scaling_relative.png")
    fig.savefig(out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
