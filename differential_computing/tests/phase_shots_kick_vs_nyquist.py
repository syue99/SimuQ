"""
phase_shots_kick_vs_nyquist.py — HYPOTHETICAL (no SimuQ compilation) shot-count
phase diagram: full-gradient shots for kick vs Nyquist over (#parameters, #terms).

Ideal Hamiltonian level only — H(θ)=Σ_j u_j(θ) H_j with two-local Pauli generators
on 6 qubits, all-to-all pool {XX,YY,ZZ}. No dressing / rydberg2d / AAIS.

Cost model (target gradient variance ε²; both ∝1/ε² and ∝T², so the RATIO is ε- and
T-independent — set to 1):
  Nyquist (no reuse):  N_ℓ = (diam(A_ℓ))²,  A_ℓ=Σ_j b_{jℓ}H_j.  N_nyq = Σ_ℓ N_ℓ.
      (stochastic estimator variance (2πK)²=(T·diam A)², state-independent — this is
       generous to kick: Nyquist could do better exploiting a state-dependent K_eff.)
  Kick (with reuse):   measure each generator's co-located ± branch ONCE (shared
      across ALL parameters). σ_j = sqrt((1−f₊,j²)+(1−f₋,j²)) (branch shot std).
      g_ℓ = Σ_j b_{jℓ}(f̄₋−f̄₊)_j, Var(g_ℓ)=Σ_j b_{jℓ}²σ_j²/n_j.  Optimal per-ℓ
      allocation gives N_ℓ=2S_ℓ², S_ℓ=Σ_j|b_{jℓ}|σ_j; reuse ⇒ n_j=max_ℓ(S_ℓ|b_{jℓ}|σ_j),
      N_kick = 2·Σ_j n_j.  (No-reuse baseline: Σ_ℓ 2S_ℓ².)

Sweep P (#params) × k (#terms/param), average over parameterization seeds & states.
Diagram: log10(N_nyq/N_kick) — blue Nyquist fewer shots, red kick fewer.
Run: conda run -n qec_pg python differential_computing/tests/phase_shots_kick_vs_nyquist.py
"""
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import qutip as qp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NQ, T = 6, 1.0
FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
X, Y, Z, Iop = qp.sigmax(), qp.sigmay(), qp.sigmaz(), qp.qeye(2)


def op1(P, i):
    ops = [Iop] * NQ; ops[i] = P; return qp.tensor(ops)


def op2(P, i, Q, j):
    ops = [Iop] * NQ; ops[i] = P; ops[j] = Q; return qp.tensor(ops)


def build_pool():
    pool = []
    for i, j in itertools.combinations(range(NQ), 2):
        pool += [op2(X, i, X, j), op2(Y, i, Y, j), op2(Z, i, Z, j)]
    return pool                                    # 15 edges × 3 = 45


def kick_sigma(pool, U_half, O, psi0):
    """σ_j = sqrt((1−f₊²)+(1−f₋²)) for each generator's co-located ± kick (τ=T/2)."""
    st = U_half * psi0
    sig = np.zeros(len(pool))
    for k, Gj in enumerate(pool):
        kp = (Iop_full() - 1j * Gj) / np.sqrt(2)   # exp(−i Gj π/4)  (Gj²=I)
        km = (Iop_full() + 1j * Gj) / np.sqrt(2)   # exp(+i Gj π/4)
        fp = float(qp.expect(O, U_half * (kp * st)).real)
        fm = float(qp.expect(O, U_half * (km * st)).real)
        sig[k] = np.sqrt(max(0.0, (1 - fp ** 2) + (1 - fm ** 2)))
    return sig


_IFULL = None
def Iop_full():
    global _IFULL
    if _IFULL is None:
        _IFULL = qp.tensor([Iop] * NQ)
    return _IFULL


def diam(A):
    e = A.eigenenergies()
    return float(e[-1] - e[0])


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    rng = np.random.default_rng(0)
    pool = build_pool(); G = len(pool)
    # one fixed random two-local system + extensive observable
    c = rng.normal(size=G) * 0.5
    B = sum(c[k] * pool[k] for k in range(G)) + sum(rng.normal() * 0.3 * op1(Z, i) for i in range(NQ))
    U_half = (-1j * B * (T / 2)).expm()
    O = sum(op1(Z, i) for i in range(NQ))

    # precompute branch stds σ_j, averaged over a few random states (reused everywhere)
    n_state = 4
    sig = np.zeros(G)
    for _ in range(n_state):
        v = rng.normal(size=2 ** NQ) + 1j * rng.normal(size=2 ** NQ)
        v /= np.linalg.norm(v)
        psi = qp.Qobj(v.reshape(-1, 1), dims=[[2] * NQ, [1] * NQ])
        sig += kick_sigma(pool, U_half, O, psi)
    sig /= n_state
    print(f"pool G={G}, mean branch std σ={sig.mean():.3f} (range {sig.min():.3f}–{sig.max():.3f})")

    Ps = [1, 2, 3, 4, 6, 8, 12, 16]
    ks = [1, 2, 3, 4, 6, 9, 12]
    seeds = 4
    ratio = np.zeros((len(ks), len(Ps)))
    for a, k in enumerate(ks):
        for b, P in enumerate(Ps):
            lr = []
            for s in range(seeds):
                r2 = np.random.default_rng(1000 * s + 7 * P + k)
                # parameterization: each of P params picks k generators, coef ±1
                bmat = np.zeros((G, P))
                for l in range(P):
                    idx = r2.choice(G, size=min(k, G), replace=False)
                    bmat[idx, l] = r2.choice([-1.0, 1.0], size=len(idx))
                # Nyquist total
                N_nyq = 0.0
                for l in range(P):
                    A = sum(bmat[j, l] * pool[j] for j in range(G) if bmat[j, l] != 0)
                    N_nyq += diam(A) ** 2
                # Kick total with reuse
                Sl = np.array([np.sum(np.abs(bmat[:, l]) * sig) for l in range(P)])  # S_ℓ
                nj = np.zeros(G)
                for l in range(P):
                    contrib = Sl[l] * np.abs(bmat[:, l]) * sig
                    nj = np.maximum(nj, contrib)
                N_kick = 2.0 * nj.sum()
                lr.append(np.log10(N_nyq / N_kick))
            ratio[a, b] = np.mean(lr)
        print(f"k={k:2d} done")

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    vmax = np.abs(ratio).max()
    im = ax.imshow(ratio, origin="lower", aspect="auto", cmap="RdBu",
                   vmin=-vmax, vmax=vmax,
                   extent=[Ps[0] - 0.5, Ps[-1] + 0.5, -0.5, len(ks) - 0.5])
    ax.set_xticks(Ps); ax.set_yticks(range(len(ks))); ax.set_yticklabels(ks)
    # contour of the tie line (ratio=0)
    Pg, Kg = np.meshgrid(Ps, range(len(ks)))
    ax.contour(np.array(Ps)[None, :].repeat(len(ks), 0), Kg, ratio, levels=[0.0],
               colors="k", linewidths=1.5)
    ax.set_xlabel("# parameters  P"); ax.set_ylabel("# terms per parameter  k")
    ax.set_title("Full-gradient shots: $\\log_{10}(N_{\\rm Nyquist}/N_{\\rm kick})$\n"
                 "(6 qubits, all-to-all two-local; RED = Nyquist wins, BLUE = kick wins)",
                 fontsize=9)
    ax.text(3.0, len(ks) - 1.6, "Nyquist wins\n(few params:\nfolds many terms/θ,\n"
            "no reuse to lose)", fontsize=7.5, color="white", ha="center", va="center")
    ax.text(13.5, 1.0, "kick wins\n(many params:\nbranch reuse\namortizes)",
            fontsize=7.5, color="#10307a", ha="center", va="center")
    cb = fig.colorbar(im); cb.set_label(r"$\log_{10}(N_{\rm Nyq}/N_{\rm kick})$  ($<0$: Nyquist fewer)")
    fig.tight_layout()
    out = os.path.join(FIGDIR, "phase_shots_kick_vs_nyquist.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    json.dump(dict(Ps=Ps, ks=ks, ratio=ratio.tolist(), sigma_mean=float(sig.mean())),
              open(os.path.join(FIGDIR, "phase_shots_kick_vs_nyquist.json"), "w"),
              indent=2, default=float)
    print(f"figure: {out}")


if __name__ == "__main__":
    main()
