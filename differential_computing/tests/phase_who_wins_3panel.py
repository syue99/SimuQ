"""
phase_who_wins_3panel.py — three (P, k) who-wins phase diagrams (region style, no
magnitude), Hamiltonian-level only (no SimuQ/rydberg2d/dressing).

Full-gradient shots for kick vs Nyquist over (#parameters P, #terms/param k) on a
7-qubit all-to-all two-local pool.  Same cost model as phase_shots_kick_vs_nyquist.
The balance is tilted by the tangent structure (foldability ρ=diam(A)/Σ|v|):

  (a) GENERAL           random two-local tangents (mixed ρ).
  (b) FAVOR KICK        ZZ-only tangents — commuting, aligned (ρ→2): Nyquist pays
                        the full diameter, so kick's co-located ± + reuse win.
  (c) FAVOR NYQUIST     HEISENBERG bonds (XX+YY+ZZ per edge) — strongly foldable
                        (ρ≪2): Nyquist's one combined shift beats per-term kicks.
All at high-entropy states (⟨σ⟩≈1.37, kick's WEAKEST co-located regime — so the
kick regions are conservative; polarized states would enlarge them).

Each panel just draws the two regions and the tie boundary (kick blue / Nyquist red).
Run: conda run -n qec_pg python differential_computing/tests/phase_who_wins_3panel.py
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import qutip as qp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NQ, T = 7, 1.0
FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
X, Y, Z, Iop = qp.sigmax(), qp.sigmay(), qp.sigmaz(), qp.qeye(2)
_IF = None


def IF():
    global _IF
    if _IF is None:
        _IF = qp.tensor([Iop] * NQ)
    return _IF


def op1(P, i):
    o = [Iop] * NQ; o[i] = P; return qp.tensor(o)


def op2(P, i, Q, j):
    o = [Iop] * NQ; o[i] = P; o[j] = Q; return qp.tensor(o)


def build_pool():
    """pool[k]=operator; edges[(i,j)] = [idx_XX, idx_YY, idx_ZZ]."""
    pool, edges = [], {}
    for i, j in itertools.combinations(range(NQ), 2):
        idx = []
        for P in (X, Y, Z):
            idx.append(len(pool)); pool.append(op2(P, i, P, j))
        edges[(i, j)] = idx
    return pool, edges, list(edges.keys())


def branch_sigma(pool, U_half, O, states):
    sig = np.zeros(len(pool))
    for psi in states:
        st = U_half * psi
        for k, Gj in enumerate(pool):
            kp = (IF() - 1j * Gj) / np.sqrt(2)
            km = (IF() + 1j * Gj) / np.sqrt(2)
            fp = float(qp.expect(O, U_half * (kp * st)).real)
            fm = float(qp.expect(O, U_half * (km * st)).real)
            sig[k] += np.sqrt(max(0.0, (1 - fp ** 2) + (1 - fm ** 2)))
    return sig / len(states)


def param_terms(kind, k, edgelist, edges, G, rng):
    """Return dict {generator_index: coef} for one parameter with ~k Pauli terms."""
    if kind == "random":
        idx = rng.choice(G, size=min(k, G), replace=False)
        return {int(j): float(rng.choice([-1.0, 1.0])) for j in idx}
    elif kind == "zz":                              # commuting Z-type → high ρ (aligned-ish)
        zz = [j for j in range(G) if j % 3 == 2]    # pool built X,Y,Z per edge
        idx = rng.choice(zz, size=min(k, len(zz)), replace=False)
        return {int(j): float(rng.choice([-1.0, 1.0])) for j in idx}
    else:  # "heisenberg": XX+YY+ZZ per edge (same sign) → foldable
        d = {}
        es = rng.permutation(len(edgelist))
        for e in es:
            for j in edges[edgelist[e]]:
                if len(d) >= k:
                    return d
                d[int(j)] = 1.0
        return d


def who_wins(kind, states_type, Ps, ks, seeds, pool, edges, edgelist, U_half, O, base_rng):
    G = len(pool)
    if states_type == "highent":
        states = []
        for _ in range(4):
            v = base_rng.normal(size=2 ** NQ) + 1j * base_rng.normal(size=2 ** NQ)
            v /= np.linalg.norm(v); states.append(qp.Qobj(v.reshape(-1, 1), dims=[[2] * NQ, [1] * NQ]))
    else:  # "polarized": product states (tensor of random single-qubit) — low entropy
        states = []
        for _ in range(4):
            kets = []
            for _q in range(NQ):
                a = base_rng.normal(size=2) + 1j * base_rng.normal(size=2); a /= np.linalg.norm(a)
                kets.append(qp.Qobj(a.reshape(-1, 1)))
            states.append(qp.tensor(kets))
    sig = branch_sigma(pool, U_half, O, states)
    Zg = np.zeros((len(ks), len(Ps)))
    for a, k in enumerate(ks):
        for b, P in enumerate(Ps):
            lr = []
            for s in range(seeds):
                rng = np.random.default_rng(97 * s + 3 * P + 11 * k)
                params = [param_terms(kind, k, edgelist, edges, G, rng) for _ in range(P)]
                N_nyq = 0.0
                for d in params:
                    A = sum(c * pool[j] for j, c in d.items())
                    e = A.eigenenergies(); N_nyq += (e[-1] - e[0]) ** 2
                nj = np.zeros(G)
                for d in params:
                    Sl = sum(abs(c) * sig[j] for j, c in d.items())
                    for j, c in d.items():
                        nj[j] = max(nj[j], Sl * abs(c) * sig[j])
                N_kick = 2.0 * nj.sum()
                lr.append(np.log10(N_nyq / max(N_kick, 1e-12)))
            Zg[a, b] = np.mean(lr)
    return Zg, sig.mean()


def draw(ax, Zg, Ps, ks, title):
    from scipy.ndimage import gaussian_filter
    Zs = gaussian_filter(Zg, sigma=0.8)            # smooth the tie boundary
    Pg, Kg = np.meshgrid(Ps, ks)
    ax.contourf(Pg, Kg, Zs, levels=[-100, 0, 100], colors=["#cfe3f2", "#f6d6c8"])
    ax.contour(Pg, Kg, Zs, levels=[0.0], colors="k", linewidths=1.8)
    # region labels
    if (Zg < 0).mean() > 0.05:
        ax.text(0.28, 0.86, "NSR\nwins", transform=ax.transAxes, color="#a0451a",
                fontsize=11, weight="bold", ha="center")
    if (Zg > 0).mean() > 0.05:
        ax.text(0.80, 0.18, "PSR\nwins", transform=ax.transAxes, color="#10507a",
                fontsize=11, weight="bold", ha="center")
    ax.set_xlabel("# parameters  P"); ax.set_ylabel("# terms per parameter  k")
    ax.set_title(title, fontsize=9)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    rng = np.random.default_rng(0)
    pool, edges, edgelist = build_pool(); G = len(pool)
    c = rng.normal(size=G) * 0.5
    B = sum(c[k] * pool[k] for k in range(G)) + sum(rng.normal() * 0.3 * op1(Z, i) for i in range(NQ))
    U_half = (-1j * B * (T / 2)).expm()
    O = sum(op1(Z, i) for i in range(NQ))

    Ps = list(range(1, 21)); ks = list(range(1, 15)); seeds = 6
    panels = [
        ("random", "highent", "(a) general\n(random two-local, $\\chi\\!\\sim\\!$mixed)"),
        ("zz", "highent", "(b) favors PSR\n(ZZ-only tangents → aligned, $\\chi\\!\\to\\!1$)"),
        ("heisenberg", "highent", "(c) favors NSR\n(Heisenberg bonds → foldable, $\\chi\\!\\ll\\!1$)"),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for ax, (kind, st, title) in zip(axs, panels):
        Zg, sm = who_wins(kind, st, Ps, ks, seeds, pool, edges, edgelist, U_half, O,
                         np.random.default_rng(1))
        draw(ax, Zg, Ps, ks, title + f"   ⟨σ⟩={sm:.2f}")
        print(f"{kind}/{st}: mean σ={sm:.3f}, Nyquist-win fraction={ (Zg<0).mean():.2f}")
    fig.suptitle("F3 — who needs fewer shots for the full gradient: PSR reuse vs NSR folding "
                 "(7q all-to-all two-local; Hamiltonian-level under T4 noise model; χ=diam(A)/2Σ|v|)",
                 fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIGDIR, "phase_who_wins_3panel.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"figure: {out}")


if __name__ == "__main__":
    main()
