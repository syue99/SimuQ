"""
regime_kick_vs_nyquist.py — WHERE does kick save shots vs where does Nyquist?

Both cost ∼ (T·L1)² per gradient, but the two differ in TWO independent ways:

  structural   ρ = diam(A)/Σ_j|v_j| ∈ (0,2]   (Pauli involutions, diam(H_j)=2)
      Nyquist pays diam(A); kick pays Σ|v_j|diam(H_j) unless it FOLDS commuting /
      same-axis involutions. For any foldable tangent ρ=2 (equal L1s); ρ<2 only for
      non-foldable subextensive-diameter tangents.
  branch-corr  the kick's ± branches sit at the SAME θ (only the kick sign flips),
      so its paired single-shot estimator has 2nd moment (2−2f₊f₋), f₊f₋≈⟨O⟩² —
      SMALL near polarized states. Nyquist's ±s sit at far-apart points θ±s → no
      such reduction. This is kick's structural both-branches discount.

Cost ratio  kick/Nyquist = 4(1−f₊f₋)/ρ².  Kick wins ⟺ ρ > 2√(1−f₊f₋).

(a) verify on H=θZ+X (A=Z, foldable ρ=2): sweep θ, measure the kick branches'
    f₊f₋ and the resulting kick/Nyquist cost ratio — kick wins everywhere (ρ=2),
    more strongly toward polarized ⟨Z⟩.
(b) the regime map in (ρ, 1−f₊f₋): boundary ρ=2√(1−f₊f₋); kick-wins vs Nyquist-wins.

Run: conda run -n qec_pg python differential_computing/tests/regime_kick_vs_nyquist.py
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
from qutip_sequential import QuTiPSequentialRunner

T = 1.5
FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))


def kick_branches(expfn, H, theta):
    """f₊, f₋ for a Z-kick on H(θ)=θZ+X at midpoint τ=T/2 (deterministic)."""
    He = H.set_parameterizedHam({"x": float(theta)})
    qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    Zk = 1.0 * q[0].Z                                   # generator of the θ-term
    Zk_neg = -1.0 * q[0].Z
    tau = T / 2
    fm = expfn([[He, tau], [Zk, np.pi / 4], [He, T - tau]])       # −kick
    fp = expfn([[He, tau], [Zk_neg, np.pi / 4], [He, T - tau]])   # +kick (short)
    return fp, fm


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    x = sp.Symbol("x"); qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    H = x * q[0].Z + q[0].X
    r = QuTiPSequentialRunner(2, nsteps=200000)
    obs = qp.tensor(qp.sigmaz(), qp.qeye(2))
    expfn = r.make_expectation_fn(r.zero_state(), obs)

    # (a) H=θZ+X, A=Z ⇒ ρ=2. SHOT-NOISE COST = VARIANCE, not the 2nd moment.
    #   kick per-sample var = T²·Var(f̂₋−f̂₊) = T²[(1−f₊²)+(1−f₋²)] at 2 exec;
    #   Nyquist stochastic var = (2πK)² at 1 exec (each far-apart shift: E[f̂²]=1).
    #   (2πK)²=(2T)² for A=Z ⇒ ratio = 2T²[(1−f₊²)+(1−f₋²)]/(2T)² = varfac ≤ 1.
    thetas = np.linspace(0.1, 2.4, 40)
    O, ratio, varfac = [], [], []
    for th in thetas:
        Oz = expfn([[H.set_parameterizedHam({"x": float(th)}), T]])
        fp, fm = kick_branches(expfn, H, th)
        vf = ((1 - fp ** 2) + (1 - fm ** 2)) / 2       # ≈ 1−⟨O⟩², ∈[0,1]
        O.append(Oz); varfac.append(vf); ratio.append(vf)
    print(f"H=θZ+X (ρ=2): kick/Nyquist VARIANCE ratio {min(ratio):.2f}–{max(ratio):.2f} "
          f"(all ≤1 ⇒ KICK WINS everywhere); biggest win (→0) at polarized ⟨Z⟩, "
          f"tie (→1) at the equator.")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    Oarr = np.array(O); rarr = np.array(ratio)
    axA.scatter(Oarr, rarr, c="#009E73", s=22)
    axA.axhline(1.0, color="#999", lw=0.9, ls="--")
    axA.text(0.02, 1.03, "tie", fontsize=7, color="#666", transform=axA.get_yaxis_transform())
    axA.set_xlabel(r"$\langle Z\rangle(\theta)$")
    axA.set_ylabel(r"kick / Nyquist shot cost  ($\leq1\Rightarrow$ kick wins)")
    axA.set_title(r"(a) foldable $A{=}Z$ ($\rho{=}2$): ratio $\leq1$ everywhere"
                  "\n$=\\frac{1}{2}[(1{-}f_+^2){+}(1{-}f_-^2)]$ (kick co-located $\\pm$)",
                  fontsize=8.3)
    axA.set_ylim(0, 1.1); axA.grid(True, alpha=0.15)

    # (b) regime map: ρ vs varfac∈[0,1]. kick wins ⟺ ρ > 2√(varfac).
    c = np.linspace(0.0, 1.0, 200); rho_b = 2 * np.sqrt(c)
    axB.fill_betweenx(c, 0, 2.0, color="#0072B2", alpha=0.10)          # Nyquist base
    axB.fill_betweenx(c, rho_b, 2.0, color="#009E73", alpha=0.28)      # kick wins
    axB.plot(rho_b, c, "k-", lw=1.5)
    axB.text(1.55, 0.28, "KICK wins\n(aligned)", color="#00695c", fontsize=9, ha="center", weight="bold")
    axB.text(0.5, 0.75, "Nyquist wins\n(non-foldable\nsubextensive)", color="#0072B2",
             fontsize=9, ha="center", weight="bold")
    axB.axvline(2.0, color="#333", lw=1.0, ls=":")
    axB.text(1.98, 0.05, "foldable", fontsize=7, ha="right", color="#333")
    axB.scatter(np.full_like(rarr, 2.0) - 0.02, rarr, c="#009E73", s=14, zorder=5)  # all kick-wins
    axB.plot([0.5], [0.3], "^", color="#0072B2", ms=9, zorder=5)      # non-foldable subextensive
    axB.set_xlabel(r"$\rho=\mathrm{diam}(A)/\Sigma_j|v_j|$  (structural alignment)")
    axB.set_ylabel(r"kick branch shot variance $\in[0,1]$")
    axB.set_title(r"(b) regime: kick wins iff $\rho>2\sqrt{\mathrm{var}}$"
                  "\n(dots: real $A{=}Z$, $\\rho{=}2$ — ALL kick-wins)", fontsize=8.3)
    axB.set_xlim(0, 2.08); axB.set_ylim(0, 1.0)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "regime_kick_vs_nyquist.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    json.dump(dict(thetas=list(thetas), O=O, ratio=ratio), open(
        os.path.join(FIGDIR, "regime_kick_vs_nyquist.json"), "w"), indent=2, default=float)
    print(f"figure: {out}")


if __name__ == "__main__":
    main()
