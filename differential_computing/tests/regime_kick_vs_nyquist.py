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

    # (a) H=θZ+X, A=Z ⇒ ρ=2 (foldable). Nyquist cost = (2πK)² = (T·diam A)² = (2T)².
    #     Kick paired cost = 4T²(Σ|v|)²(1−f₊f₋), Σ|v|=1 ⇒ ratio = (1−f₊f₋).
    thetas = np.linspace(0.1, 2.4, 40)
    O, ratio, oneminus = [], [], []
    for th in thetas:
        Oz = expfn([[H.set_parameterizedHam({"x": float(th)}), T]])
        fp, fm = kick_branches(expfn, H, th)
        ff = fp * fm
        O.append(Oz); oneminus.append(1 - ff); ratio.append(1 - ff)   # ρ=2 ⇒ ratio=1−ff
    nwin = sum(1 for x in ratio if x < 1)
    print(f"H=θZ+X (ρ=2): kick/Nyquist ratio range {min(ratio):.2f}–{max(ratio):.2f}; "
          f"kick wins at {nwin}/{len(ratio)} θ (ratio<1, f₊f₋>0), loses at the rest "
          f"(f₊f₋<0). Landscape-dependent, O(1).")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    Oarr = np.array(O); rarr = np.array(ratio)
    axA.scatter(Oarr[rarr < 1], rarr[rarr < 1], c="#009E73", s=22, label="kick wins")
    axA.scatter(Oarr[rarr >= 1], rarr[rarr >= 1], c="#D55E00", s=22, label="Nyquist wins")
    axA.axhline(1.0, color="#999", lw=0.9, ls="--")
    axA.set_xlabel(r"$\langle Z\rangle(\theta)$")
    axA.set_ylabel(r"kick / Nyquist shot cost  $=1-f_+f_-$")
    axA.set_title(r"(a) foldable $A{=}Z$ ($\rho{=}2$): kick's both-branches is"
                  "\n" r"landscape-dependent — wins where $f_+f_->0$, loses where $<0$",
                  fontsize=8.3)
    axA.legend(fontsize=7.5); axA.grid(True, alpha=0.15)

    # (b) regime map: ρ vs 1−f₊f₋∈[0,2]. Boundary ρ = 2√(1−f₊f₋).
    c = np.linspace(0.0, 1.0, 200); rho_b = 2 * np.sqrt(c)
    axB.plot(rho_b, c, "k-", lw=1.5)
    axB.fill_betweenx(np.linspace(0, 2, 200), 0, 2, color="#0072B2", alpha=0.12)  # Nyquist base
    axB.fill_betweenx(c, rho_b, 2.0, color="#009E73", alpha=0.25)                  # kick wins
    axB.text(1.72, 0.12, "KICK wins\n(aligned +\ncorrelated)", color="#00695c",
             fontsize=8.5, ha="center", weight="bold")
    axB.text(0.55, 1.35, "Nyquist wins\n(non-foldable\nsubextensive, or\n$f_+f_-<0$)",
             color="#0072B2", fontsize=8.5, ha="center", weight="bold")
    axB.axvline(2.0, color="#333", lw=1.0, ls=":")
    axB.text(1.98, 1.85, "foldable\ntangents", fontsize=7, ha="right", color="#333")
    # real H=θZ+X points (all ρ=2), spread over y by the landscape:
    axB.scatter(np.full_like(rarr, 2.0) - 0.02, rarr, c=np.where(rarr < 1, "#009E73", "#D55E00"),
                s=14, zorder=5)
    axB.plot([0.5], [0.6], "^", color="#0072B2", ms=9, zorder=5)   # non-foldable subextensive
    axB.set_xlabel(r"$\rho=\mathrm{diam}(A)/\Sigma_j|v_j|$  (structural alignment)")
    axB.set_ylabel(r"$1-f_+f_-$  (branch decorrelation)")
    axB.set_title(r"(b) regime: kick wins iff $\rho>2\sqrt{1-f_+f_-}$"
                  "\n(dots = real $A{=}Z$ points, $\\rho{=}2$)", fontsize=8.3)
    axB.set_xlim(0, 2.08); axB.set_ylim(0, 2.0)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "regime_kick_vs_nyquist.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    json.dump(dict(thetas=list(thetas), O=O, ratio=ratio), open(
        os.path.join(FIGDIR, "regime_kick_vs_nyquist.json"), "w"), indent=2, default=float)
    print(f"figure: {out}")


if __name__ == "__main__":
    main()
