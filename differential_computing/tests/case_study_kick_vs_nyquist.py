"""
case_study_kick_vs_nyquist.py — shots for kick vs Nyquist are the SAME L1
functional; who wins is set by diam(A) vs ‖v‖₁, not by any π² bookkeeping.

Per-shot variance (⇒ shots @ fixed ε_g):
  Nyquist (stochastic): (Σ|w_n|)² = (2πK)² = (T·diam(A))²      [L1 = T·diam(A)]
  Kick:                 (2T·Σ_j|v_j|diam(H_j))²·(1−f₊f₋)       [L1 = 2T·‖v‖₁·diamH]
Subadditivity diam(A) ≤ Σ_j|v_j|diam(H_j) ⇒ Nyquist ≤ kick, EQUAL when the
tangent terms are aligned (a tie), Nyquist ahead only when diam(A) is subextensive
relative to ‖v‖₁.

Three tangent families vs system size m (diam(A) computed exactly):
  A) uniform  A=Σ Z_j            → diam=2m, ‖v‖₁=m      → TIE (∝m²)
  B) rotated  A=Σ(cφ X_j+sφ Z_j) → diam=2m, kick decomposes to X&Z (‖v‖₁≈1.3m)
                                                       → Nyquist O(1) win (≤√2)
  C) telescope A=Σ(Z_j−Z_{j+1})  → diam=4 (O(1)), ‖v‖₁=2m
                                                       → Nyquist ∝m² win IFF kick
                                                         differentiates term-by-term
Plots relative shot cost ∝ L1² and prints the table.
Run: conda run -n qec_pg python differential_computing/tests/case_study_kick_vs_nyquist.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simuq import QSystem, Qubit
from nyquist_shift import spectral_diameter

T = 1.5
PHI = np.pi / 4
MS = list(range(2, 9))    # diam via exact eig; cap size (telescope uses m+1 qubits)
FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))


def norms(family, m):
    """Return (diam(A), kick_L1_terms) for a tangent family on m qubits.
    kick_L1_terms = Σ_j |v_j| diam(H_j) (the reach kick sums term-by-term)."""
    qs = QSystem(); q = [Qubit(qs) for _ in range(m + 1)]
    if family == "uniform":
        A = sum((q[j].Z for j in range(m)), 0 * q[0].Z)
        kick_terms = m * 2.0                                  # m Z-terms, diam(Z)=2
    elif family == "rotated":
        c, s = np.cos(PHI), np.sin(PHI)
        A = sum((c * q[j].X + s * q[j].Z for j in range(m)), 0 * q[0].Z)
        kick_terms = m * (abs(c) + abs(s)) * 2.0              # kick splits X_j & Z_j
    elif family == "telescope":
        A = sum((q[j].Z - q[j + 1].Z for j in range(m)), 0 * q[0].Z)
        kick_terms = 2 * m * 2.0                              # 2m Z-terms unfolded
    return spectral_diameter(A), kick_terms


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    fams = {"uniform": "A) uniform  ΣZ_j",
            "rotated": "B) rotated  Σ(cφX+sφZ)",
            "telescope": "C) telescope  Σ(Z_j−Z_{j+1})"}
    data = {}
    print(f"{'family':>10} {'m':>3} {'diam(A)':>8} {'kickΣ|v|dH':>11} "
          f"{'L1_nyq':>8} {'L1_kick':>8} {'kick/nyq':>9}")
    for fam in fams:
        rows = []
        for m in MS:
            diamA, kt = norms(fam, m)
            L1_nyq = T * diamA                                # 2πK = T·diam(A)
            L1_kick = T * kt                                  # T·Σ|v|diam(H) (both-branch factor ~absorbed)
            rows.append(dict(m=m, diamA=diamA, kick_terms=kt,
                             L1_nyq=L1_nyq, L1_kick=L1_kick,
                             cost_nyq=L1_nyq ** 2, cost_kick=L1_kick ** 2,
                             ratio=L1_kick ** 2 / L1_nyq ** 2))
            print(f"{fam:>10} {m:>3} {diamA:>8.1f} {kt:>11.1f} "
                  f"{L1_nyq:>8.2f} {L1_kick:>8.2f} {rows[-1]['ratio']:>9.2f}")
        data[fam] = rows

    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, (fam, title) in zip(axs, fams.items()):
        rows = data[fam]; m = [r["m"] for r in rows]
        ax.loglog(m, [r["cost_kick"] for r in rows], "o-", color="#009E73",
                  ms=5, label="kick  ∝ ‖v‖₁²")
        ax.loglog(m, [r["cost_nyq"] for r in rows], "^-", color="#0072B2",
                  ms=5, label="Nyquist  ∝ diam(A)²")
        ax.set_title(title, fontsize=8.5); ax.set_xlabel("qubits m")
        ax.grid(True, which="both", alpha=0.15)
        r = rows[-1]["ratio"]
        tag = "tie" if 0.7 < r < 1.5 else (f"Nyquist ×{r:.0f} cheaper" if r > 1.5 else "")
        ax.text(0.05, 0.9, tag, transform=ax.transAxes, fontsize=8, color="#333")
    axs[0].set_ylabel("relative shot cost  ∝ L1²")
    axs[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Kick vs Nyquist: same L1 functional — verdict = diam(A) vs ‖v‖₁ "
                 "(no π² advantage)", fontsize=9)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "case_study_kick_vs_nyquist.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    json.dump({"T": T, "phi": PHI, "data": data},
              open(os.path.join(FIGDIR, "case_study_kick_vs_nyquist.json"), "w"),
              indent=2, default=float)
    print(f"\nfigure: {out}")


if __name__ == "__main__":
    main()
