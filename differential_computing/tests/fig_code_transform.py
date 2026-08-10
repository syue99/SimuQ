"""
fig_code_transform.py — Fig 2 (A2): the differential semantics as a CODE
TRANSFORMATION.  A concrete source analog observable program P(x,θ) is
transformed into a parameter-shift BRANCH program ∂P/∂θ: a stochastic time-split
τ + a ±kick on each generator H_j, giving a three-segment Hamiltonian sequence
per (j, s, τ) branch.  Schematic only (no compute).  Redrawn from scratch (no
SimuQ figure reuse).  Saves figures/fig_code_transform.png.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
MONO = dict(family="monospace", fontsize=10.5)
TEAL, ORANGE = "#00897b", "#e8710a"


def rbox(ax, x0, y0, x1, y1, fc, ec):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.006,rounding_size=0.018",
                                fc=fc, ec=ec, lw=1.5, zorder=1))


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.0, 7.4), dpi=150)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ── SOURCE program ──
    rbox(ax, 0.05, 0.76, 0.95, 0.965, "#eef4ff", "#4a72b0")
    ax.text(0.07, 0.95, "SOURCE  —  analog observable program  P(x, θ)",
            fontsize=11.5, fontweight="bold", va="top", color="#26456e")
    src = ("x, θ : real parameters\n"
           "J  = sin(2*x + θ)             # control coefficient  u_j(x,θ)\n"
           "H  = J * Z0*Z1  +  J * X0     # H(θ) = Hc + Σ_j u_j(θ,t) * H_j\n"
           "P  = ev(0, T, H) ;  meas M    # evolve for T, measure observable M")
    ax.text(0.08, 0.912, src, va="top", ha="left", **MONO)

    # ── transformation arrow ──
    ax.add_patch(FancyArrowPatch((0.30, 0.755), (0.30, 0.70), arrowstyle="-|>",
                                 mutation_scale=24, lw=2.4, color="#333"))
    ax.text(0.34, 0.727,
            r"$\partial/\partial\theta_\ell$   differential semantics"
            "\n(parameter-shift code transform, §2.4)",
            fontsize=10.5, va="center", ha="left", color="#333")

    # ── TRANSFORMED program ──
    rbox(ax, 0.05, 0.405, 0.95, 0.685, "#eafaf0", "#3a8f5e")
    ax.text(0.07, 0.672,
            r"TRANSFORMED  —  parameter-shift branch program  $\partial P/\partial\theta_\ell$",
            fontsize=11.5, fontweight="bold", va="top", color="#20623d")
    ax.text(0.085, 0.628,
            r"$\dfrac{\partial P}{\partial\theta_\ell}=T\,\mathbb{E}_{\tau\sim U(0,T)}"
            r"\ \sum_j \dfrac{\partial u_j}{\partial\theta_\ell}(\tau)"
            r"\ \sum_{s=\pm 1} s\ \cdot\ B_{j,s}(\tau)$",
            fontsize=13, va="top", ha="left")
    tr = ("branch  B[j,s](τ) :\n"
          "    ev(0,   τ,   H)               # 1. evolve to split time τ\n"
          "    ev(0, (1 + s*3/4)*π, H_j)     # 2. ±kick on generator H_j\n"
          "    ev(τ,   T,   H)               # 3. finish the evolution\n"
          "    meas M                        #    measure the branch")
    ax.text(0.085, 0.545, tr, va="top", ha="left", **MONO)

    # ── one-branch timeline ──
    ax.text(0.07, 0.35,
            "one branch  B[j,s](τ)  =  a three-segment Hamiltonian sequence:",
            fontsize=10.5, va="top", color="#333")
    y, h = 0.225, 0.055
    ax.add_patch(Rectangle((0.10, y), 0.34, h, fc=TEAL, ec="k", lw=1.2))
    ax.text(0.27, y + h / 2, "ev(0, τ, H)", ha="center", va="center", color="w",
            family="monospace", fontsize=10)
    ax.add_patch(Rectangle((0.44, y), 0.075, h, fc=ORANGE, ec="k", lw=1.2))
    ax.text(0.4775, y + h + 0.018, "kick H_j", ha="center", va="bottom",
            color=ORANGE, fontsize=9.5, fontweight="bold")
    ax.add_patch(Rectangle((0.515, y), 0.34, h, fc=TEAL, ec="k", lw=1.2))
    ax.text(0.685, y + h / 2, "ev(τ, T, H)", ha="center", va="center", color="w",
            family="monospace", fontsize=10)
    ax.text(0.87, y + h / 2, "meas M", ha="left", va="center", family="monospace",
            fontsize=10)
    # time axis
    ax.annotate("", xy=(0.86, y - 0.035), xytext=(0.10, y - 0.035),
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))
    for xx, lab in [(0.10, "0"), (0.4775, "τ"), (0.855, "T")]:
        ax.plot([xx, xx], [y - 0.045, y - 0.03], color="#555", lw=1.2)
        ax.text(xx, y - 0.06, lab, ha="center", va="top", fontsize=10)

    # ── gradient estimator line ──
    ax.text(0.5, 0.065,
            r"gradient estimate:  $\hat g_\ell=\frac{T}{K}\sum_{k}\sum_j"
            r"\frac{\partial u_j}{\partial\theta_\ell}(\tau_k)\,(f^-_k-f^+_k)$"
            r"    —  no step size $\varepsilon$; exact under the (noisy) dynamics",
            ha="center", va="center", fontsize=10, color="#444")

    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig_code_transform.png")
    fig.savefig(out); print(f"saved: {out}")


if __name__ == "__main__":
    main()
