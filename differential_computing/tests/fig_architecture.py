"""
fig_architecture.py — Fig 1 (A1): full-stack overview, program -> atoms.

Top-to-bottom pipeline: a parametrized analog program is differentiated into
parameter-shift branch programs, compiled to a tweezer schedule + pulse ledger,
lowered to physical channels and AWG waveforms; the measured branch values are
assembled into the gradient off-device.  Two callouts: the pipeline runs at
scale (constant pulse depth) and the assembled gradient is exact for the
deployed noisy dynamics.  Schematic only (redrawn, no SimuQ reuse).
Saves figures/fig_architecture.png.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
X0, X1 = 0.06, 0.70                                   # box left/right
LAYERS = [
    ("#e8f0fe", "#2f5aa0", "PROGRAM",
     "H(θ) = Hc + Σⱼ uⱼ(θ,t)·Hⱼ ;  observable M",
     "Parametrized_Hamiltonian", "H(θ), M, T"),
    ("#e6f4ea", "#2f8f56", "DIFFERENTIAL TRANSFORM",
     "→ branch set { (j, s, τ) },  each a 3-segment H-list",
     "observable_program_generator", "branch H-lists"),
    ("#f3e8fd", "#7b3fb0", "TWEEZER COMPILATION",
     "schedule ops + PulseLedger;  zones: interaction / gate / idle,  AOD transport",
     "tweezer_mapper", "schedule + ledger"),
    ("#fff0e0", "#c9700f", "PHYSICAL CHANNELS",
     "5 fixed lines: TRANSPORT_AOD · ADDR_DET · ADDR_RABI · DRESSING_AOM · GATE_AOM",
     "physical_channels.to_physical", "comb tones"),
    ("#eeeeee", "#666666", "PULSES / AWG",
     "per-channel multi-tone waveforms",
     "PulseDSL", "measured  f⁻, f⁺  per branch"),
    ("#e8f0fe", "#2f5aa0", "GRADIENT ASSEMBLY",
     "ĝ_ℓ = (T/K) Σₖ Σⱼ (∂uⱼ/∂θ_ℓ)(τₖ) · (f⁻ₖ − f⁺ₖ)",
     "combine_gradient", None),
]


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 9.6), dpi=150)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    bh, gap = 0.118, 0.035
    top = 0.965
    ys = []
    for i, (fc, ec, name, body, mod, dat) in enumerate(LAYERS):
        y1 = top - i * (bh + gap); y0 = y1 - bh; ys.append((y0, y1))
        ax.add_patch(FancyBboxPatch((X0, y0), X1 - X0, bh,
                     boxstyle="round,pad=0.006,rounding_size=0.02",
                     fc=fc, ec=ec, lw=1.6, zorder=1))
        ax.text(X0 + 0.02, y1 - 0.026, name, fontsize=12, fontweight="bold",
                va="center", color=ec)
        ax.text(X0 + 0.02, y1 - 0.065, body, fontsize=9.3, va="center", color="#222")
        ax.text(X1 - 0.02, y0 + 0.02, mod, fontsize=8, va="center", ha="right",
                style="italic", color=ec, family="monospace")
        # arrow + data-interface label
        if dat is not None:
            ax.add_patch(FancyArrowPatch(((X0 + X1) / 2, y0),
                         ((X0 + X1) / 2, y0 - gap), arrowstyle="-|>",
                         mutation_scale=18, lw=2, color="#444"))
            ax.text((X0 + X1) / 2 + 0.02, y0 - gap / 2, dat, fontsize=8.3,
                    va="center", ha="left", style="italic", color="#444")

    # ── side callouts ──
    def callout(text, y, tgt_y, col):
        ax.add_patch(FancyBboxPatch((0.735, y - 0.05), 0.245, 0.10,
                     boxstyle="round,pad=0.006,rounding_size=0.02",
                     fc="white", ec=col, lw=1.3, zorder=2))
        ax.text(0.7575, y, text, fontsize=8.2, va="center", ha="left", color=col)
        ax.add_patch(FancyArrowPatch((0.735, y), (X1, tgt_y), arrowstyle="-|>",
                     mutation_scale=13, lw=1.3, color=col,
                     connectionstyle="arc3,rad=-0.2"))

    callout("constant pulse depth T at\nany n → runs where full-state\n"
            "simulation cannot (cost wall)", (ys[2][0] + ys[3][1]) / 2,
            (ys[3][0] + ys[3][1]) / 2, "#c9700f")
    callout("ĝ is the EXACT gradient of\nthe deployed NOISY program\n"
            "(soundness theorem, §4)", (ys[5][0] + ys[5][1]) / 2 + 0.005,
            (ys[5][0] + ys[5][1]) / 2, "#2f5aa0")

    ax.text(0.5, 0.018, "differentiable analog program  →  atoms  (one full-stack, "
            "differentiation-through-compilation pipeline)", ha="center",
            fontsize=9.5, color="#555", style="italic")
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig_architecture.png")
    fig.savefig(out); print(f"saved: {out}")


if __name__ == "__main__":
    main()
