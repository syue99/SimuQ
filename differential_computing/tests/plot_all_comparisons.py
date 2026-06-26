"""
plot_all_comparisons.py — one-page summary of every PSR-vs-FD comparison.

A results-table figure: each row is an experiment (regime), with the key metric,
PSR vs FD outcome, and the winner.  Numbers are the committed results from the
individual scripts (all under the corrected faithful kick=gate model and, where
applicable, the fair paired protocol).  Also writes the matching detail figures
list for reference.

Saves figures/all_comparisons_summary.png.

Run:  conda run -n qec_pg python differential_computing/tests/plot_all_comparisons.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (experiment, regime, metric, PSR, FD, winner)
ROWS = [
    ("Gradient correctness", "free shots, same start",
     "final energy", "= FD (|Δ|≤1e-4)", "= PSR", "tie (PSR exact)"),
    ("MaxCut free shots", "exact, noiseless",
     "cut (max 4)", "3.87 (n=5)", "3.86", "tie (PSR exact)"),
    ("Fair paired (H2)", "T2=2, equal budget",
     "energy gap", "0.010", "0.023", "PSR (13/15)"),
    ("Shot budget — low", "1152 shots/grad",
     "energy gap", "0.025", "0.141", "PSR (5.6x)"),
    ("Shot budget — high", "9600 shots/grad",
     "energy gap", "0.028", "0.032", "tie"),
    ("Dephasing — mild", "T/T2*=0.2",
     "energy gap", "0.008", "0.033", "PSR (8/8)"),
    ("Dephasing — strong", "T/T2*=1.0",
     "energy gap", "0.025", "0.146", "PSR (8/8)"),
    ("Optim. loop (settle)", "ε-dilemma sweep",
     "settle jitter", "0.000", "0.06-0.33", "PSR (no ε)"),
    ("H2 VQE small ε", "FD ε=0.01, shots",
     "energy gap", "0.005", "1.70 (stalls)", "PSR"),
    ("MaxCut small ε", "FD ε=0.01, shots",
     "cut deficit", "0.60", "2.30 (stalls)", "PSR"),
    ("Sharp landscape", "long T, floored ε",
     "sign accuracy", "100%", "25-78%", "PSR"),
]

NOTE = ("PSR vs FD across regimes (faithful kick=gate model; fair paired where "
        "applicable).\nPSR wins or ties everywhere; advantage grows at low shots, "
        "high noise, and sharp landscapes.\nFD's only edge: a tuned ε on smooth, "
        "high-shot landscapes. PSR needs no ε (the natural small ε always fails).")


def main():
    cols = ["experiment", "regime", "metric", "PSR", "FD", "winner"]
    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=150)
    ax.axis("off")
    table = ax.table(cellText=ROWS, colLabels=cols, loc="center",
                     cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.5)
    # style: header bold + shade; winner column color by PSR/FD/tie
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#34495e"); cell.set_text_props(color="white",
                                                               fontweight="bold")
        elif c == 5:
            w = ROWS[r - 1][5]
            cell.set_facecolor("#d6eaf8" if w.startswith("PSR")
                               else ("#fadbd8" if w.startswith("FD")
                                     else "#eaeded"))
            cell.set_text_props(fontweight="bold")
    ax.set_title("PSR vs FD — all comparison results", fontsize=13,
                 fontweight="bold", pad=14)
    fig.text(0.5, 0.06, NOTE, ha="center", fontsize=8.5, style="italic")
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "figures", "all_comparisons_summary.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"saved: {out}")
    print("\nDetail figures in differential_computing/figures/:")
    for fn in ["fair_protocol_comparison.png", "optimization_loop_epsilon.png",
               "h2_vqe_psr_vs_fd.png", "maxcut_psr_vs_fd.png",
               "optimization_loop.png", "fair_shots_dephasing.png"]:
        print(f"  - {fn}")


if __name__ == "__main__":
    main()
