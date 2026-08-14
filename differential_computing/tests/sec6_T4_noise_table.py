"""
sec6_T4_noise_table.py — P0-B: T4, the calibrated Hamiltonian-level noise model.

Single source of truth for every noise number in Section 6.  These are the channels
of noise_model.py (+ the parameter-level control error δ), applied by NoisyQuTiPRunner
at the segment level.  It is a LITERATURE-CALIBRATED model, NOT a device calibration
file — provenance strings marked (best-guess) are defaults from the codebase, to be
confirmed.  KEY ASSUMPTION (flagged): every rate is θ-INDEPENDENT — this is what makes
the compiled PSR/NSR estimator unbiased for ∇C_noisy (T1's noisy-gradient cell).

Writes T4.csv and renders figures/sec6_T4_noise_table.png.
Run: conda run -n qec_pg python differential_computing/tests/sec6_T4_noise_table.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
OUTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# columns: channel, operator, rate (default), source, segments applied, θ-indep, note
ROWS = [
    ["dressed T2* dephasing", "Z_i  (sqrt(γφ/2))", "γφ = 1/T2*,  T2* = 5.0 μs",
     "dressed-state T2*/scattering (literature, Evered/Bluvstein-class)",
     "interaction / dressed-evolution segments", "yes",
     "regime set by T/T2* (0.15 headline, 0.5 appendix); T=T/T2*·T2*"],
    ["T1 relaxation", "σ⁻_i  (sqrt(1/T1))", "1/T1,  T1 = ∞ (OFF)",
     "amplitude damping (long-clock)",
     "all segments (1q / halt / transport)", "yes",
     "off by default in Sec 6 figs; long-clock price of Sec 5.2 is dephasing-dominated"],
    ["gate-channel error, 1q (Z-type)", "coh. exp(-iθZ) + incoh. Z-dephasing",
     "ε_1q = 1e-4  (99.99% 1q fidelity, best-guess)",
     "Evered et al. 2026 (arXiv:2604.25987); 1q ≈ 0",
     "single-qubit gate-insertion segments", "yes",
     "Z/phase-dominated (Doppler, |r'>), X≈0; magnitude best-guess"],
    ["gate-channel error, 2q (Z-type)", "coh. exp(-iθZ) + incoh. Z-dephasing",
     "ε_2q = 1e-3  (99.9% 2q fidelity, best-guess)",
     "Evered et al. 2026 (arXiv:2604.25987)",
     "two-qubit (ZZ) gate-insertion segments", "yes",
     "coherent fraction = 0.5 of ε (Doppler/laser-phase); rest incoherent Z"],
    ["post-selected leakage", "Γ|1><1|_i (no-jump, renorm.)",
     "Γ = 0 (OFF)",
     "dressed |1> loss to dark ground sublevel",
     "dressed segments", "yes",
     "off in Sec 6 gradient figs; conditional (post-selected) evolution when on"],
    ["control setpoint error δ", "additive θ-offset ~ N(0, r)",
     "r = 0.02  (BEST-GUESS)",
     "control resolution — NOT a device calibration; default from landscape_device.py",
     "programmed θ setpoint: FD's θ±ε, PSR/NSR base θ", "yes",
     "floors FD's ε and enters as δ/ε; ε-free PSR/NSR see only a 2nd-order shift"],
]
HEADER = ["channel", "operator", "rate (default)", "source", "segments applied",
          "θ-independent", "note"]


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "T4.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(HEADER); w.writerows(ROWS)
    print("wrote", os.path.join(OUTDIR, "T4.csv"))

    fig, ax = plt.subplots(figsize=(13.5, 3.4)); ax.axis("off")
    show = [[r[0], r[1], r[2], r[4], r[5]] for r in ROWS]   # drop source/note cols for width
    cols = ["channel", "operator", "rate (default)", "segments", "θ-indep"]
    t = ax.table(cellText=show, colLabels=cols, cellLoc="left", loc="center",
                 colWidths=[0.22, 0.24, 0.20, 0.28, 0.08])
    t.auto_set_font_size(False); t.set_fontsize(7.6); t.scale(1, 1.7)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#eeeeee"); cell.set_text_props(weight="bold")
    ax.set_title("T4 — calibrated Hamiltonian-level noise model (single source of truth for Sec 6)\n"
                 "all rates θ-INDEPENDENT (⇒ PSR/NSR unbiased for ∇C_noisy);  full source/notes in T4.csv;"
                 "  values marked best-guess pending calibration confirmation", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "sec6_T4_noise_table.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/sec6_T4_noise_table.png")


if __name__ == "__main__":
    main()
