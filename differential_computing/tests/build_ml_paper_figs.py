"""
build_ml_paper_figs.py — render the ML-conference (local transfer map / rescale)
paper figures into paper_fig_ml/.

IDEAL-target framing: the estimand is ∇C_ideal (the noise-free gradient, i.e.
Hamiltonian-learning / inverse-simulation).  Raw PSR gives the exact *device*
gradient ∇C_noisy, which is attenuated vs ∇C_ideal; the analytic light-cone
transfer map ("rescale") is the ONLY estimator that converges to ∇C_ideal —
shots kill the variance, the transfer map kills the attenuation bias.

This is the complement of build_paper_figs.py (the DiffSimuQ PL+Sys paper, which
is device-target and drops rescale).  Same ACM style; reads JSON caches only.

  fig_shots       RMSE vs N: only PSR-corrected converges (~N^-1/2); FD-best &
                  PSR-raw floor at the attenuation bias.
  fig_decomp      bias/variance decomposition vs N (why: shots→variance,
                  transfer map→bias).
  fig_bias_n      corrected-PSR residual bias vs system size (light-cone
                  truncation stays flat; FD-best floors high).
  fig_gate_error  robustness of the transfer map to finite CZ gate error.

Run:  conda run -n qec_pg python differential_computing/tests/build_ml_paper_figs.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
OUTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "paper_fig_ml"))

# Okabe-Ito
C_FD = "#D55E00"       # vermillion — finite differences
C_RAW = "#8c8c8c"      # gray — raw PSR (= device gradient, attenuated)
C_RES = "#0072B2"      # blue — rescaled / transfer-map-corrected PSR
C_ALT = "#009E73"      # green — 99.95% variant / guides
C_ALT2 = "#E69F00"     # orange — 99.9% variant / extensive obs

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "font.family": "serif", "mathtext.fontset": "stix",
    "axes.linewidth": 0.7, "lines.linewidth": 1.6,
    "legend.frameon": False, "savefig.dpi": 300,
})
COL = 3.3


def load(name):
    return json.load(open(os.path.join(FIGDIR, name)))


def save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"),
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote paper_fig_ml/{name}.pdf/.png")


# ── shots scaling: only the transfer map converges to ∇C_ideal ───────────────

def fig_shots():
    d = load("shots_scaling_data.json")
    regime = d["T"] / d["T2"]; N = np.array(d["budgets"])
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    ax.loglog(N, d["fd_best"], "s-", color=C_FD, ms=4, label=r"FD, oracle-tuned $\varepsilon$")
    ax.loglog(N, d["psr_raw"], "o--", color=C_RAW, ms=4, label=r"PSR raw ($\nabla C_{\rm noisy}$)")
    ax.loglog(N, d["psr_res"], "o-", color=C_RES, ms=4, lw=2.0, label="PSR + transfer map")
    ax.loglog(N, np.array(d["psr_res"])[0] * (N / N[0]) ** -0.5, ":", color=C_ALT,
              lw=1.0, label=r"$N^{-1/2}$")
    ax.set_xlabel("total shots $N$")
    ax.set_ylabel(r"error vs $\nabla C_{\rm ideal}$ (RMSE)")
    ax.text(0.97, 0.94, rf"$T/T_2^*={regime:.2f}$", transform=ax.transAxes,
            fontsize=7, color="#555", ha="right")
    ax.legend(handlelength=1.6, loc="lower left")
    save(fig, "fig_shots_scaling")
    return dict(regime=regime, fd_floor=float(d["fd_best"][-1]),
                raw_floor=float(d["psr_raw"][-1]), resc_at_max=float(d["psr_res"][-1]))


def fig_decomp():
    d = load("shots_decomposition_data.json")
    regime = d["T"] / d["T2"]; N = np.array(d["budgets"])
    key_fd = [k for k in d["D"] if k.startswith("FD")][0]
    colors = {"PSR raw": C_RAW, "PSR rescaled": C_RES, key_fd: C_FD}
    labels = {"PSR raw": "PSR raw", "PSR rescaled": "PSR + transfer map",
              key_fd: key_fd.replace("ε", r"$\varepsilon$")}
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    for k, c in colors.items():
        ax.loglog(N, d["D"][k]["bias"], "-", color=c, lw=1.8, label=f"{labels[k]} — bias")
        ax.loglog(N, d["D"][k]["std"], "--", color=c, lw=1.0, alpha=0.75, label=f"{labels[k]} — std")
    ax.set_xlabel("total shots $N$")
    ax.set_ylabel("error component")
    ax.text(0.03, 0.05, rf"$T/T_2^*={regime:.2f}$", transform=ax.transAxes, fontsize=7, color="#555")
    ax.legend(ncol=2, handlelength=1.5, fontsize=6)
    save(fig, "fig_shots_decomposition")
    return dict(regime=regime)


def fig_bias_n():
    d = load("bias_scaling_relative_data.json")
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    for key, mk, obs in (("loc", "s", r"local $\langle Z_0Z_1\rangle$"),
                         ("ext", "^", r"extensive $\langle\Sigma ZZ\rangle$")):
        rows = [r for r in d[key] if r["n"] >= 3]
        nn = [r["n"] for r in rows]
        ax.semilogy(nn, [100 * r["fd_rel"] for r in rows], mk + "-", color=C_FD, ms=4,
                    label=f"FD best-$\\varepsilon$ — {obs}")
        ax.semilogy(nn, [100 * r["res_rel"] for r in rows], mk + "-", color=C_RES, ms=4,
                    mfc="white", label=f"PSR + transfer map — {obs}")
    ax.set_xlabel("qubits $n$")
    ax.set_ylabel(r"relative bias vs $\nabla C_{\rm ideal}$ (%)")
    ax.set_xticks([3, 4, 5, 6, 7])
    ax.text(0.03, 0.5, "$T/T_2^*=0.15$, $\\infty$ shots", transform=ax.transAxes,
            fontsize=7, color="#555")
    ax.legend(handlelength=1.6, loc="center right", fontsize=6)
    save(fig, "fig_bias_vs_n")
    loc = [r for r in d["loc"] if r["n"] >= 3]; ext = [r for r in d["ext"] if r["n"] >= 3]
    return dict(fd_range=(100 * min(r["fd_rel"] for r in loc + ext),
                          100 * max(r["fd_rel"] for r in loc + ext)),
                res_range=(100 * min(r["res_rel"] for r in loc + ext),
                           100 * max(r["res_rel"] for r in loc + ext)))


def fig_gate_error():
    ideal = load("bias_scaling_relative_data.json")
    ge99 = load("bias_scaling_gate_error_data.json")
    ge9995 = load("bias_scaling_gate_error_9995_data.json")
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    for key, mk in (("loc", "s"), ("ext", "^")):
        lab = "local" if key == "loc" else "extensive"
        rows = [r for r in ideal[key] if r["n"] >= 3]
        ax.plot([r["n"] for r in rows], [100 * r["res_rel"] for r in rows], mk + "-",
                color=C_RES, ms=4, mfc="white", label=f"ideal kick — {lab}")
        rows = [r for r in ge9995[key] if r["n"] >= 3]
        ax.plot([r["n"] for r in rows], [100 * r["res_rel"] for r in rows], mk + "-",
                color=C_ALT, ms=4, label=f"99.95% CZ (cryo) — {lab}")
        rows = [r for r in ge99[key] if r["n"] >= 3]
        ax.plot([r["n"] for r in rows], [100 * r["res_rel"] for r in rows], mk + "--",
                color=C_ALT2, ms=4, alpha=0.85, label=f"99.9% CZ — {lab}")
    ax.set_xlabel("qubits $n$")
    ax.set_ylabel("transfer-map residual bias (%)")
    ax.set_xticks([3, 4, 5, 6, 7]); ax.set_ylim(0.35, 2.25)
    ax.text(0.97, 0.55, "$T/T_2^*=0.15$", transform=ax.transAxes, fontsize=7, color="#555", ha="right")
    ax.legend(ncol=2, handlelength=1.4, fontsize=5.8, loc="upper right")
    save(fig, "fig_gate_error")
    deltas = {}
    for tag, ge in (("99.95%", ge9995), ("99.9%", ge99)):
        dmax = -1
        for key in ("loc", "ext"):
            for r in ge[key]:
                if r["n"] < 3:
                    continue
                m = [q for q in ideal[key] if q["n"] == r["n"]]
                if m:
                    dmax = max(dmax, 100 * (r["res_rel"] - m[0]["res_rel"]))
        deltas[tag] = dmax
    return deltas


def main():
    print("building ML (transfer-map) paper figures →", OUTDIR)
    rs = fig_shots(); fig_decomp(); rb = fig_bias_n(); rg = fig_gate_error()
    print("\n── prose numbers ──")
    print(f"shots (T/T2*={rs['regime']:.2f}): FD floor {rs['fd_floor']:.4f}, raw {rs['raw_floor']:.4f}, "
          f"corrected@max {rs['resc_at_max']:.4f}")
    print(f"bias-vs-n (0.15): FD {rb['fd_range'][0]:.1f}–{rb['fd_range'][1]:.1f}%, "
          f"corrected {rb['res_range'][0]:.2f}–{rb['res_range'][1]:.2f}%")
    print(f"gate-error insertion cost: 99.95% → +{rg['99.95%']:.2f}pp; 99.9% → +{rg['99.9%']:.2f}pp")


if __name__ == "__main__":
    main()
