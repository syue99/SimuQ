"""
build_paper_figs.py — render the ASPLOS paper figures into paper_fig/.

Reads ONLY the JSON data caches (no simulation); see figures_data_spec.md for
the per-figure spec.  Global conventions applied here:
  - time labels in T/T2* units, regime stated per panel
  - ACM single-column width 3.3 in, fonts ≥ 7pt at final size
  - Okabe-Ito colorblind-safe palette, stix (serif) mathtext
Outputs .pdf (paper) + .png (preview) per figure into
differential_computing/paper_fig/.

Run:  conda run -n qec_pg python differential_computing/tests/build_paper_figs.py
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
OUTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "paper_fig"))

# Okabe-Ito
C_FD = "#D55E00"       # vermillion — finite differences
C_RAW = "#8c8c8c"      # gray — raw PSR
C_RES = "#0072B2"      # blue — rescaled/corrected PSR
C_ALT = "#009E73"      # green — 99.95% variant / guides
C_ALT2 = "#E69F00"     # orange — 99.9% variant / extensive obs accents
C_INK = "#1a1a1a"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "font.family": "serif", "mathtext.fontset": "stix",
    "axes.linewidth": 0.7, "lines.linewidth": 1.6,
    "legend.frameon": False, "savefig.dpi": 300,
})

COL = 3.3  # ACM single-column width (in)


def load(name):
    return json.load(open(os.path.join(FIGDIR, name)))


def save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"),
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote paper_fig/{name}.pdf/.png")


# ── Fig 3 — the finite-difference trap ───────────────────────────────────────

def fig3():
    d = load("landscape_and_distance_noisy_data.json")
    regime = d["T"] / d["T2"]
    x_star, g_real = d["x_star"], d["g_real"]
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(COL, 4.9))

    gx = np.array(d["gx"])
    axA.plot(gx, d["y_ideal"], "--", color="#9a9a9a", lw=1.2, label="ideal")
    axA.plot(gx, d["y_noisy"], color=C_INK, lw=1.6, label="noisy")
    z0 = d["z0"]; ex = np.array([x_star - 0.28, x_star + 0.28])
    axA.plot(ex, z0 + g_real * (ex - x_star), color=C_RES, lw=2.2,
             label=f"true gradient ({g_real:+.2f})")
    ramp = plt.cm.Oranges(np.linspace(0.45, 0.9, len(d["secants"])))
    for k, (sec, c) in enumerate(zip(d["secants"], ramp)):
        e, fm, fp = sec["eps"], sec["fm"], sec["fp"]
        axA.plot([x_star - e, x_star + e], [fm, fp], "o-", color=c, lw=1.2,
                 ms=2.6, label=(r"FD secants, $\varepsilon$=0.15–0.6"
                                " (all wrong sign)") if k == 0 else None)
    axA.plot(ex, z0 + d["sl01"] * (ex - x_star), ":", color=C_FD, lw=1.8,
             label=r"FD $\varepsilon$=0.01, finite shots (wrong sign)")
    axA.plot(ex, z0 + d["psr_slope"] * (ex - x_star), "-", color=C_ALT, lw=1.8,
             label=f"PSR corrected ({d['psr_slope']:+.2f})")
    axA.plot([x_star], [z0], "o", color=C_INK, ms=4)
    axA.set_xlabel("parameter $\\theta$")
    axA.set_ylabel(r"$\langle O\rangle(\theta)$")
    axA.text(0.97, 0.94, f"$T/T_2^*={regime:.2f}$", transform=axA.transAxes,
             fontsize=7, color="#555", ha="right")
    axA.set_ylim(bottom=min(d["y_noisy"]) - 0.34)
    axA.legend(loc="lower left", handlelength=1.5, fontsize=6)

    eps = np.array(d["eps_grid"]); rmse = np.array(d["fd_rmse"])
    wrong = np.array(d["fd_wrongfrac"]) > 0.20
    axB.loglog(eps, rmse, "-", color=C_FD, lw=1.6, label="FD (finite shots)")
    axB.loglog(eps[~wrong], rmse[~wrong], "o", color=C_FD, ms=3.5)
    axB.loglog(eps[wrong], rmse[wrong], "X", color=C_INK, ms=6,
               label=">20% wrong sign")
    axB.axhline(abs(g_real), color="#999", lw=0.9, ls="-.")
    axB.text(eps[-1] * 0.95, abs(g_real) * 1.04, "|true gradient|", fontsize=6.5,
             color="#666", ha="right", va="bottom")
    axB.axhline(d["psr_raw_rmse"], color=C_RAW, lw=1.4, ls="--",
                label="compiled PSR, raw")
    axB.axhline(d["psr_resc_rmse"], color=C_RES, lw=2.0,
                label="compiled PSR, corrected")
    axB.set_xlabel(r"FD step size $\varepsilon$")
    axB.set_ylabel("gradient error (RMSE)")
    axB.text(0.97, 0.05, f"$T/T_2^*={regime:.2f}$, $N$={d['N_SHOTS']} shots",
             transform=axB.transAxes, fontsize=7, color="#555", ha="right")
    axB.legend(loc="upper left", ncol=1, handlelength=1.6, fontsize=6)
    save(fig, "fig3_fd_trap")
    return dict(regime=regime,
                wrong_eps=f"{int(wrong.sum())}/{len(eps)}",
                fd_best_rmse=float(rmse.min()),
                psr_raw_rmse=d["psr_raw_rmse"], psr_resc_rmse=d["psr_resc_rmse"])


# ── Fig 5 — shots scaling + decomposition ────────────────────────────────────

def fig5a():
    d = load("shots_scaling_data.json")
    regime = d["T"] / d["T2"]
    N = np.array(d["budgets"])
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    ax.loglog(N, d["fd_best"], "s-", color=C_FD, ms=4,
              label=r"FD, oracle-tuned $\varepsilon$")
    ax.loglog(N, d["psr_raw"], "o--", color=C_RAW, ms=4, label="PSR raw")
    ax.loglog(N, d["psr_res"], "o-", color=C_RES, ms=4, lw=2.0,
              label="PSR corrected")
    ax.loglog(N, d["psr_res"][0] * (N / N[0]) ** -0.5, ":", color=C_ALT,
              lw=1.0, label=r"$N^{-1/2}$")
    ax.set_xlabel("total shots $N$")
    ax.set_ylabel("gradient error (RMSE)")
    ax.text(0.97, 0.94, f"$T/T_2^*={regime:.2f}$", transform=ax.transAxes,
            fontsize=7, color="#555", ha="right")
    ax.legend(handlelength=1.6, loc="lower left")
    save(fig, "fig5a_shots_scaling")
    return dict(regime=regime, fd_floor=float(d["fd_best"][-1]),
                raw_floor=float(d["psr_raw"][-1]),
                resc_at_max=float(d["psr_res"][-1]))


def fig5b():
    d = load("shots_decomposition_data.json")
    regime = d["T"] / d["T2"]
    N = np.array(d["budgets"])
    key_fd = [k for k in d["D"] if k.startswith("FD")][0]
    colors = {"PSR raw": C_RAW, "PSR rescaled": C_RES, key_fd: C_FD}
    labels = {"PSR raw": "PSR raw", "PSR rescaled": "PSR corrected",
              key_fd: key_fd.replace("ε", r"$\varepsilon$")}
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    for k, c in colors.items():
        ax.loglog(N, d["D"][k]["bias"], "-", color=c, lw=1.8,
                  label=f"{labels[k]} — bias")
        ax.loglog(N, d["D"][k]["std"], "--", color=c, lw=1.0, alpha=0.75,
                  label=f"{labels[k]} — std")
    ax.set_xlabel("total shots $N$")
    ax.set_ylabel("error component")
    ax.text(0.03, 0.05, f"$T/T_2^*={regime:.2f}$", transform=ax.transAxes,
            fontsize=7, color="#555")
    ax.legend(ncol=2, handlelength=1.5, fontsize=6)
    save(fig, "fig5b_decomposition")
    return dict(regime=regime)


# ── Fig 6 — bias vs size + gate-error variants ───────────────────────────────

def fig6a():
    d = load("bias_scaling_relative_data.json")
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    for key, mk, obs in (("loc", "s", r"local $\langle Z_0Z_1\rangle$"),
                         ("ext", "^", r"extensive $\langle\Sigma ZZ\rangle$")):
        rows = [r for r in d[key] if r["n"] >= 3]
        nn = [r["n"] for r in rows]
        ax.semilogy(nn, [100 * r["fd_rel"] for r in rows], mk + "-",
                    color=C_FD, ms=4, label=f"FD best-$\\varepsilon$ — {obs}")
        ax.semilogy(nn, [100 * r["res_rel"] for r in rows], mk + "-",
                    color=C_RES, ms=4, mfc="white",
                    label=f"PSR corrected — {obs}")
    ax.set_xlabel("qubits $n$")
    ax.set_ylabel("relative gradient bias (%)")
    ax.set_xticks([3, 4, 5, 6, 7])
    ax.text(0.03, 0.5, "$T/T_2^*=0.15$, $\\infty$ shots",
            transform=ax.transAxes, fontsize=7, color="#555")
    ax.legend(handlelength=1.6, loc="center right", fontsize=6)
    save(fig, "fig6a_bias_vs_n")
    loc = [r for r in d["loc"] if r["n"] >= 3]
    ext = [r for r in d["ext"] if r["n"] >= 3]
    return dict(fd_range=(100 * min(r["fd_rel"] for r in loc + ext),
                          100 * max(r["fd_rel"] for r in loc + ext)),
                res_range=(100 * min(r["res_rel"] for r in loc + ext),
                           100 * max(r["res_rel"] for r in loc + ext)))


def fig6b():
    ideal = load("bias_scaling_relative_data.json")
    ge99 = load("bias_scaling_gate_error_data.json")
    ge9995 = load("bias_scaling_gate_error_9995_data.json")
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    for key, mk in (("loc", "s"), ("ext", "^")):
        lab = "local" if key == "loc" else "extensive"
        rows = [r for r in ideal[key] if r["n"] >= 3]
        ax.plot([r["n"] for r in rows], [100 * r["res_rel"] for r in rows],
                mk + "-", color=C_RES, ms=4, mfc="white",
                label=f"ideal kick — {lab}")
        rows = [r for r in ge9995[key] if r["n"] >= 3]
        ax.plot([r["n"] for r in rows], [100 * r["res_rel"] for r in rows],
                mk + "-", color=C_ALT, ms=4,
                label=f"99.95% CZ (cryo) — {lab}")
        rows = [r for r in ge99[key] if r["n"] >= 3]
        ax.plot([r["n"] for r in rows], [100 * r["res_rel"] for r in rows],
                mk + "--", color=C_ALT2, ms=4, alpha=0.85,
                label=f"99.9% CZ — {lab}")
    ax.set_xlabel("qubits $n$")
    ax.set_ylabel("corrected-PSR bias (%)")
    ax.set_xticks([3, 4, 5, 6, 7])
    ax.set_ylim(0.35, 2.25)
    ax.text(0.97, 0.55, "$T/T_2^*=0.15$", transform=ax.transAxes,
            fontsize=7, color="#555", ha="right")
    ax.legend(ncol=2, handlelength=1.4, fontsize=5.8, loc="upper right")
    save(fig, "fig6b_gate_error")
    # the [X]pp deltas (per spec: report explicitly)
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


# ── Fig 7 — compile at scale ─────────────────────────────────────────────────

def fig7():
    rows = [r for r in load("compile_scaling_data.json") if r["status"] == "ok"]
    nn = [r["n"] for r in rows]
    fig, axs = plt.subplots(1, 3, figsize=(COL, 1.55))
    axs[0].semilogy(nn, [r["compile_s"] for r in rows], "o-", color=C_RES, ms=3)
    axs[0].set_ylabel("compile (s)", fontsize=7)
    axs[1].plot(nn, [r["branches"] for r in rows], "o-", color=C_RES, ms=3)
    b0 = rows[0]
    axs[1].plot(nn, [b0["branches"] / (b0["n"] - 1) * (n - 1) for n in nn],
                ":", color="#999", lw=0.9)
    axs[1].set_ylabel("branches", fontsize=7)
    axs[2].plot(nn, [r["dur"] for r in rows], "o-", color=C_RES, ms=3)
    axs[2].set_ylim(0, max(r["dur"] for r in rows) * 1.6)
    axs[2].set_ylabel("pulse depth", fontsize=7)
    for ax in axs:
        ax.set_xlabel("$n$", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig7_compile_scaling")
    return dict(n_max=max(nn), t_max=max(r["compile_s"] for r in rows))


# ── Fig 8 — the cost wall ────────────────────────────────────────────────────

def fig8():
    d = load("cost_wall_data.json")
    sim = d["sim"]
    ns = np.array([r["n"] for r in sim])
    tg = np.array([r["t_gradient"] for r in sim])
    coef = np.polyfit(ns[2:], np.log(tg[2:]), 1)
    n_ext = np.arange(ns[-1], 21)
    n_hour = (np.log(3600) - coef[1]) / coef[0]

    comp = [r for r in d["compile_rows"] if r["status"] == "ok"]
    fig, ax = plt.subplots(figsize=(COL, 2.7))
    ax.semilogy(ns, tg, "o-", color=C_FD, ms=4, label="exact noisy simulation")
    ax.semilogy(n_ext, np.exp(coef[1] + coef[0] * n_ext), "--", color=C_FD,
                lw=1.1, alpha=0.8)
    ax.semilogy([r["n"] for r in comp], [r["compile_s"] for r in comp], "s-",
                color="#7b1fa2", ms=4, label="compilation")
    corr_ns = [2, 4, 7, 12, 20]
    ax.semilogy(corr_ns, [d["t_corr"]] * len(corr_ns), "^:", color=C_RES,
                ms=4.5, lw=1.1, label="$O(1)$ correction")
    ax.axhline(3600, color="#999", lw=0.8, ls="-.")
    ax.text(2.1, 4600, "1 hour", fontsize=6.5, color="#777")
    ax.axvspan(n_hour, 21, color="#f2f2f2", zorder=0)
    ax.text((n_hour + 20.5) / 2, np.sqrt(tg[0] * 3600), "simulation\nintractable",
            ha="center", fontsize=7, color="#999")
    ax.set_xlim(1.5, 20.5)
    ax.set_xlabel("qubits $n$")
    ax.set_ylabel("wall-clock (s)")
    ax.legend(handlelength=1.6, loc="upper left", fontsize=6.2)
    save(fig, "fig8_cost_wall")
    return dict(n_hour=float(n_hour), per_qubit=float(np.exp(coef[0])),
                t_corr=d["t_corr"])


def main():
    print("building paper figures →", OUTDIR)
    r3 = fig3()
    r5a = fig5a(); r5b = fig5b()
    r6a = fig6a(); r6b = fig6b()
    r7 = fig7()
    r8 = fig8()

    print("\n── prose numbers (spec: report, prose follows data) ──")
    print(f"Fig 3 (T/T2*={r3['regime']:.2f}): wrong-sign ε {r3['wrong_eps']}; "
          f"FD best RMSE {r3['fd_best_rmse']:.3f}; PSR raw {r3['psr_raw_rmse']:.3f}; "
          f"corrected {r3['psr_resc_rmse']:.3f}")
    print(f"Fig 5 REGIME FLAG: data at T/T2*={r5a['regime']:.2f}, spec headline is "
          f"0.15 — re-run is the author's decision (expensive)")
    print(f"Fig 5a floors: FD-best {r5a['fd_floor']:.4f}, raw {r5a['raw_floor']:.4f}, "
          f"corrected@2e5 {r5a['resc_at_max']:.4f}")
    print(f"Fig 6a (0.15): FD {r6a['fd_range'][0]:.1f}–{r6a['fd_range'][1]:.1f}%, "
          f"corrected {r6a['res_range'][0]:.2f}–{r6a['res_range'][1]:.2f}%")
    print(f"Fig 6b gate-insertion cost: 99.95% → +{r6b['99.95%']:.2f}pp (≤0.4pp claim); "
          f"99.9% → +{r6b['99.9%']:.2f}pp  ← the [X]pp placeholder")
    print(f"Fig 7: compiled to n={r7['n_max']} (max {r7['t_max']:.1f}s)")
    print(f"Fig 8: sim ×{r8['per_qubit']:.1f}/qubit, crosses 1h at n≈{r8['n_hour']:.1f}; "
          f"correction {r8['t_corr']:.2f}s at any n")


if __name__ == "__main__":
    main()
