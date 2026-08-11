"""
build_paper_figs.py — render the ASPLOS (DiffSimuQ, PL+Sys) paper figures into
paper_fig/.

Device-target framing (raw PSR = the EXACT gradient of the deployed noisy program;
NO rescale — the rescale/transfer-map results live in build_ml_paper_figs.py).
Reads ONLY the JSON caches (no simulation).  Global ACM conventions applied here.

  fig3  FD trap: on the sharp noisy landscape FD needs small ε (large ε → truncation,
        wrong sign) but small ε amplifies the control error δ/ε — no ε works; PSR ε-free.
  fig5  device-gradient accuracy vs control resolution r (relative, two noise levels):
        the δ/ε disadvantage is a CONTROL-resolution effect (γ-independent); PSR exact.
  fig6  finite shots: PSR converges ~N^-1/2 to ∇C_noisy; oracle-FD floors at the δ/ε bias.
  fig7  compile at scale.
  fig8  cost wall (sim vs compile).
  figR  resource pillar (analog-native vs digital Trotter emulation per branch).

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
C_RAW = "#8c8c8c"      # gray
C_RES = "#0072B2"      # blue — raw PSR (the exact device gradient)
C_ALT = "#009E73"      # green — guides / N^-1/2
C_ALT2 = "#E69F00"     # orange — second noise level
C_INK = "#1a1a1a"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "font.family": "serif", "mathtext.fontset": "stix",
    "axes.linewidth": 0.7, "lines.linewidth": 1.6,
    "legend.frameon": False, "savefig.dpi": 300,
})
COL = 3.3  # ACM single-column width (in)
RS = [0.01, 0.03, 0.06, 0.1, 0.15]   # control resolutions swept in fig5


def load(name):
    return json.load(open(os.path.join(FIGDIR, name)))


def save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"),
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote paper_fig/{name}.pdf/.png")


# ── Fig 3 — the finite-difference trap (device-target, with control error δ) ──

def fig3():
    d = load("landscape_device_data.json")
    regime = d["T"] / d["T2"]; x_star = d["x_star"]; g = d["g_real"]
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(COL, 4.9))

    gx = np.array(d["gx"])
    axA.plot(gx, d["y_noisy"], color=C_INK, lw=1.6, label="noisy device landscape")
    z0 = d["z0"]; ex = np.array([x_star - 0.30, x_star + 0.30])
    axA.plot(ex, z0 + g * (ex - x_star), color=C_RES, lw=2.2,
             label=rf"$\nabla C_{{\rm noisy}}$ = raw PSR ({g:+.2f})")
    ramp = plt.cm.Oranges(np.linspace(0.45, 0.9, len(d["secants"])))
    for k, (sec, c) in enumerate(zip(d["secants"], ramp)):
        e, fm, fp = sec["eps"], sec["fm"], sec["fp"]
        axA.plot([x_star - e, x_star + e], [fm, fp], "o-", color=c, lw=1.2, ms=2.6,
                 label=(r"FD secants $\varepsilon$=0.15–0.6 (wrong sign)") if k == 0 else None)
    axA.plot(ex, z0 + d["sl_small"] * (ex - x_star), ":", color=C_FD, lw=1.8,
             label=r"FD $\varepsilon$=0.05 + control $\delta$ (wrong sign)")
    axA.plot([x_star], [z0], "o", color=C_INK, ms=4)
    axA.set_xlabel(r"parameter $\theta$")
    axA.set_ylabel(r"$\langle O\rangle_{\rm noisy}(\theta)$")
    axA.text(0.97, 0.94, rf"$T/T_2^*={regime:.2f}$", transform=axA.transAxes,
             fontsize=7, color="#555", ha="right")
    axA.set_ylim(bottom=min(d["y_noisy"]) - 0.30)
    axA.legend(loc="lower left", handlelength=1.5, fontsize=6)

    eps = np.array(d["eps_grid"]); rmse = np.array(d["fd_rmse"])
    wrong = np.array(d["fd_wrong"]) > 0.20
    axB.loglog(eps, rmse, "-", color=C_FD, lw=1.6, label=r"FD (shots + control error $\delta$)")
    axB.loglog(eps[~wrong], rmse[~wrong], "o", color=C_FD, ms=3.5)
    axB.loglog(eps[wrong], rmse[wrong], "X", color=C_INK, ms=6, label=">20% wrong sign")
    axB.axhline(d["psr_rmse"], color=C_RES, lw=2.0, label="raw PSR (exact device grad)")
    axB.set_xlabel(r"FD step size $\varepsilon$")
    axB.set_ylabel(r"error vs $\nabla C_{\rm noisy}$ (RMSE)")
    axB.set_ylim(top=rmse.max() * 1.9)
    axB.annotate(r"$\delta/\varepsilon$ floor", xy=(eps[1], rmse[1]),
                 xytext=(eps[0] * 1.1, rmse.max() * 1.4), fontsize=6, color="#a0451a",
                 arrowprops=dict(arrowstyle="->", color="#a0451a", lw=0.7))
    axB.annotate("truncation", xy=(eps[-2], rmse[-2]),
                 xytext=(eps[-1] * 0.5, rmse.max() * 1.55), fontsize=6, color="#a0451a",
                 ha="right", arrowprops=dict(arrowstyle="->", color="#a0451a", lw=0.7))
    axB.text(0.5, 0.96, rf"$T/T_2^*={regime:.2f}$, $N$={d['N_SHOTS']}, control $r$={d['r_ctrl']}",
             transform=axB.transAxes, fontsize=6.3, color="#555", ha="center", va="top")
    axB.legend(loc="center", handlelength=1.6, fontsize=6)
    save(fig, "fig3_fd_trap")
    return dict(regime=regime, wrong_eps=f"{int(wrong.sum())}/{len(eps)}",
                fd_best=float(rmse.min()), psr=d["psr_rmse"])


# ── Fig 5 — device-gradient accuracy vs control resolution (delta noise) ──────

def fig5():
    d = load("psr_fd_device_gradient_data.json")["partA"]
    fig, ax = plt.subplots(figsize=(COL, 2.6)); psr_all = []
    for label, col in (("T/T2*=0.15", C_FD), ("T/T2*=0.5", C_ALT2)):
        rows = d[label]; psr_all += rows["psr_rel"]
        med = np.median(np.array(rows["fd_rel"]), axis=0)
        reg = label.split("=")[1]
        ax.loglog(RS, med, "s-", color=col, ms=3.6, label=rf"oracle-FD, $T/T_2^*{{=}}{reg}$")
    ax.axhline(np.median(psr_all), color=C_RES, lw=2.2, label="raw PSR (exact)")
    ax.set_xlabel(r"control resolution $r$  (floors $\varepsilon$, sets $\delta$)")
    ax.set_ylabel(r"relative error vs $\nabla C_{\rm noisy}$")
    ax.text(0.03, 0.06, r"FD's $\delta/\varepsilon$ penalty is control-" "\n"
            r"resolution (two $\gamma$ overlap); PSR exact", transform=ax.transAxes,
            fontsize=6, color="#666")
    ax.legend(handlelength=1.6, loc="upper left")
    save(fig, "fig5_device_gradient_accuracy")
    return dict(psr=float(np.median(psr_all)),
                fd15=float(np.median(np.array(d["T/T2*=0.15"]["fd_rel"]), 0)[-1]),
                fd50=float(np.median(np.array(d["T/T2*=0.5"]["fd_rel"]), 0)[-1]))


# ── Fig 6 — finite shots (delta noise): PSR converges, FD floors at δ/ε ───────

def fig6():
    b = load("psr_fd_device_gradient_data.json")["partB"]
    N = np.array(b["N"])
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    ax.loglog(N, b["psr_rmse"], "o-", color=C_RES, ms=4, lw=1.9,
              label="raw PSR (exact device grad)")
    ax.loglog(N, b["fd_rmse"], "s-", color=C_FD, ms=4,
              label=rf"oracle-FD (control $r{{=}}{b['r']}$)")
    ax.loglog(N, np.array(b["psr_rmse"])[0] * (N / N[0]) ** -0.5, ":", color=C_ALT,
              lw=1.0, label=r"$N^{-1/2}$")
    floor = float(np.median(b["fd_rmse"][-2:]))
    ax.axhline(floor, color=C_FD, lw=0.9, ls="-.")
    ax.text(N[2], floor * 0.80, r"$\delta/\varepsilon$ bias floor (finite shots cannot remove)",
            fontsize=6, color="#a0451a", va="top", ha="center")
    ax.set_xlabel("total shots per gradient component $N$")
    ax.set_ylabel(r"RMSE vs $\nabla C_{\rm noisy}$")
    ax.legend(handlelength=1.6, loc="lower left")
    save(fig, "fig6_device_finite_shot")
    return dict(psr_min=float(b["psr_rmse"][-1]), fd_floor=floor)


# ── Fig 7 — compile at scale ─────────────────────────────────────────────────

def fig7():
    rows = [r for r in load("compile_scaling_data.json") if r["status"] == "ok"]
    nn = [r["n"] for r in rows]
    fig, axs = plt.subplots(1, 3, figsize=(COL, 1.55))
    axs[0].semilogy(nn, [r["compile_s"] for r in rows], "o-", color=C_RES, ms=3)
    axs[0].set_ylabel("compile (s)", fontsize=7)
    axs[1].plot(nn, [r["branches"] for r in rows], "o-", color=C_RES, ms=3)
    b0 = rows[0]
    axs[1].plot(nn, [b0["branches"] / (b0["n"] - 1) * (n - 1) for n in nn], ":",
                color="#999", lw=0.9)
    axs[1].set_ylabel("branches", fontsize=7)
    axs[2].plot(nn, [r["dur"] for r in rows], "o-", color=C_RES, ms=3)
    axs[2].set_ylim(0, max(r["dur"] for r in rows) * 1.6)
    axs[2].set_ylabel("pulse depth", fontsize=7)
    for ax in axs:
        ax.set_xlabel("$n$", fontsize=7); ax.tick_params(labelsize=6)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig7_compile_scaling")
    return dict(n_max=max(nn), t_max=max(r["compile_s"] for r in rows))


# ── Fig 8 — the cost wall (sim vs compile; no correction line) ────────────────

def fig8():
    d = load("cost_wall_data.json")
    sim = d["sim"]; ns = np.array([r["n"] for r in sim]); tg = np.array([r["t_gradient"] for r in sim])
    coef = np.polyfit(ns[2:], np.log(tg[2:]), 1)
    n_ext = np.arange(ns[-1], 21); n_hour = (np.log(3600) - coef[1]) / coef[0]
    comp = [r for r in d["compile_rows"] if r["status"] == "ok"]
    fig, ax = plt.subplots(figsize=(COL, 2.7))
    ax.semilogy(ns, tg, "o-", color=C_FD, ms=4, label="exact noisy simulation")
    ax.semilogy(n_ext, np.exp(coef[1] + coef[0] * n_ext), "--", color=C_FD, lw=1.1, alpha=0.8)
    ax.semilogy([r["n"] for r in comp], [r["compile_s"] for r in comp], "s-",
                color=C_RES, ms=4, label="differentiable compilation")
    ax.axhline(3600, color="#999", lw=0.8, ls="-.")
    ax.text(2.1, 4600, "1 hour", fontsize=6.5, color="#777")
    ax.axvspan(n_hour, 21, color="#f2f2f2", zorder=0)
    ax.text((n_hour + 20.5) / 2, np.sqrt(tg[0] * 3600), "simulation\nintractable",
            ha="center", fontsize=7, color="#999")
    ax.annotate("constant pulse depth", xy=(comp[-1]["n"], comp[-1]["compile_s"]),
                xytext=(7, 0.04), fontsize=6, color=C_RES,
                arrowprops=dict(arrowstyle="->", color=C_RES, lw=0.8))
    ax.set_xlim(1.5, 20.5); ax.set_xlabel("qubits $n$"); ax.set_ylabel("wall-clock (s)")
    ax.legend(handlelength=1.6, loc="upper left", fontsize=6.2)
    save(fig, "fig8_cost_wall")
    return dict(n_hour=float(n_hour), per_qubit=float(np.exp(coef[0])))


# ── Fig R — resource pillar (analytic) ───────────────────────────────────────

def figR():
    Tval = 1.5
    n = np.arange(2, 65)
    gd = lambda eps: n ** 2 * Tval ** 2 / (2 * eps)
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    ax.axhspan(1 / (1 - 0.999), 1e9, color="#f5f5f5", zorder=0)
    for F in (0.999, 0.9999):
        ax.axhline(1 / (1 - F), color="#bbb", ls="--", lw=0.8)
    ax.text(2.2, 1 / (1 - 0.999) * 1.3, "99.9% 2q gate-error wall", fontsize=5.6, color="#777")
    ax.loglog(n, gd(1e-2), "-", color=C_FD, lw=1.8,
              label=r"digital Trotter, $\varepsilon{=}10^{-2}$ ($\sim n^2T^2/\varepsilon$)")
    ax.loglog(n, gd(1e-3), "--", color=C_FD, lw=1.3, label=r"digital Trotter, $\varepsilon{=}10^{-3}$")
    ax.loglog(n, np.ones_like(n), "-", color=C_RES, lw=2.2,
              label="analog-native: 1 evolution (const. in $n$)")
    ax.set_xlabel("qubits $n$"); ax.set_ylabel("2q ops per gradient branch")
    ax.set_ylim(0.4, 1e7); ax.set_xlim(2, 64)
    ax.legend(handlelength=1.5, loc="upper left", fontsize=5.8)
    save(fig, "figR_resource_pillar")
    return dict(g_n10=float(gd(1e-2)[8]))


def main():
    print("building DiffSimuQ paper figures →", OUTDIR)
    r3 = fig3(); r5 = fig5(); r6 = fig6(); r7 = fig7(); r8 = fig8(); rR = figR()
    print("\n── prose numbers ──")
    print(f"Fig 3 (T/T2*={r3['regime']:.2f}): wrong-sign ε {r3['wrong_eps']}; "
          f"FD best RMSE {r3['fd_best']:.3f}; raw PSR {r3['psr']:.3f}")
    print(f"Fig 5: raw PSR rel err ~{r5['psr']:.0e}; oracle-FD rel err at r=0.15: "
          f"{r5['fd15']:.2f} (0.15) / {r5['fd50']:.2f} (0.5) — γ-independent")
    print(f"Fig 6: PSR rmse→{r6['psr_min']:.3f} (converging); FD floors ~{r6['fd_floor']:.2f}")
    print(f"Fig 7: compiled to n={r7['n_max']} (max {r7['t_max']:.1f}s)")
    print(f"Fig 8: sim ×{r8['per_qubit']:.1f}/qubit, crosses 1h at n≈{r8['n_hour']:.1f}")
    print(f"Fig R: digital 2q-gates at n=10, ε=1e-2 ≈ {rR['g_n10']:.0f} (> NISQ wall)")


if __name__ == "__main__":
    main()
