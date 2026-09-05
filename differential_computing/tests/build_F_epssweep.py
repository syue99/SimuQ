"""
build_F_epssweep.py — P1-1 (App C.3 companion): FD's bias floor as a function of the step ε
at three operating points of increasing landscape sharpness.  SHOT-FREE (N → ∞): the figure
compares bias floors only, so the shot budget never enters.

Per panel (T = 1, 2.5, 5 µs; T/T₂* = 0.15 held; same 2q TFIM, same device model r = 0.02):
  θ0    a generic point (not an inflection): a random steep point, fixed seed, whose
        shared-draw displacement |f''|·r/|∇C| lies in PSR_BAND = 2–5%, so PSR's floor is of
        the same order in every panel and the panels differ in the step scale (owner,
        2026-09-05).  The landscape itself is drawn above each sweep; T and θ0 are not
        printed on the figure (caption / data note only).
  FD    RMSE(ε)/|∇C| with the paper's probes θ ± ε/2 each carrying its own setpoint draw
        (per-change rule, 2 draws per estimate), 2000 draws per step, 30 steps in [0.02, 3.0].
        × = steps where ≥ 20% of draws give the wrong sign.  Shaded = usable window
        RMSE ≤ WIN of |∇C|.  ε* = argmin.
  PSR   shot-free floor = the shared-draw displacement |f''|·r (zero only at C'' = 0).
  NSR   no floor under the per-change rule (fresh draw per execution).
Owner's design 2026-09-05.  Cache: figures/F_epssweep_data.json (REPLOT=1 reuses it).
Run: conda run -n qec_pg python differential_computing/tests/build_F_epssweep.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sympy as sp
import qutip as qp
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simuq import QSystem, Qubit
from noisy_qutip import NoisyQuTiPRunner
from noise_model import NoiseModel

R_CTRL = 0.02
REGIME = 0.15
TS = [1.0, 2.5, 5.0]
EPS = np.geomspace(0.02, 3.0, 30)          # wider than Fig 8's inset grid so the window is seen closing
NDRAW = 2000
WIN = 0.30                                   # usable-window threshold (fraction of |∇C|)
WRONG = 0.20                                 # × marker threshold (fraction of draws)
LAND_HALF = 1.6                              # landscape drawn over θ0 ± LAND_HALF (same for all panels)
FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
OUT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "paper_fig_2"))
OUT3 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "paper_fig_3", "figs"))
CACHE = os.path.join(FIGDIR, "F_epssweep_data.json")
C_FD, C_PSR, C_NSR = "#D55E00", "#0072B2", "#009E73"


def landscape(T):
    th = sp.Symbol("th"); qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    H = th * q[0].Z * q[1].Z + (q[0].X + q[1].X)
    ex = NoisyQuTiPRunner(2, noise=NoiseModel(n_qubits=2, T2=T / REGIME, gate_error_2q=None)) \
        .make_expectation_fn(qp.tensor(qp.basis(2, 0), qp.basis(2, 0)),
                             qp.tensor(qp.sigmaz(), qp.sigmaz()))
    C = lambda t: ex([[H.set_parameterizedHam({"th": float(t)}), T]])
    h = 0.004
    g = np.arange(0.2 - 1.8, 3.4 + 1.8, h)         # covers θ0 ± ε/2 ± 3r for ε up to 3
    v = np.array([C(t) for t in g])
    d1 = savgol_filter(v, 51, 7, deriv=1, delta=h)
    d2 = savgol_filter(v, 51, 7, deriv=2, delta=h)
    d3 = savgol_filter(v, 51, 7, deriv=3, delta=h)
    return g, v, d1, d2, d3


PSR_BAND = (0.02, 0.05)                      # |f''| r / |f'| band for the operating point
PICK_SEED = 11


def pick_theta(g, d1, d2, d3):
    """A generic operating point (owner, 2026-09-05: NOT an inflection): a random steep point
    (|∇C| ≥ ½ max, θ ∈ [0.5, 3]) whose shared-draw displacement |f''|·r/|∇C| lies in PSR_BAND,
    so PSR's floor is of the same order in every panel, and whose sharpness |f'''|/|f'| is
    typical of its landscape (middle quintile over the steep points).  Fixed seed.
    Returns (θ0, predicted FD floor)."""
    eps = np.geomspace(0.01, 2.0, 500)
    m = (g >= 0.5) & (g <= 3.0) & (np.abs(d1) >= 0.5 * np.abs(d1).max())
    disp = np.abs(d2) * R_CTRL / np.abs(d1)
    sharp = np.abs(d3) / np.abs(d1)
    lo, hi = np.percentile(sharp[m], [40, 60])          # typical sharpness for THIS landscape
    ok = m & (disp >= PSR_BAND[0]) & (disp <= PSR_BAND[1]) & (sharp >= lo) & (sharp <= hi)
    idx = np.flatnonzero(ok)
    assert idx.size, "no steep point with the PSR displacement in band"
    i = int(np.random.default_rng(PICK_SEED).choice(idx))
    a, b, c = float(d1[i]), float(d2[i]), float(d3[i])
    fd = np.sqrt((np.sqrt(2) * R_CTRL * abs(a) / eps) ** 2 + (abs(b) * R_CTRL / np.sqrt(2)) ** 2
                 + (eps ** 2 * abs(c) / 24) ** 2)
    return float(g[i]), float(fd.min() / abs(a))


def panel(T, rng):
    g, v, d1, d2, d3 = landscape(T)
    th0, fl_pred = pick_theta(g, d1, d2, d3)
    Cint = interp1d(g, v, kind="cubic")
    i0 = int(np.argmin(np.abs(g - th0)))
    f1, f2, f3 = float(d1[i0]), float(d2[i0]), float(d3[i0])
    rows = []
    for e in EPS:
        dp, dm = rng.normal(0, R_CTRL, NDRAW), rng.normal(0, R_CTRL, NDRAW)
        est = (Cint(th0 + e / 2 + dp) - Cint(th0 - e / 2 + dm)) / e
        rows.append(dict(eps=float(e), rmse_rel=float(np.sqrt(np.mean((est - f1) ** 2)) / abs(f1)),
                         bias_rel=float((np.mean(est) - f1) / abs(f1)),
                         wrong=float(np.mean(np.sign(est) != np.sign(f1)))))
    rr = np.array([r["rmse_rel"] for r in rows])
    ok = rr <= WIN
    win = (float(EPS[ok].min()), float(EPS[ok].max())) if ok.any() else None
    xs = np.linspace(-LAND_HALF, LAND_HALF, 361)
    return dict(T=T, T2=T / REGIME, theta0=th0, f1=f1, f2=f2, f3=f3,
                land_x=[float(x) for x in xs], land_y=[float(Cint(th0 + x)) for x in xs],
                C0=float(Cint(th0)),
                eps_star=float(EPS[int(np.argmin(rr))]), floor_rel=float(rr.min()),
                floor_pred_rel=fl_pred,
                eps_star_b64=float((24 * abs(f1) * R_CTRL / abs(f3)) ** (1 / 3)),
                floor_b64_rel=float(0.60 * abs(f3) ** (1 / 3) * (abs(f1) * R_CTRL) ** (2 / 3) / abs(f1)),
                psr_disp_rel=float(abs(f2) * R_CTRL / abs(f1)),
                window=win, window_decades=(float(np.log10(win[1] / win[0])) if win else 0.0),
                sweep=rows)


def compute():
    rng = np.random.default_rng(7)
    return dict(r=R_CTRL, regime=REGIME, eps=[float(e) for e in EPS], ndraw=NDRAW, win=WIN,
                wrong=WRONG, panels=[panel(T, rng) for T in TS])


def render(d):
    plt.rcParams.update({"font.size": 7})
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 3.6), dpi=300,
                            gridspec_kw=dict(height_ratios=[0.62, 1.0], hspace=0.38, wspace=0.10))
    labels = ["(a) healthy: wide step window", "(b) intermediate", "(c) ill: narrow step window"]
    ally = np.concatenate([p["land_y"] for p in d["panels"]])
    ylo, yhi = float(ally.min()) - 0.12, float(ally.max()) + 0.12
    for k, (p, lab) in enumerate(zip(d["panels"], labels)):
        # ── top: the landscape around θ0, tangent = ∇C_device, best FD secant at ε* ──
        axL = axs[0, k]
        x = np.array(p["land_x"]); y = np.array(p["land_y"]); f1 = p["f1"]; C0 = p["C0"]
        axL.plot(x, y, color="#1a1a1a", lw=1.3)
        axL.plot([0], [C0], "o", color="#1a1a1a", ms=3.2, zorder=5)
        ht = min(0.45, 0.22 / abs(f1))          # tangent of fixed vertical extent
        axL.plot([-ht, ht], [C0 - ht * f1, C0 + ht * f1], color=C_PSR, lw=2.2, zorder=4,
                 solid_capstyle="round")
        if p["window"]:
            axL.axvspan(-p["window"][1] / 2, p["window"][1] / 2, color=C_FD, alpha=0.08, lw=0)
        e = p["eps_star"]
        axL.plot([-e / 2, e / 2], [np.interp(-e / 2, x, y), np.interp(e / 2, x, y)], "o-",
                 color=C_FD, lw=1.1, ms=2.6, zorder=3)
        axL.set_xlim(-LAND_HALF, LAND_HALF); axL.set_ylim(ylo, yhi)
        axL.tick_params(labelsize=6.5)
        if k > 0:
            axL.tick_params(labelleft=False)
        axL.set_xlabel(r"$\theta-\theta_0$", fontsize=7, labelpad=1)
        axL.set_title(lab, fontsize=7.2)
        axL.grid(True, alpha=0.12)
        if k == 0:
            axL.set_ylabel(r"$C_{\rm device}(\theta)$", fontsize=7.5)
        # ── bottom: shot-free bias floor vs ε ──
        ax = axs[1, k]
        e = np.array([r["eps"] for r in p["sweep"]]); rr = np.array([r["rmse_rel"] for r in p["sweep"]])
        wr = np.array([r["wrong"] for r in p["sweep"]]) >= d["wrong"]
        if p["window"]:
            ax.axvspan(p["window"][0], p["window"][1], color=C_FD, alpha=0.08, lw=0)
        ax.loglog(e, rr, "-", color=C_FD, lw=1.3)
        ax.loglog(e[~wr], rr[~wr], "o", color=C_FD, ms=2.6, mec="white", mew=0.3)
        ax.loglog(e[wr], rr[wr], "X", color="#1a1a1a", ms=4.5)
        ax.axvline(p["eps_star"], color=C_FD, lw=0.6, ls=":")
        ax.axhline(d["win"], color="#888888", lw=0.7, ls="--")
        ax.text(p["eps_star"] * 1.1, 0.9, rf"$\varepsilon^*={p['eps_star']:.2f}$", color=C_FD,
                fontsize=6, va="top")
        ax.axhline(p["psr_disp_rel"], color=C_PSR, lw=1.3)
        ax.text(0.021, p["psr_disp_rel"] * 1.15, r"PSR floor $|f''|\,r$ (shared draw)", color=C_PSR,
                fontsize=5.8, va="bottom")
        ax.text(0.021, 0.0045, "NSR: no floor", color=C_NSR, fontsize=5.8, va="bottom")
        ax.set_xlabel(r"FD step $\varepsilon$", fontsize=7.5, labelpad=1)
        ax.set_xlim(0.018, 3.3); ax.set_ylim(0.004, 1.5)
        ax.grid(True, which="both", alpha=0.12)
        ax.tick_params(labelsize=6.5)
        if k > 0:
            ax.tick_params(labelleft=False)
    axs[1, 0].set_ylabel(r"FD bias floor RMSE$/|\nabla C|$ ($N\to\infty$)", fontsize=7.5)
    axs[1, 2].text(0.97, 0.31, rf"$r={d['r']}$; shaded: RMSE $\leq$ {int(100 * d['win'])}%",
                   transform=axs[1, 2].transAxes, fontsize=6, va="top", ha="right", color="#52514e")
    from matplotlib.lines import Line2D
    axs[0, 2].legend(handles=[Line2D([], [], color=C_PSR, lw=2.2, label="shift-rule tangent"),
                              Line2D([], [], color=C_FD, lw=1.1, marker="o", ms=2.6, label=r"FD secant at $\varepsilon^*$")],
                     fontsize=5.6, loc="lower right", frameon=False, handlelength=1.6, borderpad=0.2)
    for out in (FIGDIR, OUT2, OUT3):
        os.makedirs(out, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out, f"F_epssweep.{ext}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    if os.environ.get("REPLOT") and os.path.exists(CACHE):
        d = json.load(open(CACHE))
    else:
        d = compute()
        json.dump(d, open(CACHE, "w"), indent=1)
    render(d)
    for p in d["panels"]:
        print(f"T={p['T']:<4g} θ0={p['theta0']:.3f} f'={p['f1']:+.3f} f''={p['f2']:+.2f} f'''={p['f3']:+.1f} | "
              f"ε*={p['eps_star']:.3f} (B.6.4 {p['eps_star_b64']:.3f}) floor {100 * p['floor_rel']:.1f}% "
              f"(pred {100 * p['floor_pred_rel']:.1f}%, B.6.4 {100 * p['floor_b64_rel']:.1f}%) | "
              f"window {p['window']} = {p['window_decades']:.2f} decades | PSR disp {100 * p['psr_disp_rel']:.1f}%")
    print("wrote F_epssweep.pdf/.png ->", FIGDIR, OUT2, OUT3)


if __name__ == "__main__":
    main()
