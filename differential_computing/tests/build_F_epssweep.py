"""
build_F_epssweep.py — P1-1 (App C.3 companion): FD's bias floor as a function of the step ε
at three operating points of increasing landscape sharpness.  SHOT-FREE (N → ∞): the figure
compares bias floors only, so the shot budget never enters.

Per panel (T = 1, 2.5, 5 µs; T/T₂* = 0.15 held; same 2q TFIM, same device model r = 0.02):
  θ0    the median point of FD's predicted floor among steep points (|∇C| ≥ ½ max), no
        constraint on f'' — f'' and f''' are reported.
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


def pick_theta(g, d1, d2, d3):
    """Median of FD's predicted shot-free floor over steep points (|∇C| ≥ ½ max), θ ∈ [0.5, 3]."""
    eps = np.geomspace(0.01, 2.0, 500)
    m = (g >= 0.5) & (g <= 3.0) & (np.abs(d1) >= 0.5 * np.abs(d1).max())
    fl = []
    for a, b, c in zip(d1[m], d2[m], d3[m]):
        fd = np.sqrt((np.sqrt(2) * R_CTRL * abs(a) / eps) ** 2 + (abs(b) * R_CTRL / np.sqrt(2)) ** 2
                     + (eps ** 2 * abs(c) / 24) ** 2)
        fl.append(fd.min() / abs(a))
    fl = np.array(fl)
    i = int(np.argmin(np.abs(fl - np.median(fl))))
    return float(g[m][i]), float(fl[i])


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
    return dict(T=T, T2=T / REGIME, theta0=th0, f1=f1, f2=f2, f3=f3,
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
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.35), dpi=300, sharey=True)
    labels = ["(a) well-conditioned", "(b) intermediate", "(c) ill-conditioned"]
    for ax, p, lab in zip(axs, d["panels"], labels):
        e = np.array([r["eps"] for r in p["sweep"]]); rr = np.array([r["rmse_rel"] for r in p["sweep"]])
        wr = np.array([r["wrong"] for r in p["sweep"]]) >= d["wrong"]
        if p["window"]:
            ax.axvspan(p["window"][0], p["window"][1], color=C_FD, alpha=0.08, lw=0)
        ax.loglog(e, rr, "-", color=C_FD, lw=1.3)
        ax.loglog(e[~wr], rr[~wr], "o", color=C_FD, ms=2.6, mec="white", mew=0.3)
        ax.loglog(e[wr], rr[wr], "X", color="#1a1a1a", ms=4.5)
        ax.axvline(p["eps_star"], color=C_FD, lw=0.6, ls=":")
        ax.axhline(d["win"], color="#888888", lw=0.7, ls="--")
        if p["psr_disp_rel"] > 0.004:
            ax.axhline(p["psr_disp_rel"], color=C_PSR, lw=1.3)
            ax.text(0.021, p["psr_disp_rel"] * 1.15, "PSR (shared draw)", color=C_PSR, fontsize=5.8, va="bottom")
        else:
            ax.text(0.021, 0.0062, "PSR: no floor at $C''=0$", color=C_PSR, fontsize=5.8, va="bottom")
        ax.text(0.021, 0.0045, "NSR: no floor", color=C_NSR, fontsize=5.8, va="bottom")
        ax.text(p["eps_star"] * 1.08, 0.9, rf"$\varepsilon^*={p['eps_star']:.2f}$", color=C_FD,
                fontsize=6, va="top")
        ax.set_title(lab + rf"  $T={p['T']:g}\,\mu$s, $\theta_0={p['theta0']:.2f}$", fontsize=7)
        ax.text(0.98, 0.04, rf"$|f'''|/|f'|={abs(p['f3'] / p['f1']):.0f}$, $f''r/|f'|={p['psr_disp_rel']:.2f}$",
                transform=ax.transAxes, fontsize=5.8, ha="right", va="bottom", color="#52514e")
        ax.set_xlabel(r"FD step $\varepsilon$", fontsize=7.5)
        ax.set_xlim(0.018, 3.3); ax.set_ylim(0.004, 1.5)
        ax.grid(True, which="both", alpha=0.12)
        ax.tick_params(labelsize=6.5)
    axs[0].set_ylabel(r"bias floor  RMSE$/|\nabla C_{\rm device}|$  ($N\to\infty$)", fontsize=7.5)
    axs[2].text(0.03, 0.40, rf"$T/T_2^*={d['regime']}$, $r={d['r']}$" + "\n" + rf"shaded: RMSE $\leq$ {int(100 * d['win'])}%",
                transform=axs[2].transAxes, fontsize=6, va="top", ha="left", color="#52514e")
    fig.tight_layout(pad=0.4)
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
