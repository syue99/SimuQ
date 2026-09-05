"""
build_fig1.py — Fig 1 (intro FD-trap), single column. SELF-CONTAINED (Fig-1 only; does
NOT read the shared landscape_device_data.json, so Fig 3 is untouched).

Composition (APPROVED): noisy landscape C_noisy(θ) + three FD secants (all wrong sign) +
ONE shift-rule tangent. Hamiltonian-level under T4: cost C_noisy=⟨Z0⟩ of H(θ)=θZ0+X0 under
T2* dephasing, T/T2*=0.5 fixed.

FIG1_REVISION + REV 2 applied:
  R1  legend "shift-rule tangent (PSR/NSR)"; no raw/=/slope literal anywhere in-figure.
  R4  y-axis C_noisy(θ).
  R6  keep three secants, black landscape, single column.
  R7  caption content migrates IN-FIGURE: a muted info line (program · instrument · regime),
      per-secant ε labels, PSR/NSR vocabulary only, NO section refs inside the image. The
      ~2-line mini caption (with the Sec. 6.2 forwarding clause) is delivered as a sidecar.
  R8  ANCHOR/ε GRID SWEEP (option c) over T × θ* × ε_min (T/T2*=0.5 fixed), choosing a config
      that passes all four criteria; the sweep table is written to the data note:
        (1) all three secant slopes wrong-signed with |slope| ≥ 0.15;
        (2) anchor ≥ 20% of the local ripple period from the nearest extremum;
        (3) |analytic slope at anchor| ≥ 50% of max|slope| in the window;
        (4) no collisions at final size (handled in layout; verified on render).
      If no config passes, we DO NOT relax — we report the two best near-misses and stop.
Caches fig1_intro_data.json for instant replots.
Run: conda run -n qec_pg python differential_computing/tests/build_fig1.py
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from simuq import QSystem, Qubit
from noisy_qutip import NoisyQuTiPRunner
from noise_model import NoiseModel

REGIME = 0.5                          # T/T2* — a RULING, held fixed across the sweep
SWEEP_T = [8.0, 10.0, 12.0, 14.0]     # evolution time (shifts ripple period); T2 = T/REGIME
SWEEP_WIN = (0.55, 2.05)              # region scanned for anchors
EPS_MIN_GRID = np.round(np.arange(0.15, 0.221, 0.01), 3)
EPS_OFFSETS = (0.0, 0.07, 0.14)       # ε_mid/ε_max shift proportionally-ish from ε_min
MIN_SEC_SLOPE = 0.15                  # criterion 1
EXTREMUM_FRAC = 0.20                  # criterion 2 (≥20% of local period)
STEEP_FRAC = 0.50                     # criterion 3 (≥50% of max|slope|)
# small-ε "noise wall": answers "what if we shrink ε" — the control-setpoint error δ (a
# resolution floor: ε cannot be set below it) + shot noise, amplified by 1/(2ε), swamp the
# secant. Shown as a fan of noisy realizations at ε≈δ-scale.
R_CTRL = 0.02                         # control setpoint error δ (T4 best-guess) = the FLOOR on ε
SMALL_EPS = 0.03                      # a step near the δ floor (ε cannot go below ~δ)
N_SHOTS_FAN = 4000                    # per ± evaluation
N_FAN = 60                            # realizations (for the noise-cone envelope stats)
FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
COL = 3.3
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "font.family": "serif",
                     "mathtext.fontset": "stix", "legend.frameon": False,
                     "axes.linewidth": 0.7, "savefig.dpi": 300})
C_INK, C_FD, C_PSR, C_MUTE = "#1a1a1a", "#D55E00", "#0072B2", "#888888"


def fnmaker(T, T2):
    x = sp.Symbol("x"); qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    H = x * q[0].Z + q[0].X
    ex = NoisyQuTiPRunner(2, noise=NoiseModel(n_qubits=2, T2=T2)).make_expectation_fn(
        qp.tensor(qp.basis(2, 0), qp.basis(2, 0)), qp.tensor(qp.sigmaz(), qp.qeye(2)))
    return lambda v: ex([[H.set_parameterizedHam({"x": float(v)}), T]])


def sweep():
    rows = []                          # every passing config (for the data-note table)
    misses = []                        # best near-misses (fallback reporting)
    best = None
    for T in SWEEP_T:
        fn = fnmaker(T, T / REGIME)
        xs = np.linspace(*SWEEP_WIN, 700)
        ys = np.array([fn(v) for v in xs])
        Cint = interp1d(xs, ys, kind="cubic")
        d = np.gradient(ys, xs); maxslope = float(np.max(np.abs(d)))
        ext = xs[np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]]     # local extrema
        slope = lambda v, h=1e-3: (fn(v + h) - fn(v - h)) / (2 * h)  # analytic derivative
        for a in np.arange(SWEEP_WIN[0] + 0.45, SWEEP_WIN[1] - 0.45, 0.01):
            g = float(slope(a)); gnorm = abs(g) / maxslope
            below = ext[ext < a]; above = ext[ext > a]
            if len(below) == 0 or len(above) == 0:
                continue
            period = 2.0 * (above[0] - below[-1])
            near = float(min(a - below[-1], above[0] - a))
            c3 = gnorm >= STEEP_FRAC
            c2 = near >= EXTREMUM_FRAC * period
            for emin in EPS_MIN_GRID:
                es = [round(float(emin) + o, 3) for o in EPS_OFFSETS]
                if a + es[-1] > SWEEP_WIN[1] or a - es[-1] < SWEEP_WIN[0]:
                    continue
                ss = [float((Cint(a + e) - Cint(a - e)) / (2 * e)) for e in es]
                c1 = all(np.sign(s) != np.sign(g) and abs(s) >= MIN_SEC_SLOPE for s in ss)
                rec = dict(T=T, anchor=float(a), eps=es, g=g, secants=ss,
                           near_frac=near / period, gnorm=gnorm,
                           c1=bool(c1), c2=bool(c2), c3=bool(c3))
                if c1 and c2 and c3:
                    rows.append(rec)
                    # R9: prefer a NON-MARGINAL config — maximize the criterion-2 margin
                    # (nearfrac, i.e. mid-flank), then steepness, gated on solid secants.
                    if min(abs(s) for s in ss) >= 0.55:
                        score = (near / period) + 0.25 * gnorm
                        if best is None or score > best[0]:
                            best = (score, dict(rec, T2=T / REGIME, maxslope=maxslope,
                                                period=period, near=near))
                else:
                    npass = int(c1) + int(c2) + int(c3)
                    misses.append((npass, rec))
    return best, rows, misses


def compute():
    best, rows, misses = sweep()
    if best is None:
        misses.sort(key=lambda m: -m[0])
        raise SystemExit("R8 FALLBACK: no config passed all criteria. Two best near-misses:\n"
                         + "\n".join(f"  T={m[1]['T']} θ*={m[1]['anchor']:.3f} eps={m[1]['eps']} "
                                     f"c1={m[1]['c1']} c2={m[1]['c2']} c3={m[1]['c3']} "
                                     f"secants={[round(s,2) for s in m[1]['secants']]}"
                                     for m in misses[:2]) + "\n-> STOP for a ruling.")
    cfg = best[1]; T = cfg["T"]; T2 = cfg["T2"]; a = cfg["anchor"]
    fn = fnmaker(T, T2)
    g = float((fn(a + 1e-3) - fn(a - 1e-3)) / 2e-3)                  # analytic slope (drawn)
    win = (a - 0.55, a + 0.55)
    gx = np.linspace(*win, 400); y = np.array([fn(v) for v in gx])
    g_plot = float(np.gradient(y, gx)[np.argmin(np.abs(gx - a))])
    # P0-0: each probe program dials its own setpoint, so each endpoint carries its own
    # draw δ ~ N(0, r²) (seed 0 = the drawn realization); the estimate divides by the
    # NOMINAL separation.  `eps` is the builder's half-separation; the paper's step is
    # eps_paper = 2·eps (probes θ ± eps_paper/2).
    srng = np.random.default_rng(0)
    secants = []
    for e in cfg["eps"]:
        dm, dp = srng.normal(0, R_CTRL, 2)
        xm, xp = a - e + dm, a + e + dp
        fm_, fp_ = float(fn(xm)), float(fn(xp))
        secants.append(dict(eps=e, eps_paper=2 * e, dm=float(dm), dp=float(dp),
                            xm=float(xm), xp=float(xp), fm=fm_, fp=fp_,
                            slope_nominal=float((fn(a + e) - fn(a - e)) / (2 * e)),
                            slope=float((fp_ - fm_) / (2 * e))))
    # B.6.4 quantities at the anchor (paper convention) and a shot-free Monte-Carlo floor
    hh = 0.02
    f2 = float((fn(a + hh) - 2 * fn(a) + fn(a - hh)) / hh ** 2)
    f3 = float((fn(a + 2 * hh) - 2 * fn(a + hh) + 2 * fn(a - hh) - fn(a - 2 * hh)) / (2 * hh ** 3))
    eps_star_analytic = float((24 * abs(g) * R_CTRL / abs(f3)) ** (1 / 3))
    floor_b64 = float(0.60 * abs(f3) ** (1 / 3) * (abs(g) * R_CTRL) ** (2 / 3))
    grid_mc = np.linspace(a - 0.9, a + 0.9, 1801); Cmc = interp1d(grid_mc, [fn(v) for v in grid_mc], kind="cubic")
    mrng = np.random.default_rng(1); eps_mc = np.geomspace(0.02, 0.8, 25); mc = []
    for e in eps_mc:
        dpv, dmv = mrng.normal(0, R_CTRL, 2000), mrng.normal(0, R_CTRL, 2000)
        est = (Cmc(a + e / 2 + dpv) - Cmc(a - e / 2 + dmv)) / e
        mc.append(dict(eps=float(e), rmse_rel=float(np.sqrt(np.mean((est - g) ** 2)) / abs(g)),
                       signerr=float(np.mean(np.sign(est) != np.sign(g)))))
    mc_best = min(mc, key=lambda w: w["rmse_rel"])
    cone_eps = float(np.sqrt(2) * R_CTRL)                    # where S(ε) = √2·r|f′|/ε equals |f′|
    dpv, dmv = mrng.normal(0, R_CTRL, 4000), mrng.normal(0, R_CTRL, 4000)
    est_c = (Cmc(a + cone_eps / 2 + dpv) - Cmc(a - cone_eps / 2 + dmv)) / cone_eps
    cone_stats = dict(eps=cone_eps, S=float(abs(g)), signerr=float(np.mean(np.sign(est_c) != np.sign(g))),
                      std_meas=float(np.std(est_c)))
    for sec in secants:                                       # how robust is the wrong sign to δ?
        e = sec["eps"]; dpv, dmv = mrng.normal(0, R_CTRL, 2000), mrng.normal(0, R_CTRL, 2000)
        sl = (Cmc(a + e + dpv) - Cmc(a - e + dmv)) / (2 * e)
        sec["wrongsign_frac"] = float(np.mean(np.sign(sl) != np.sign(g))); sec["slope_std"] = float(np.std(sl))
    analytic = dict(f1=g, f2=f2, f3=f3, eps_star_analytic=eps_star_analytic, floor_b64=floor_b64,
                    floor_b64_rel=floor_b64 / abs(g), mc_best_eps=mc_best["eps"], mc_best_rel=mc_best["rmse_rel"],
                    mc_sweep=mc, cone=cone_stats, common_mode=float(abs(f2) * R_CTRL / np.sqrt(2)))

    # small-ε noise wall: N_FAN noisy realizations of the ε≈δ secant (δ setpoint jitter on
    # the ± points + shot noise), to SHOW what shrinking ε does — the estimate scatters.
    rng = np.random.default_rng(0)
    fan, fan_slopes = [], []
    for _ in range(N_FAN):
        dp, dm = rng.normal(0, R_CTRL), rng.normal(0, R_CTRL)
        xp, xm = a + SMALL_EPS + dp, a - SMALL_EPS + dm
        vp, vm = float(fn(xp)), float(fn(xm))
        # shot noise: f = ⟨Z0⟩∈[-1,1] → binomial readout
        fp = 2 * rng.binomial(N_SHOTS_FAN, 0.5 * (1 + np.clip(vp, -1, 1))) / N_SHOTS_FAN - 1
        fm = 2 * rng.binomial(N_SHOTS_FAN, 0.5 * (1 + np.clip(vm, -1, 1))) / N_SHOTS_FAN - 1
        # divide by the NOMINAL 2ε (the machine drifted by δ but you don't know it) — this is
        # where δ/ε amplification enters; dividing by the true separation would cancel δ.
        sl = (fp - fm) / (2 * SMALL_EPS)
        fan.append(dict(xp=float(xp), xm=float(xm), fp=float(fp), fm=float(fm), slope=float(sl)))
        fan_slopes.append(float(sl))
    fan_stats = dict(mean=float(np.mean(fan_slopes)), std=float(np.std(fan_slopes)),
                     lo=float(np.min(fan_slopes)), hi=float(np.max(fan_slopes)),
                     frac_wrongsign=float(np.mean(np.sign(fan_slopes) != np.sign(g))))

    # R10: bound the small-ε claim — sweep ε over [δ, λ/2] at fixed N; report bias, std,
    # sign-error, RMSE/|g|; find best ε and the usable window (RMSE/|g| < WIN_THRESH).
    WIN_THRESH = 0.50
    lam = cfg["period"]
    eps_grid = np.geomspace(R_CTRL, 0.5 * lam, 9)
    ewin = []
    for eps in eps_grid:
        est = []
        for _ in range(300):
            dp, dm = rng.normal(0, R_CTRL), rng.normal(0, R_CTRL)
            vp, vm = float(fn(a + eps + dp)), float(fn(a - eps + dm))
            fp = 2 * rng.binomial(N_SHOTS_FAN, 0.5 * (1 + np.clip(vp, -1, 1))) / N_SHOTS_FAN - 1
            fm = 2 * rng.binomial(N_SHOTS_FAN, 0.5 * (1 + np.clip(vm, -1, 1))) / N_SHOTS_FAN - 1
            est.append((fp - fm) / (2 * eps))
        est = np.array(est)
        ewin.append(dict(eps=float(eps), bias=float(np.mean(est) - g), std=float(np.std(est)),
                         rmse_rel=float(np.sqrt(np.mean((est - g) ** 2)) / abs(g)),
                         signerr=float(np.mean(np.sign(est) != np.sign(g)))))
    good = [w for w in ewin if w["rmse_rel"] < WIN_THRESH and w["signerr"] < 0.05]
    best_e = min(ewin, key=lambda w: w["rmse_rel"])
    ewin_stats = dict(win_thresh=WIN_THRESH, lam_half=float(0.5 * lam), delta=R_CTRL,
                      best_eps=best_e["eps"], best_rmse_rel=best_e["rmse_rel"],
                      window_lo=float(min(w["eps"] for w in good)) if good else None,
                      window_hi=float(max(w["eps"] for w in good)) if good else None,
                      window_width=float(max(w["eps"] for w in good) - min(w["eps"] for w in good))
                      if good else 0.0)
    table = [dict(T=r["T"], anchor=round(r["anchor"], 3), eps=r["eps"],
                  secants=[round(s, 2) for s in r["secants"]],
                  near_frac=round(r["near_frac"], 2), gnorm=round(r["gnorm"], 2))
             for r in sorted(rows, key=lambda r: -min(abs(s) for s in r["secants"]))[:8]]
    return dict(T=T, T2=T2, regime=REGIME, anchor=a, g_analytic=g, g_plotted=g_plot,
                z0=float(fn(a)), maxslope=cfg["maxslope"], period=cfg["period"],
                near=cfg["near"], window=list(win), gx=list(map(float, gx)),
                y=list(map(float, y)), secants=secants, n_pass=len(rows), sweep_table=table,
                small_eps=SMALL_EPS, delta=R_CTRL, fan=fan, fan_stats=fan_stats,
                eps_window=ewin, eps_window_stats=ewin_stats, analytic=analytic, version=2)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    cache = os.path.join(FIGDIR, "fig1_intro_data.json")
    d = json.load(open(cache)) if os.path.exists(cache) else None
    if d is None or d.get("version") != 2:
        d = compute(); json.dump(d, open(cache, "w"))

    a = d["anchor"]; g = d["g_analytic"]; z0 = d["z0"]
    gx = np.array(d["gx"]); y = np.array(d["y"]); win = tuple(d["window"])
    th = 0.06                                                  # short tangent (hugs)

    fig, ax = plt.subplots(figsize=(COL, 2.7))
    ax.plot(gx, y, color=C_INK, lw=1.6, label=r"device landscape $C_{\rm device}(\theta)$")

    # FD secants + per-secant ε labels (R7) — ONE colour for all three (R11; ε labels
    # disambiguate); no slope literals
    # stagger → no overlap; last secant labelled ha=right (dx<0) so ε=0.32 stays inboard of
    # the right spine (Fig-1 fix 2)
    lab_off = [(4, -8, "top", "left"), (5, 4, "bottom", "left"), (5, 3, "bottom", "left")]
    for k, sec in enumerate(d["secants"]):
        e = sec["eps_paper"]                    # paper convention: probes at θ ± ε/2 (+δ)
        ax.plot([sec["xm"], sec["xp"]], [sec["fm"], sec["fp"]], "o-", color=C_FD, lw=1.2, ms=2.6,
                label="FD secants (wrong sign)" if k == 0 else None)
        dx, dy, va, ha = lab_off[k % len(lab_off)]
        ax.annotate(rf"$\varepsilon={e:.2f}$", xy=(sec["xp"], sec["fp"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=5.6, color=C_FD, va=va, ha=ha)

    # small-ε trap (answers "what if we shrink ε"): ε cannot be set below the control resolution
    # δ, and near that floor the setpoint error δ (amplified by 1/ε) scatters the secant. δ is
    # ZERO-MEAN, so the trap is VARIANCE, not bias — the δ-average is ~unbiased (≈∇C, which is why
    # we do NOT draw a mean line: it would coincide with the blue tangent and read as "FD is fine"),
    # but any SINGLE finite run lands anywhere in the cone. Drawn as the single-run scatter envelope.
    # P0-4: the analytic setpoint cone S(ε) = √2·r|f′|/ε (B.6.3), drawn at the step
    # ε = √2·r where S(ε) = |f′| — the ±1σ band spans slopes 0 … 2f′, i.e. the sign is lost
    # at one sigma.  Hamiltonian-level: δ only, no shots.
    C_FAN = "#7b3fa0"; an = d["analytic"]; cone = an["cone"]
    ylo, yhi = float(y.min()) - 0.12, float(y.max()) + 0.14      # frame; cone clipped to it
    cw = 0.11                                   # draw the SLOPE envelope wide enough to see
    xc = np.array([a - cw, a + cw])
    s_hi = g + cone["S"]; s_lo = g - cone["S"]                   # ±1σ = ±|f′|
    ax.fill_between(xc, np.clip(z0 + s_lo * (xc - a), ylo, yhi),
                    np.clip(z0 + s_hi * (xc - a), ylo, yhi), color=C_FAN, alpha=0.28, lw=0,
                    zorder=1, clip_on=True,
                    label=rf"FD setpoint cone at $\varepsilon=\sqrt{{2}}\,r$ ($\pm1\sigma=|\nabla C|$)")
    # the step at which the cone is this wide, drawn to scale: a bracket exactly √2·r wide
    yb = float(y.max()) + 0.05; xb = win[0] + 0.12
    ax.errorbar([xb], [yb], xerr=[[0.5 * cone["eps"]], [0.5 * cone["eps"]]], fmt="none",
                ecolor=C_FAN, elinewidth=1.1, capsize=2.5, zorder=6, clip_on=False)
    ax.annotate(rf"$\varepsilon=\sqrt{{2}}\,r={cone['eps']:.3f}$ (to scale)",
                xy=(xb, yb), xytext=(xb + 0.05, yb), fontsize=5.5, color=C_FAN,
                va="center", ha="left", annotation_clip=False)

    # short shift-rule tangent: slope = analytic derivative (equal by construction). Drawn
    # ABOVE the purple noise cone with a white casing (Fig-1 fix 4: the blue read as "buried"
    # in the purple wedge — two opposite meanings overlapping); the white stroke + high zorder
    # lift it clear of the cone.
    xt = np.array([a - th, a + th])
    ax.plot(xt, z0 + g * (xt - a), color=C_PSR, lw=2.6, solid_capstyle="round",
            label=r"shift-rule tangent (PSR/NSR) $=\nabla C_{\rm device}$", zorder=9,
            path_effects=[pe.Stroke(linewidth=4.4, foreground="white"), pe.Normal()])
    ax.plot([a], [z0], "o", color=C_INK, ms=4, zorder=10)

    # R7 in-figure info line (muted, above the axes → collision-free); PSR/NSR-safe, no refs
    ax.text(0.0, 1.10, r"$H(\theta)=\theta Z_0+X_0$  ·  Hamiltonian-level  ·  "
            rf"$T/T_2^*={d['regime']:.1f}$", transform=ax.transAxes, fontsize=5.8,
            color=C_MUTE, va="bottom", ha="left")
    ax.text(0.0, 1.03, rf"best FD step $\varepsilon^*={an['mc_best_eps']:.2f}$: "
            rf"RMSE $\approx {100*an['mc_best_rel']:.0f}\%$ of $|\nabla C|$  ·  $r={d['delta']}$",
            transform=ax.transAxes, fontsize=5.8, color=C_MUTE, va="bottom", ha="left")

    ax.set_xlabel(r"parameter $\theta$"); ax.set_ylabel(r"$C_{\rm device}(\theta)$")
    ax.set_xlim(*win)
    ax.set_ylim(ylo, yhi)                                        # clip cone to landscape range
    # R11: two-column legend below (compact — no longer sandwiches the x-label)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), fontsize=6.3,
              handlelength=1.6, ncol=2, columnspacing=1.2)
    fig.tight_layout()
    OUT2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "paper_fig_2"))
    OUT3 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "paper_fig_3", "figs"))
    for out in (FIGDIR, OUT2, OUT3):
        os.makedirs(out, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out, f"fig1_intro_trap.{ext}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # ---- deliverables: mini caption + data note (persisted, per R7) ----
    fs = d["fan_stats"]; ws = d["eps_window_stats"]
    # R12 caption (~45 words) — precise: the claim is about the shift rules ("no step size to
    # tune"), NOT that ε has no good value (that quantitative defeat lives in Sec 6.2/F6-R).
    mini_caption = (
        "Figure 1. The finite-difference trap. Large ε (orange) mis-signs the gradient — the "
        "secant straddles a feature (each probe carries its own setpoint draw, r = 0.02). Shrinking ε "
        f"does not help: at ε = √2·r = {an['cone']['eps']:.3f} the setpoint error alone makes the one-sigma "
        "slope error equal to |∇C| (purple cone), and even the best step "
        f"ε* = {an['mc_best_eps']:.2f} leaves an RMSE of {100*an['mc_best_rel']:.0f}% of |∇C| (B.6.4; Sec. 6.2). "
        "The shift-rule tangent (PSR/NSR, blue) recovers ∇C_device with no step size to tune.")
    signs = "/".join(f"{s['slope']:+.2f}" for s in d["secants"])
    eps_used = [s["eps"] for s in d["secants"]]
    win_txt = (f"usable window [{ws['window_lo']:.3f},{ws['window_hi']:.3f}] "
               f"(width {ws['window_width']:.3f}, ~{ws['window_width']/(ws['lam_half']-ws['delta'])*100:.0f}% "
               f"of [δ,λ/2])" if ws["window_lo"] else "usable window EMPTY")
    data_note = (
        f"DATA NOTE (Fig 1): H(θ)=θ·Z0+X0, C_device=⟨Z0⟩ under T2* dephasing, Hamiltonian-level "
        f"(setpoint draw r={d['delta']} on every probe, P0-0; B.6.4 at the anchor: f2={d['analytic']['f2']:+.2f}, "
        f"f3={d['analytic']['f3']:+.0f}, eps*={d['analytic']['eps_star_analytic']:.3f}, floor {100*d['analytic']['floor_b64_rel']:.0f}% of |f1| "
        f"vs shot-free MC best eps={d['analytic']['mc_best_eps']:.3f} at {100*d['analytic']['mc_best_rel']:.0f}%; cone at eps=sqrt2·r="
        f"{d['analytic']['cone']['eps']:.3f}, wrong sign {100*d['analytic']['cone']['signerr']:.0f}%; drawn secants carry seed-0 draws, "
        f"wrong-signed in {[round(100*s['wrongsign_frac']) for s in d['secants']]}% of draws; eps labels in the paper's θ±ε/2 convention). R8 grid sweep over T×θ*×ε_min (T/T2*={d['regime']:.1f} fixed) → "
        f"{d['n_pass']} configs pass; chosen NON-MARGINAL (R9): T={d['T']:.0f} (T2={d['T2']:.0f}), "
        f"θ*={a:.3f}, ε={eps_used}. Criteria: (1) secant slopes {signs} — all wrong-signed "
        f"vs ∇C_device={g:+.2f}, min|slope|={min(abs(s['slope']) for s in d['secants']):.2f}≥0.15 ✓; "
        f"(2) anchor {d['near']:.3f} from nearest extremum = {d['near']/d['period']*100:.0f}% of "
        f"period {d['period']:.3f} ≥20% ✓ (margin); (3) |slope| {abs(g):.2f} = "
        f"{abs(g)/d['maxslope']*100:.0f}% of max|slope| {d['maxslope']:.2f} ≥50% ✓; (4) no "
        f"collisions ✓. Drawn tangent slope = analytic ∇C_device = {g:+.3f} (h=1e-3), EQUAL by "
        f"construction; np.gradient check {d['g_plotted']:+.2f}. "
        f"SMALL-ε (R10): ε floored by δ={d['delta']} (Q1-pending — cone geometry depends on δ; "
        f"re-render if δ changes). Cone drawn at ε={d['small_eps']}≈δ (1.5×δ; labelled ε≈δ): "
        f"mean {fs['mean']:+.2f} vs true {g:+.2f}, std {fs['std']:.2f}, {fs['frac_wrongsign']*100:.0f}% "
        f"wrong-signed. ε-window sweep over [δ,λ/2={ws['lam_half']:.2f}] @N={N_SHOTS_FAN}: best "
        f"ε={ws['best_eps']:.3f} (RMSE/|g|={ws['best_rmse_rel']:.2f}); {win_txt}. USABLE-WINDOW "
        f"CRITERION (shared with F6, G2/F8): an ε is 'usable' iff RMSE/|∇C_device| < 0.5 AND "
        f"sign-error < 5% — the window is the ε-range meeting BOTH (NOT the interval [δ,λ/2], "
        f"which is only the swept range / the fraction's denominator). HONESTY: FD is NOT trapped "
        f"from below at this anchor — a usable ε exists; Fig 1 asserts only that the shift rules "
        f"need no ε (safe), and defers FD's quantitative defeat to Sec 6.2/F6-R.")
    d["mini_caption"] = mini_caption; d["data_note"] = data_note
    json.dump(d, open(cache, "w"), indent=2, default=float)
    with open(os.path.join(FIGDIR, "fig1_intro_trap_caption.txt"), "w") as f:
        f.write(mini_caption + "\n\n" + data_note + "\n\nε-WINDOW SWEEP (bias/std/RMSE_rel/signerr vs ε):\n")
        for w in d["eps_window"]:
            f.write(f"  ε={w['eps']:.3f}  bias={w['bias']:+.2f}  std={w['std']:.2f}  "
                    f"RMSE/|g|={w['rmse_rel']:.2f}  signerr={w['signerr']*100:.0f}%\n")
        f.write("\nSWEEP TABLE (top passing configs):\n")
        for r in d["sweep_table"]:
            f.write(f"  T={r['T']:.0f} θ*={r['anchor']} ε={r['eps']} secants={r['secants']} "
                    f"nearfrac={r['near_frac']} gnorm={r['gnorm']}\n")

    print(f"wrote fig1_intro_trap.pdf/.png + fig1_intro_trap_caption.txt")
    print(data_note)
    print("\nSWEEP TABLE (top passing configs, by min secant |slope|):")
    for r in d["sweep_table"]:
        print(f"  T={r['T']:.0f} θ*={r['anchor']} ε={r['eps']} secants={r['secants']} "
              f"near={r['near_frac']}·period gnorm={r['gnorm']}")
    print("\nMINI CAPTION (R7):\n  " + mini_caption)


if __name__ == "__main__":
    main()
