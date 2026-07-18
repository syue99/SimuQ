"""
inhom_direction_AB.py — Cases A and B under INHOMOGENEOUS per-qubit T2*:
does inhomogeneity widen rescaled-PSR's advantage over oracle-tuned FD?

System: the ASYMMETRIC C3 (θ1·Z0 + θ2·Z0Z1 + X-drive, obs/cost <Z0>) — on a
swap-symmetric system per-qubit responses coincide and inhomogeneity is
invisible at first order.  Rates: r = Γ0/Γ1 = 3 at fixed mean (nominal
T/T2* = 0.15): T2*_0 ≈ 6.7, T2*_1 = 20.

At ∞ shots oracle-FD ≡ raw PSR (lemma, incl. inhomogeneous noise) — so the
"oracle-FD floor" IS the raw map/trajectory.  Estimators:
  raw (= FD best-ε)      : the rotation floor — GROWS under inhomogeneity
  rescaled, naive model  : homogeneous mean-rate factors (no calibration map)
  rescaled, per-qubit    : aware factors (consumes the T2* map)
  FD ε = 0.3 / 0.6       : floored steps

A: 13×13 angle maps + stats (compare vs homogeneous C3 stats: raw 2.11°,
   resc 0.23° median).  B: descent trajectories on the <Z0> landscape with a
   start auto-picked for FD-0.6 misdirection AND verified so that ideal GD
   converges (the Case-B step-size lesson).

Run:  conda run -n qec_pg python differential_computing/tests/inhom_direction_AB.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import qutip as qp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analytic_rescale as ar

T = 1.5
GBAR = 1.0 / (2.0 * 10.0)
RATIO = 3.0
G0 = RATIO * 2.0 * GBAR / (1.0 + RATIO)   # 0.075  (T2*_0 ≈ 6.7)
G1 = 2.0 * GBAR / (1.0 + RATIO)           # 0.025  (T2*_1 = 20)
H_FD = 1e-3
EPS_FLOORS = [0.3, 0.6]
GRID = np.linspace(0.2, 1.4, 13)
G_MIN, G_COMP, FAC_LO, FAC_HI = 0.15, 0.10, 0.25, 4.0
ETA, STEPS = 0.10, 70
DOM = (0.1, 1.5)

I2 = qp.qeye(2); X = qp.sigmax(); Z = qp.sigmaz()
Z0 = qp.tensor(Z, I2); ZZ = qp.tensor(Z, Z)
XD = qp.tensor(X, I2) + qp.tensor(I2, X)
OBS = Z0
PSI0 = qp.tensor(qp.basis(2, 0), qp.basis(2, 0))
C_OPS = [np.sqrt(G0) * qp.tensor(Z, I2), np.sqrt(G1) * qp.tensor(I2, Z)]


def Hq(t1, t2):
    return t1 * Z0 + t2 * ZZ + 1.0 * XD


def fclean(t1, t2):
    return float(qp.expect(OBS, (-1j * Hq(t1, t2) * T).expm() * PSI0).real)


def fnoisy(t1, t2):
    rho = PSI0 * PSI0.dag()
    res = qp.mesolve(Hq(t1, t2), rho, [0.0, T], c_ops=C_OPS)
    return float(qp.expect(OBS, res.states[-1]).real)


def grad2(f, t1, t2, h=H_FD):
    return np.array([(f(t1 + h, t2) - f(t1 - h, t2)) / (2 * h),
                     (f(t1, t2 + h) - f(t1, t2 - h)) / (2 * h)])


def per_qubit_X(t1, t2):
    Xm = np.zeros((2, 2))
    for i in range(2):
        def D(a, b, i=i):
            return ar.dO_dGamma(Hq(a, b), OBS, PSI0, T, 2, z_sites=[i],
                                n_grid=100)
        Xm[i, 0] = (D(t1 + H_FD, t2) - D(t1 - H_FD, t2)) / (2 * H_FD)
        Xm[i, 1] = (D(t1, t2 + H_FD) - D(t1, t2 - H_FD)) / (2 * H_FD)
    return Xm


def factors(t1, t2, g_true):
    """(aware, naive) rescale factors = 1/λ_pred per parameter."""
    Xm = per_qubit_X(t1, t2)
    lam_aware = np.exp((G0 * Xm[0] + G1 * Xm[1]) / g_true)
    lam_naive = np.exp(GBAR * (Xm[0] + Xm[1]) / g_true)
    return 1.0 / lam_aware, 1.0 / lam_naive


def gate(g_raw, fac):
    return np.where((np.abs(g_raw) >= G_COMP) & (fac >= FAC_LO)
                    & (fac <= FAC_HI), fac, 1.0)


def angle_deg(v, ref):
    c = float(np.dot(v, ref) / (np.linalg.norm(v) * np.linalg.norm(ref)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def main():
    figdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    os.makedirs(figdir, exist_ok=True)
    cache = os.path.join(figdir, "inhom_direction_AB_data.json")

    if os.path.exists(cache):
        d = json.load(open(cache))
        print("loaded cache — replotting only")
    else:
        # ── A: angle maps ──
        pts = []
        for i, t1 in enumerate(GRID):
            for j, t2 in enumerate(GRID):
                g_true = grad2(fclean, t1, t2)
                if np.linalg.norm(g_true) < G_MIN:
                    pts.append(dict(i=i, j=j)); continue
                g_raw = grad2(fnoisy, t1, t2)
                fa, fn_ = factors(t1, t2, g_true)
                rec = dict(i=i, j=j,
                           ang_raw=angle_deg(g_raw, g_true),
                           ang_aware=angle_deg(g_raw * gate(g_raw, fa), g_true),
                           ang_naive=angle_deg(g_raw * gate(g_raw, fn_), g_true))
                for e in EPS_FLOORS:
                    rec[f"ang_fd{e}"] = angle_deg(grad2(fnoisy, t1, t2, e),
                                                  g_true)
                pts.append(rec)
            print(f"  A: row {i + 1}/{len(GRID)}", flush=True)

        # ── B: trajectories ──
        gx = np.linspace(DOM[0], DOM[1], 41)
        L = np.array([[fclean(a, b) for a in gx] for b in gx])
        jm, im = np.unravel_index(np.argmin(L), L.shape)
        p_min = (float(gx[im]), float(gx[jm]))
        print(f"  B: ideal min ≈ {p_min}, C={L[jm, im]:.3f}")

        def descend(grad_fn, start):
            p = np.array(start, float)
            path = [p.copy()]
            for _ in range(STEPS):
                p = np.clip(p - ETA * grad_fn(p[0], p[1]), DOM[0], DOM[1])
                path.append(p.copy())
            return np.array(path)

        # start: FD0.6-misdirected AND ideal-GD-convergent (Case-B lesson)
        cands = []
        for a in np.linspace(0.2, 1.4, 13):
            for b in np.linspace(0.2, 1.4, 13):
                dist = np.hypot(a - p_min[0], b - p_min[1])
                if not (0.5 <= dist <= 1.0):
                    continue
                g_true = grad2(fclean, a, b)
                if np.linalg.norm(g_true) < 0.3:
                    continue
                g_fd = grad2(fnoisy, a, b, EPS_FLOORS[1])
                cands.append((angle_deg(g_fd, g_true), (float(a), float(b))))
        cands.sort(reverse=True)
        start = None
        for ang, c in cands[:10]:
            p_end = descend(lambda a, b: grad2(fclean, a, b), c)[-1]
            if fclean(*p_end) < L[jm, im] + 0.08:
                start = c
                print(f"  B: start {c} (FD0.6 angle {ang:.0f}°, ideal GD "
                      f"converges to C={fclean(*p_end):.3f})")
                break
        if start is None:
            start = cands[0][1]
            print(f"  B: fallback start {start}")

        def g_aware(a, b):
            gr = grad2(fnoisy, a, b)
            fa, _ = factors(a, b, grad2(fclean, a, b))
            return gr * gate(gr, fa)

        def g_naive(a, b):
            gr = grad2(fnoisy, a, b)
            _, fn_ = factors(a, b, grad2(fclean, a, b))
            return gr * gate(gr, fn_)

        trajs = {"ideal": descend(lambda a, b: grad2(fclean, a, b), start),
                 "raw": descend(lambda a, b: grad2(fnoisy, a, b), start),
                 "aware": descend(g_aware, start),
                 "naive": descend(g_naive, start)}
        for e in EPS_FLOORS:
            trajs[f"fd{e}"] = descend(
                lambda a, b, e=e: grad2(fnoisy, a, b, e), start)
        costs = {k: [fclean(*q) for q in p] for k, p in trajs.items()}
        for k in trajs:
            print(f"  B: {k:>6} final C = {costs[k][-1]:+.4f}")

        d = dict(pts=pts, gx=list(map(float, gx)),
                 L=[list(map(float, r)) for r in L], p_min=p_min, start=start,
                 trajs={k: [list(map(float, q)) for q in p]
                        for k, p in trajs.items()},
                 costs=costs)
        json.dump(d, open(cache, "w"), default=float)
        print(f"cached: {cache}")

    # ── report ──
    G = len(GRID)
    keys = ["ang_raw", "ang_aware", "ang_naive"] + \
        [f"ang_fd{e}" for e in EPS_FLOORS]
    maps = {k: np.full((G, G), np.nan) for k in keys}
    for r_ in d["pts"]:
        if "ang_raw" in r_:
            for k in keys:
                maps[k][r_["j"], r_["i"]] = r_[k]
    print("\n── A stats (median° / max° / uphill%) — homogeneous C3 was "
          "raw 2.11°, resc 0.23° ──")
    stats = {}
    for k, m in maps.items():
        v = m[~np.isnan(m)]
        stats[k] = (np.median(v), v.max(), 100 * np.mean(v > 90))
        print(f"  {k:>10}: {stats[k][0]:6.2f}° / {stats[k][1]:6.1f}° / "
              f"{stats[k][2]:4.1f}%")

    fig = plt.figure(figsize=(13.6, 5.4), dpi=150)
    gs_ = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.35])
    ext = [GRID[0], GRID[-1], GRID[0], GRID[-1]]
    for ax_i, k, ttl in ((0, "ang_raw", "raw (= FD best-ε / oracle floor)"),
                         (1, "ang_aware", "rescaled, per-qubit aware")):
        ax = fig.add_subplot(gs_[ax_i])
        im = ax.imshow(np.minimum(maps[k], 20), origin="lower", extent=ext,
                       vmin=0, vmax=20, cmap="viridis", aspect="auto")
        ax.set_title(f"(A{ax_i + 1}) {ttl}\nmedian {stats[k][0]:.2f}°",
                     fontsize=9.5)
        ax.set_xlabel(r"$\theta_1$"); ax.set_ylabel(r"$\theta_2$")
    fig.colorbar(im, ax=fig.axes, shrink=0.8, label="angle (deg, cap 20)")

    ax = fig.add_subplot(gs_[2])
    gx = np.array(d["gx"]); L = np.array(d["L"])
    cs = ax.contourf(gx, gx, L, levels=24, cmap="RdBu_r", alpha=0.7)
    styles = {"ideal": ("#111111", "--", "ideal"),
              "raw": ("#9e9e9e", "-", "raw (=FD-best)"),
              "naive": ("#e65100", "-", "rescaled naive"),
              "aware": ("#00897b", "-", "rescaled aware"),
              "fd0.3": ("#7b1fa2", "-", "FD ε=0.3"),
              "fd0.6": ("#d62728", "-", "FD ε=0.6")}
    for k, (c, ls, lab) in styles.items():
        p = np.array(d["trajs"][k])
        ax.plot(p[:, 0], p[:, 1], color=c, ls=ls, lw=1.9, marker="o", ms=2.2,
                label=f"{lab} → C={d['costs'][k][-1]:+.3f}")
    ax.plot(*d["start"], "ks", ms=8, mfc="none", mew=2)
    ax.plot(*d["p_min"], marker="*", color="#ffd600", ms=16, mec="k")
    ax.set_title("(B) descent trajectories", fontsize=10)
    ax.set_xlabel(r"$\theta_1$"); ax.set_ylabel(r"$\theta_2$")
    ax.legend(frameon=True, fontsize=6.8, framealpha=0.9)
    fig.suptitle(f"Inhomogeneous T2* (r=3, T2*=6.7/20, mean T/T2*=0.15), "
                 "asymmetric 2q system: the oracle-FD floor (=raw) GROWS, "
                 "only the map-aware rescale recovers the direction",
                 fontsize=10.5)
    out = os.path.join(figdir, "inhom_direction_AB.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
