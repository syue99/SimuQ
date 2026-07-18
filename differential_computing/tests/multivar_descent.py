"""
multivar_descent.py (Case B) — the trajectory picture: gradient descent in the
2D parameter space under each estimator's direction field.

C1 system (2q, H = θ1·Z0Z1 + θ2·(X0+X1), cost C = <Z0Z1>, T/T2* = 0.15),
∞ shots (bias-only — single deterministic trajectory per estimator; the
finite-shot spread is Case A-ext's result).  Same step size η and step count
for every estimator, start chosen where floored-ε FD is directionally wrong:

  ideal gradient      — the reference flow into the minimum
  PSR raw             — rotated field (positive-diagonal preconditioner):
                        detours but provably still descends
  PSR rescaled(gated) — follows the ideal flow
  FD ε=0.3 / 0.6      — floored steps: veers, stalls, or climbs

Run:  conda run -n qec_pg python differential_computing/tests/multivar_descent.py
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

T, T2 = 1.5, 10.0
H_FD = 1e-3
EPS_FLOORS = [0.3, 0.6]
G_COMP, FAC_LO, FAC_HI = 0.10, 0.25, 4.0
ETA, STEPS = 0.10, 70
DOM = (0.1, 1.5)

I2 = qp.qeye(2); X = qp.sigmax(); Z = qp.sigmaz()
ZZ = qp.tensor(Z, Z)
XS = qp.tensor(X, I2) + qp.tensor(I2, X)
OBS = ZZ
PSI0 = qp.tensor(qp.basis(2, 0), qp.basis(2, 0))
C_OPS = [np.sqrt(1.0 / (2 * T2)) * qp.tensor(Z, I2),
         np.sqrt(1.0 / (2 * T2)) * qp.tensor(I2, Z)]


def Hq(t1, t2):
    return t1 * ZZ + t2 * XS


def fclean(t1, t2):
    return float(qp.expect(OBS, (-1j * Hq(t1, t2) * T).expm() * PSI0).real)


def fnoisy(t1, t2):
    rho = PSI0 * PSI0.dag()
    res = qp.mesolve(Hq(t1, t2), rho, [0.0, T], c_ops=C_OPS)
    return float(qp.expect(OBS, res.states[-1]).real)


def grad2(f, t1, t2, h):
    return np.array([(f(t1 + h, t2) - f(t1 - h, t2)) / (2 * h),
                     (f(t1, t2 + h) - f(t1, t2 - h)) / (2 * h)])


def factors_at(t1, t2, g_true):
    out = []
    for ell in range(2):
        e = np.zeros(2); e[ell] = H_FD
        dgdG = (ar.dO_dGamma(Hq(t1 + e[0], t2 + e[1]), OBS, PSI0, T, 2,
                             range(2), n_grid=100)
                - ar.dO_dGamma(Hq(t1 - e[0], t2 - e[1]), OBS, PSI0, T, 2,
                               range(2), n_grid=100)) / (2 * H_FD)
        g = g_true[ell] if abs(g_true[ell]) > 1e-9 else 1e-9
        slope = (dgdG / g) / (2.0 * T)
        out.append(ar.rescale_factor(slope, T, T2))
    return np.array(out)


def clipd(p):
    return np.clip(p, DOM[0], DOM[1])


def descend(grad_fn, start):
    p = np.array(start, float)
    path = [p.copy()]
    for _ in range(STEPS):
        p = clipd(p - ETA * grad_fn(p[0], p[1]))
        path.append(p.copy())
    return np.array(path)


def main():
    figdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    os.makedirs(figdir, exist_ok=True)
    cache = os.path.join(figdir, "multivar_descent_data.json")

    if os.path.exists(cache):
        d = json.load(open(cache))
        print("loaded cache — replotting only")
    else:
        # ideal landscape + minimum
        gx = np.linspace(DOM[0], DOM[1], 41)
        L = np.array([[fclean(a, b) for a in gx] for b in gx])   # [j(t2), i(t1)]
        jm, im = np.unravel_index(np.argmin(L), L.shape)
        p_min = (float(gx[im]), float(gx[jm]))
        print(f"ideal minimum ≈ {p_min}, C={L[jm, im]:.3f}")

        # start: hand-picked inside the valley channel that flows left through
        # the sharp band (Case A: FD ε=0.6 misdirects around θ2≈0.5-0.7)
        # toward the interior basin — keeps all trajectories in view.
        start = (1.05, 0.62)
        g_true = grad2(fclean, *start, H_FD)
        g_fd = grad2(fnoisy, *start, EPS_FLOORS[1])
        c = np.dot(g_fd, g_true) / (np.linalg.norm(g_fd)
                                    * np.linalg.norm(g_true))
        print(f"start {start}: |g|={np.linalg.norm(g_true):.2f}, "
              f"FD ε={EPS_FLOORS[1]} angle {np.degrees(np.arccos(np.clip(c, -1, 1))):.0f}°")

        def g_ideal(a, b):
            return grad2(fclean, a, b, H_FD)

        def g_raw(a, b):
            return grad2(fnoisy, a, b, H_FD)

        def g_resc(a, b):
            gr = grad2(fnoisy, a, b, H_FD)
            fac = factors_at(a, b, grad2(fclean, a, b, H_FD))
            fac = np.where((np.abs(gr) >= G_COMP) & (fac >= FAC_LO)
                           & (fac <= FAC_HI), fac, 1.0)
            return gr * fac

        trajs = {"ideal": descend(g_ideal, start),
                 "psr_raw": descend(g_raw, start),
                 "psr_resc": descend(g_resc, start)}
        for e in EPS_FLOORS:
            trajs[f"fd{e}"] = descend(lambda a, b, e=e: grad2(fnoisy, a, b, e),
                                      start)
        costs = {k: [fclean(*q) for q in p] for k, p in trajs.items()}
        for k, p in trajs.items():
            print(f"  {k:>9}: final C_ideal = {costs[k][-1]:+.4f}  "
                  f"dist to min = {np.hypot(*(p[-1] - np.array(p_min))):.3f}")

        d = dict(gx=list(map(float, gx)), L=[list(map(float, r)) for r in L],
                 p_min=p_min, start=start,
                 final={k: dict(C=costs[k][-1],
                                dist=float(np.hypot(*(p[-1] - np.array(p_min)))))
                        for k, p in trajs.items()},
                 costs=costs,
                 trajs={k: [list(map(float, q)) for q in p]
                        for k, p in trajs.items()})
        json.dump(d, open(cache, "w"), default=float)
        print(f"cached: {cache}")

    gx = np.array(d["gx"]); L = np.array(d["L"])
    fig, ax = plt.subplots(figsize=(7.6, 6.4), dpi=150)
    cs = ax.contourf(gx, gx, L, levels=24, cmap="RdBu_r", alpha=0.75)
    fig.colorbar(cs, ax=ax, label=r"ideal cost  $\langle Z_0Z_1\rangle$")
    styles = {"ideal": dict(color="#111111", ls="--", lw=2.0,
                            label="ideal gradient"),
              "psr_raw": dict(color="#9e9e9e", ls="-", lw=2.0,
                              label="PSR raw (rotated, still descends)"),
              "psr_resc": dict(color="#00897b", ls="-", lw=2.6,
                               label="PSR rescaled (gated)"),
              "fd0.3": dict(color="#7b1fa2", ls="-", lw=1.8,
                            label=f"FD ε={EPS_FLOORS[0]}"),
              "fd0.6": dict(color="#d62728", ls="-", lw=1.8,
                            label=f"FD ε={EPS_FLOORS[1]}")}
    for k, st in styles.items():
        p = np.array(d["trajs"][k])
        lab = f"{st.pop('label')} → C={d['final'][k]['C']:+.3f}"
        ax.plot(p[:, 0], p[:, 1], marker="o", ms=2.6, label=lab, **st)
        st["label"] = lab
    ax.plot(*d["start"], marker="s", color="k", ms=9, mfc="none", mew=2)
    ax.annotate("start", d["start"], textcoords="offset points", xytext=(6, 6),
                fontsize=9)
    ax.plot(*d["p_min"], marker="*", color="#ffd600", ms=18, mec="k")
    ax.annotate("ideal min", d["p_min"], textcoords="offset points",
                xytext=(8, -12), fontsize=9)
    ax.set_xlabel(r"$\theta_1$ (ZZ)"); ax.set_ylabel(r"$\theta_2$ (X)")
    ax.set_title("Gradient descent under each estimator's direction field "
                 f"(T/T2*=0.15, ∞ shots, η={ETA}, {STEPS} steps):\n"
                 "rescaled PSR tracks the ideal flow; raw PSR detours but "
                 "descends; floored-ε FD goes elsewhere", fontsize=10)
    ax.legend(frameon=True, fontsize=8, loc="upper right", framealpha=0.9)
    # inset: ideal cost per step
    axi = ax.inset_axes([0.62, 0.08, 0.35, 0.30])
    for k, st in styles.items():
        axi.plot(d["costs"][k], color=st["color"], ls=st["ls"], lw=1.4)
    axi.set_xlabel("step", fontsize=7); axi.set_ylabel("C_ideal", fontsize=7)
    axi.tick_params(labelsize=6)
    axi.set_title("convergence", fontsize=7)
    fig.tight_layout()
    out = os.path.join(figdir, "multivar_descent.png")
    fig.savefig(out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
