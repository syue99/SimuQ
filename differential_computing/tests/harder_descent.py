"""
harder_descent.py — the HARD Case B: sharp × asymmetric × inhomogeneous.

C3 system (θ1·Z0 + θ2·Z0Z1 + X-drive, cost <Z0>) at T = 2.5 (sharper
landscape; mean exposure T/T̄2* = 0.25, qubit-0 at 0.37 — inside first-order
validity), per-qubit rates r = Γ0/Γ1 = 3 (T2* = 6.7 / 20).  Six estimators
descend from the same start at matched η:

  ideal | raw (= oracle-FD floor) | rescaled-naive (mean-rate model) |
  rescaled-aware (per-qubit map) | FD ε=0.3 | FD ε=0.6

The start is AUTO-SEARCHED on full-trajectory outcomes: ideal GD must converge
to the best reachable basin; FD ε=0.6 must fail; score prefers starts where
raw/naive land measurably away from ideal's endpoint (qualitative separation),
with the step-size sanity lesson built in (η small; ideal convergence is a
hard filter, so estimator failures can't be step-size artifacts).

Run:  conda run -n qec_pg python differential_computing/tests/harder_descent.py
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

T = 2.5
GBAR = 1.0 / (2.0 * 10.0)
RATIO = 3.0
G0 = RATIO * 2.0 * GBAR / (1.0 + RATIO)
G1 = 2.0 * GBAR / (1.0 + RATIO)
H_FD = 1e-3
EPS_FLOORS = [0.3, 0.6]
G_COMP, FAC_LO, FAC_HI = 0.10, 0.25, 4.0
ETA, STEPS = 0.08, 90
DOM = (0.1, 1.6)

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
                                n_grid=80)
        Xm[i, 0] = (D(t1 + H_FD, t2) - D(t1 - H_FD, t2)) / (2 * H_FD)
        Xm[i, 1] = (D(t1, t2 + H_FD) - D(t1, t2 - H_FD)) / (2 * H_FD)
    return Xm


def gate(g_raw, fac):
    return np.where((np.abs(g_raw) >= G_COMP) & (fac >= FAC_LO)
                    & (fac <= FAC_HI), fac, 1.0)


def make_fields():
    def g_ideal(a, b):
        return grad2(fclean, a, b)

    def g_raw(a, b):
        return grad2(fnoisy, a, b)

    def g_resc(a, b, aware):
        gr = grad2(fnoisy, a, b)
        gt = grad2(fclean, a, b)
        Xm = per_qubit_X(a, b)
        if aware:
            lam = np.exp((G0 * Xm[0] + G1 * Xm[1]) / gt)
        else:
            lam = np.exp(GBAR * (Xm[0] + Xm[1]) / gt)
        return gr * gate(gr, 1.0 / lam)

    fields = {"ideal": g_ideal, "raw": g_raw,
              "aware": lambda a, b: g_resc(a, b, True),
              "naive": lambda a, b: g_resc(a, b, False)}
    for e in EPS_FLOORS:
        fields[f"fd{e}"] = lambda a, b, e=e: grad2(fnoisy, a, b, e)
    return fields


def descend(grad_fn, start):
    p = np.array(start, float)
    path = [p.copy()]
    for _ in range(STEPS):
        p = np.clip(p - ETA * grad_fn(p[0], p[1]), DOM[0], DOM[1])
        path.append(p.copy())
    return np.array(path)


def main():
    figdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    os.makedirs(figdir, exist_ok=True)
    cache = os.path.join(figdir, "harder_descent_data.json")

    if os.path.exists(cache):
        d = json.load(open(cache))
        print("loaded cache — replotting only")
    else:
        fields = make_fields()
        gx = np.linspace(DOM[0], DOM[1], 41)
        L = np.array([[fclean(a, b) for a in gx] for b in gx])
        c_best = float(L.min())
        print(f"landscape: C ∈ [{L.min():.3f}, {L.max():.3f}]")

        # Candidate pool: basin-BOUNDARY starts (a grid neighbor's ideal
        # endpoint differs → the start sits on a separatrix) plus the top
        # FD0.6-misdirected ones.  Cheap scoring runs only ideal/raw/fd0.6
        # trajectories (no factor computations); the full estimator set runs
        # once at the winner.  Ideal-converges stays a hard filter so failures
        # can't be step-size artifacts.
        ideal_end = {}
        gs9 = np.linspace(0.3, 1.5, 9)
        for a in gs9:
            for b in gs9:
                p_end = descend(fields["ideal"], (a, b))[-1]
                ideal_end[(round(float(a), 2), round(float(b), 2))] = \
                    fclean(*p_end)
        print("  ideal-endpoint scan done", flush=True)
        boundary = []
        keys = sorted(ideal_end)
        for (a, b) in keys:
            ci = ideal_end[(a, b)]
            if ci > c_best + 0.10:
                continue                     # ideal stuck here — not a start
            for (da, db) in ((0.15, 0), (-0.15, 0), (0, 0.15), (0, -0.15)):
                nb = (round(a + da, 2), round(b + db, 2))
                if nb in ideal_end and ideal_end[nb] > c_best + 0.10:
                    boundary.append((a, b))  # neighbor falls in the trap basin
                    break
        print(f"  boundary candidates: {boundary}", flush=True)

        best_score, best = -np.inf, None
        for s in boundary:
            t_ideal = descend(fields["ideal"], s)
            ci = fclean(*t_ideal[-1])
            if ci > c_best + 0.10:
                continue
            t_fd = descend(fields[f"fd{EPS_FLOORS[1]}"], s)
            t_raw = descend(fields["raw"], s)
            cf, cr = fclean(*t_fd[-1]), fclean(*t_raw[-1])
            sep = float(np.linalg.norm(t_raw[-1] - t_ideal[-1]))
            score = (cf - ci) + (cr - ci) * 3.0 + 2.0 * sep
            print(f"  start {s}: C ideal {ci:.3f} fd {cf:.3f} raw {cr:.3f}, "
                  f"sep {sep:.2f}, score {score:.3f}", flush=True)
            if score > best_score:
                best_score, best = score, s
        start = best
        print(f"chosen start {start}")

        trajs = {k: descend(fn, start) for k, fn in fields.items()}
        costs = {k: [fclean(*q) for q in p] for k, p in trajs.items()}
        for k in trajs:
            print(f"  {k:>6}: final C = {costs[k][-1]:+.4f}")

        d = dict(gx=list(map(float, gx)), L=[list(map(float, r)) for r in L],
                 start=list(start), c_best=c_best,
                 trajs={k: [list(map(float, q)) for q in p]
                        for k, p in trajs.items()},
                 costs=costs)
        json.dump(d, open(cache, "w"), default=float)
        print(f"cached: {cache}")

    gx = np.array(d["gx"]); L = np.array(d["L"])
    fig, ax = plt.subplots(figsize=(8.2, 6.8), dpi=150)
    cs = ax.contourf(gx, gx, L, levels=26, cmap="RdBu_r", alpha=0.75)
    fig.colorbar(cs, ax=ax, label=r"ideal cost $\langle Z_0\rangle$")
    styles = {"ideal": ("#111111", "--", 2.0, "ideal gradient"),
              "raw": ("#9e9e9e", "-", 2.0, "raw PSR (= oracle-FD floor)"),
              "naive": ("#e65100", "-", 2.0, "rescaled, naive mean-rate"),
              "aware": ("#00897b", "-", 2.6, "rescaled, per-qubit aware"),
              "fd0.3": ("#7b1fa2", "-", 1.8, "FD ε=0.3"),
              "fd0.6": ("#d62728", "-", 1.8, "FD ε=0.6")}
    for k, (c, ls, lw, lab) in styles.items():
        p = np.array(d["trajs"][k])
        ax.plot(p[:, 0], p[:, 1], color=c, ls=ls, lw=lw, marker="o", ms=2.2,
                label=f"{lab} → C={d['costs'][k][-1]:+.3f}")
        ax.plot(*p[-1], marker="D", color=c, ms=6)
    ax.plot(*d["start"], "ks", ms=9, mfc="none", mew=2)
    ax.annotate("start", d["start"], textcoords="offset points",
                xytext=(6, 6), fontsize=9)
    ax.set_xlabel(r"$\theta_1$"); ax.set_ylabel(r"$\theta_2$")
    ax.set_title("HARD Case B — sharp (T=2.5) × asymmetric × inhomogeneous "
                 f"T2* (r=3):\ndescent at matched η={ETA}, {STEPS} steps",
                 fontsize=10.5)
    ax.legend(frameon=True, fontsize=7.6, framealpha=0.92, loc="best")
    axi = ax.inset_axes([0.64, 0.66, 0.34, 0.30])
    for k, (c, ls, lw, _) in styles.items():
        axi.plot(d["costs"][k], color=c, ls=ls, lw=1.2)
    axi.set_xlabel("step", fontsize=7); axi.set_ylabel("C", fontsize=7)
    axi.tick_params(labelsize=6); axi.set_title("convergence", fontsize=7)
    fig.tight_layout()
    out = os.path.join(figdir, "harder_descent.png")
    fig.savefig(out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
