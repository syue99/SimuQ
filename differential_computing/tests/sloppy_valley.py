"""
sloppy_valley.py — small angle error → WRONG PARAMETERS: the ill-conditioned
Hamiltonian-learning case.

Task (IQS-style, self-consistent targets): recover θ* = (0.8, 0.65) by
matching two device-measured observables,
    C(θ) = (⟨Z0Z1⟩(θ) − a*)² + w·(⟨X0⟩(θ) − b*)²,
targets a*, b* measured at θ* on the same noisy device (so the noisy cost's
true minimum IS θ* — estimator differences are purely gradient quality).
The weight w is the SLOPPINESS knob: small w = a narrow valley along the
⟨Z0Z1⟩ level curve, condition number ~ 1/w.

Mechanism this exposes: PSR-family gradients share fixed points with the
exact noisy-cost gradient (positive diagonal Λ cannot move a zero of ∇⟨O⟩,
and residuals vanish at θ*), so they converge to θ* — while floored-ε FD's
SMOOTHED gradient has a DIFFERENT fixed point, and the valley amplifies that
offset by ~1/w along the sloppy direction.  Result: FD lands far from θ*
WITH a deceptively small cost — right-looking convergence, wrong physics.

At ∞ shots the PSR gradient equals the exact noisy-cost gradient by the
(repeatedly validated) lemma, chain-ruled through the residuals — plotted as
one curve "PSR (= exact noisy-cost gradient)".

Run:  conda run -n qec_pg python differential_computing/tests/sloppy_valley.py
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

# T, T2 (hence T/T2*) and output suffix are env-overridable so multiple regimes
# generate without touching the default (0.15): e.g. SV_T2=3 -> 0.5;
# SV_T=0.75 SV_T2=1 -> 0.75.
T = float(os.environ.get("SV_T", "1.5"))
T2 = float(os.environ.get("SV_T2", "10.0"))
SUF = os.environ.get("SV_SUF", "")
THETA_STAR = (0.8, 0.65)
W_SWEEP = [0.3, 0.1, 0.03]
W_SHOW = 0.03                      # panel-A valley
EPS_FLOORS = [0.3, 0.6]
# oracle-FD sweeps eps (incl. small) and keeps the min-parameter-error one per w.
# At infinite shots there is no variance penalty for small eps, so the oracle
# drives eps->0 = the exact gradient -> recovers theta* (overlaps PSR): FD's
# failure here is specifically the FLOORED eps, not FD itself.
EPS_ORACLE = [0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6]
H_FD = 1e-3
ETA, STEPS = 0.15, 900
DOM = (0.2, 1.4)

I2 = qp.qeye(2); X = qp.sigmax(); Z = qp.sigmaz()
ZZ = qp.tensor(Z, Z)
X0 = qp.tensor(X, I2)
Z0O = qp.tensor(Z, I2)
XD = qp.tensor(X, I2) + qp.tensor(I2, X)
PSI0 = qp.tensor(qp.basis(2, 0), qp.basis(2, 0))
C_OPS = [np.sqrt(1.0 / (2 * T2)) * qp.tensor(Z, I2),
         np.sqrt(1.0 / (2 * T2)) * qp.tensor(I2, Z)]


def Hq(t1, t2):
    return t1 * ZZ + t2 * XD


_cache_obs = {}


def obs_noisy(t1, t2):
    key = (round(t1, 6), round(t2, 6))
    if key not in _cache_obs:
        rho = PSI0 * PSI0.dag()
        res = qp.mesolve(Hq(t1, t2), rho, [0.0, T], c_ops=C_OPS)
        r = res.states[-1]
        _cache_obs[key] = (float(qp.expect(ZZ, r).real),
                           float(qp.expect(X0, r).real),
                           float(qp.expect(Z0O, r).real))
    return _cache_obs[key]


def main():
    figdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    os.makedirs(figdir, exist_ok=True)
    cache = os.path.join(figdir, f"sloppy_valley{SUF}_data.json")

    if os.path.exists(cache):
        d = json.load(open(cache))
        print("loaded cache — replotting only")
    else:
        a_star, b_star, d_star = obs_noisy(*THETA_STAR)
        print(f"targets at θ*={THETA_STAR}: a*={a_star:.4f}, b*={b_star:.4f}, "
              f"d*={d_star:.4f}")

        # third observable inside the weak term: two weak constraints jointly
        # pin the sloppy direction → θ* is the UNIQUE zero (the 2-observable
        # version was non-identifiable: multiple exact solutions found)
        def cost(t1, t2, w):
            a, b, dd = obs_noisy(t1, t2)
            return (a - a_star) ** 2 + w * ((b - b_star) ** 2
                                            + (dd - d_star) ** 2)

        def grad_cost(t1, t2, w, h):
            return np.array([
                (cost(t1 + h, t2, w) - cost(t1 - h, t2, w)) / (2 * h),
                (cost(t1, t2 + h, w) - cost(t1, t2 - h, w)) / (2 * h)])

        def descend_from(start, w, h):
            p = np.array(start, float)
            for _ in range(STEPS):
                p = np.clip(p - ETA * grad_cost(p[0], p[1], w, h),
                            DOM[0], DOM[1])
            return p

        # START on the valley (A ≈ a*), correct branch, ~0.5 from θ*:
        # candidates on the A-level curve, keep the one from which the EXACT
        # gradient verifiably reaches θ* (avoids the spurious ∇A≈0 critical
        # points of the least-squares cost that trap descent from arbitrary
        # starts — a real hazard this landscape demonstrated).
        cands = []
        for a in np.linspace(DOM[0], DOM[1], 25):
            for b in np.linspace(DOM[0], DOM[1], 25):
                if abs(obs_noisy(a, b)[0] - a_star) > 0.02:
                    continue
                dist = np.hypot(a - THETA_STAR[0], b - THETA_STAR[1])
                if 0.35 <= dist <= 0.75:
                    cands.append((dist, (float(a), float(b))))
        cands.sort(reverse=True)
        start = None
        for dist, s in cands:
            p_end = descend_from(s, W_SWEEP[-1], H_FD)
            err = np.hypot(p_end[0] - THETA_STAR[0], p_end[1] - THETA_STAR[1])
            print(f"  start candidate {s} (dist {dist:.2f}): exact-gradient "
                  f"err {err:.3f}", flush=True)
            if err < 0.05:
                start = s
                break
        if start is None:
            start = cands[0][1]
            print("  WARNING: no verified start found — using farthest")
        print(f"start {start}")

        def descend(w, h):
            return descend_from(start, w, h)

        rows = []
        oracle_eps = {}
        for w in W_SWEEP:
            row = dict(w=w)
            for name, h in [("psr", H_FD)] + \
                    [(f"fd{e}", e) for e in EPS_FLOORS]:
                p = descend(w, h)
                row[name] = dict(theta=list(map(float, p)),
                                 err=float(np.hypot(p[0] - THETA_STAR[0],
                                                    p[1] - THETA_STAR[1])),
                                 cost=float(cost(p[0], p[1], w)))
            # oracle-FD: best eps (min parameter error) for this w
            best = None
            for e in EPS_ORACLE:
                p = descend(w, e)
                err = float(np.hypot(p[0] - THETA_STAR[0], p[1] - THETA_STAR[1]))
                if best is None or err < best["err"]:
                    best = dict(theta=list(map(float, p)), err=err,
                                cost=float(cost(p[0], p[1], w)), eps=float(e))
            row["oracle_fd"] = best
            oracle_eps[w] = best["eps"]
            rows.append(row)
            print(f"  w={w}: " + "  ".join(
                f"{k} err={row[k]['err']:.3f} C={row[k]['cost']:.2e}"
                for k in row if k != "w"), flush=True)

        # panel-A data: valley contour + trajectories at W_SHOW
        gx = np.linspace(DOM[0], DOM[1], 41)
        LC = [[float(np.log10(cost(a, b, W_SHOW) + 1e-8)) for a in gx]
              for b in gx]

        def descend_path(w, h):
            p = np.array(start, float)
            path = [p.copy()]
            for _ in range(STEPS):
                p = np.clip(p - ETA * grad_cost(p[0], p[1], w, h),
                            DOM[0], DOM[1])
                path.append(p.copy())
            return np.array(path)

        paths = {"psr": descend_path(W_SHOW, H_FD)}
        for e in EPS_FLOORS:
            paths[f"fd{e}"] = descend_path(W_SHOW, e)
        paths["oracle_fd"] = descend_path(W_SHOW, oracle_eps[W_SHOW])

        d = dict(theta_star=list(THETA_STAR), rows=rows, start=list(start),
                 gx=list(map(float, gx)), LC=LC, w_show=W_SHOW,
                 paths={k: [list(map(float, q)) for q in p]
                        for k, p in paths.items()})
        json.dump(d, open(cache, "w"), default=float)
        print(f"cached: {cache}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 5.0), dpi=150)
    gx = np.array(d["gx"])
    cs = axA.contourf(gx, gx, np.array(d["LC"]), levels=24, cmap="viridis")
    fig.colorbar(cs, ax=axA, label=r"$\log_{10} C(\theta)$")
    oe = d["rows"][[r["w"] for r in d["rows"]].index(W_SHOW)]["oracle_fd"]["eps"]
    styles = {"psr": ("#00897b", "PSR (= exact noisy-cost gradient)"),
              "oracle_fd": ("#1f77b4", f"oracle-FD (best ε; here {oe:g})"),
              "fd0.3": ("#7b1fa2", "FD ε=0.3 (floored)"),
              "fd0.6": ("#d62728", "FD ε=0.6 (floored)")}
    for k, (c, lab) in styles.items():
        p = np.array(d["paths"][k])
        ls = "--" if k == "oracle_fd" else "-"
        axA.plot(p[:, 0], p[:, 1], color=c, lw=2, marker="o", ms=2,
                 ls=ls, label=lab)
        axA.plot(*p[-1], marker="D", color=c, ms=8, mec="w")
    axA.plot(*d["theta_star"], marker="*", color="#ffd600", ms=18, mec="k",
             label=r"true $\theta^*$")
    axA.plot(*d["start"], "ks", ms=8, mfc="none", mew=2)
    axA.set_xlabel(r"$\theta_1$"); axA.set_ylabel(r"$\theta_2$")
    axA.set_title(f"(A) sloppy valley (w={d['w_show']}): FD converges to "
                  "wrong parameters\n(residuals below realistic shot-noise "
                  "resolution)", fontsize=10)
    axA.legend(frameon=True, fontsize=7.5, framealpha=0.9, loc="lower left")

    ws = [r["w"] for r in d["rows"]]
    for k, (c, lab) in styles.items():
        ls = "--" if k == "oracle_fd" else "-"
        axB.loglog([1 / w for w in ws],
                   [max(r[k]["err"], 1e-4) for r in d["rows"]],
                   "o" + ls, color=c, lw=2.2, label=lab)
        for r in d["rows"]:
            axB.annotate(f"C={r[k]['cost']:.0e}",
                         (1 / r["w"], max(r[k]["err"], 1e-4)),
                         textcoords="offset points", xytext=(5, 4),
                         fontsize=6, color=c)
    axB.set_xlabel(r"condition ratio $1/w$  (sloppiness)")
    axB.set_ylabel(r"parameter error $\|\hat\theta - \theta^*\|$")
    axB.set_title("(B) small direction bias × ill-conditioning\n= amplified "
                  "parameter error (annotations: final cost)", fontsize=10)
    axB.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Hamiltonian-learning with a sloppy direction (T/T2*={T/T2:.2g}, "
                 "∞ shots, self-consistent targets): PSR recovers θ*;\n"
                 "floored-ε FD converges to a shifted fixed point that the "
                 "valley amplifies — wrong physics, right-looking cost",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = os.path.join(figdir, f"sloppy_valley{SUF}.png")
    fig.savefig(out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
