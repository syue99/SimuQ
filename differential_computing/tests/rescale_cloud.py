"""
rescale_cloud.py — SAME two-panel format as sloppy_valley_shots.png, now with
RESCALED PSR added and IDEAL targets so the rescale shows its advantage.

Setup identical to sloppy_valley_shots (2q, theta*=(0.8,0.65), 3 observables,
cost C = (Abar-A*)^2 + w[(Bbar-B*)^2 + (Dbar-Dd*)^2], stationary scatter from
theta*, fair shared-residual protocol, oracle-eps FD) EXCEPT:
  - targets A*,B*,Dd* are the IDEAL (noiseless) values at theta* (the true
    physics we want to recover);
  - three estimators: raw PSR, oracle-FD, RESCALED PSR.

  raw PSR / FD : residual uses measured NOISY <O>; gradient = noisy.  Drift to
                 the noise-shifted couplings (measured=ideal_target) != theta*.
  RESCALED PSR : residual uses first-order CORRECTED <O> = measured - Gamma*D
                 (~ ideal <O>); gradient = noisy / lambda (rescaled to ideal).
                 Stays at theta*.

Panel A: recovered-parameter clouds at N_show — raw/FD offset to the noise-
         shifted point, rescaled centered on theta*.
Panel B: RMS ||theta_hat-theta*|| vs shots — raw/FD FLOOR at the noise bias,
         only rescaled converges.

Reuses cached noisy/ideal/D fields (sloppy_valley_shots_field.npy,
rescale_cl_ideal.npy, rescale_cl_D.npy).

Run:  conda run -n qec_pg python differential_computing/tests/rescale_cloud.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sympy as sp
import qutip as qp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simuq import QSystem, Qubit
from observable_program_generator import observable_program_generator
import analytic_rescale as ar

T, T2 = 1.5, 10.0
GAMMA = 1.0 / (2.0 * T2)
THETA_STAR = (0.8, 0.65)
W = 0.1
N_BUDGETS = [1000, 4000, 16000, 64000, 256000]
N_SHOW = 16000
EPS_TUNE = np.geomspace(0.1, 0.9, 9)
K_STEPS, BURN, R_SEEDS, ETA = 200, 100, 32, 0.18
POOL, NS_CAP, CELL = 48, 400, 0.05
BOX = ((0.55, 1.05), (0.45, 0.85))

I2 = qp.qeye(2); X = qp.sigmax(); Z = qp.sigmaz()
ZZ = qp.tensor(Z, Z); X0 = qp.tensor(X, I2); Z0O = qp.tensor(Z, I2)
XD = qp.tensor(X, I2) + qp.tensor(I2, X)
OBS = [ZZ, X0, Z0O]
PSI0 = qp.tensor(qp.basis(2, 0), qp.basis(2, 0))
C_OPS = [np.sqrt(GAMMA) * qp.tensor(Z, I2), np.sqrt(GAMMA) * qp.tensor(I2, Z)]
FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
_GLO, _GHI, _GS = -0.1, 1.7, 0.02
_gx = np.arange(_GLO, _GHI + _GS, _GS)


def Hq(t1, t2):
    return t1 * ZZ + t2 * XD


def _bilerp(F, t1, t2):
    fx = (np.clip(t1, _GLO, _GHI) - _GLO) / _GS
    fy = (np.clip(t2, _GLO, _GHI) - _GLO) / _GS
    i0 = min(int(np.floor(fx)), len(_gx) - 2)
    j0 = min(int(np.floor(fy)), len(_gx) - 2)
    dx = fx - i0; dy = fy - j0
    return ((1 - dx) * (1 - dy) * F[i0, j0] + dx * (1 - dy) * F[i0 + 1, j0]
            + (1 - dx) * dy * F[i0, j0 + 1] + dx * dy * F[i0 + 1, j0 + 1])


def _load_fields():
    n = len(_gx); rho0 = PSI0 * PSI0.dag()
    fn = os.path.join(FIGDIR, "sloppy_valley_shots_field.npy")
    fi = os.path.join(FIGDIR, "rescale_cl_ideal.npy")
    fd = os.path.join(FIGDIR, "rescale_cl_D.npy")
    NF = np.load(fn)
    IF = np.load(fi) if os.path.exists(fi) else None
    DF = np.load(fd) if os.path.exists(fd) else None
    if IF is None or DF is None:
        IF = np.zeros((n, n, 3)); DF = np.zeros((n, n, 3))
        for i, a in enumerate(_gx):
            for j, b in enumerate(_gx):
                psi = (-1j * Hq(a, b) * T).expm() * PSI0
                IF[i, j] = [float(qp.expect(o, psi).real) for o in OBS]
                for oi, o in enumerate(OBS):
                    DF[i, j, oi] = ar.dO_dGamma(Hq(a, b), o, PSI0, T, 2,
                                                range(2), n_grid=60)
            if i % 10 == 0:
                print(f"  ideal/D row {i}/{n}", flush=True)
        np.save(fi, IF); np.save(fd, DF)
    return NF, IF, DF


NF, IF, DF = _load_fields()


def noisy(t1, t2): return _bilerp(NF, t1, t2)
def idealv(t1, t2): return _bilerp(IF, t1, t2)
def Dvec(t1, t2): return _bilerp(DF, t1, t2)


def grad_noisy(t1, t2, h=_GS):
    return np.stack([(noisy(t1 + h, t2) - noisy(t1 - h, t2)) / (2 * h),
                     (noisy(t1, t2 + h) - noisy(t1, t2 - h)) / (2 * h)], axis=1)


def grad_ideal(t1, t2, h=_GS):
    return np.stack([(idealv(t1 + h, t2) - idealv(t1 - h, t2)) / (2 * h),
                     (idealv(t1, t2 + h) - idealv(t1, t2 - h)) / (2 * h)], axis=1)


# ── PSR pools (lazy per cell) ─────────────────────────────────────────────────
_pools = {}


def _pool_expfn(H_list, obs):
    rho = PSI0 * PSI0.dag()
    for k, (Hs, dur) in enumerate(H_list):
        Hqo = Hs.to_qutip_qobj()
        if k == 1:
            U = (-1j * Hqo * float(dur)).expm(); rho = U * rho * U.dag()
        else:
            rho = qp.mesolve(Hqo, rho, [0, float(dur)], c_ops=C_OPS).states[-1]
    return float(qp.expect(obs, rho).real)


def get_pool(oi, pl, t1, t2):
    c1 = round(round(t1 / CELL) * CELL, 3); c2 = round(round(t2 / CELL) * CELL, 3)
    key = (oi, pl, c1, c2)
    if key not in _pools:
        qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
        s = sp.Symbol("v"); zz = q[0].Z * q[1].Z; xd = q[0].X + q[1].X
        if pl == 0:
            H = s * zz + float(c2) * xd; val = c1
        else:
            H = float(c1) * zz + s * xd; val = c2
        orig = np.random.rand
        np.random.rand = lambda k: (np.arange(k) + 0.5) / k
        try:
            progs = observable_program_generator(H, T, n_sample=POOL,
                                                 n_repetition=1, diff_var="v",
                                                 value=float(val))
        finally:
            np.random.rand = orig
        blocks = []
        for H_tot, ug, _ in progs:
            b = len(H_tot) // 2
            em = np.array([_pool_expfn(H_tot[2 * i], OBS[oi]) for i in range(b)])
            ep = np.array([_pool_expfn(H_tot[2 * i + 1], OBS[oi])
                           for i in range(b)])
            blocks.append((em, ep, float(ug)))
        _pools[key] = blocks
    return _pools[key]


def psr_grad(oi, pl, t1, t2, N_g, rng):
    blocks = get_pool(oi, pl, t1, t2)
    M = len(blocks)
    ns = int(min(NS_CAP, max(1, N_g // (2 * M))))
    npr = int(max(1, round(N_g / (2 * M * ns))))
    g = 0.0
    for (em, ep, ug) in blocks:
        idx = rng.integers(0, len(em), size=ns)
        fm = 2.0 * rng.binomial(npr, 0.5 * (1 + np.clip(em[idx], -1, 1))) / npr - 1
        fp = 2.0 * rng.binomial(npr, 0.5 * (1 + np.clip(ep[idx], -1, 1))) / npr - 1
        g += (T / ns) * ug * np.sum(fm - fp)
    return g


def sample_obs(t1, t2, n, rng):
    ex = noisy(t1, t2)
    return 2.0 * rng.binomial(n, 0.5 * (1 + np.clip(ex, -1, 1))) / n - 1.0


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    cache = os.path.join(FIGDIR, "rescale_cloud_data.json")
    target = idealv(*THETA_STAR).copy()
    weights = np.array([1.0, W, W])
    print(f"ideal targets {np.round(target,4).tolist()}, "
          f"noisy(theta*) {np.round(noisy(*THETA_STAR),4).tolist()}, "
          f"corrected {np.round(noisy(*THETA_STAR)-GAMMA*Dvec(*THETA_STAR),4).tolist()}")

    def clip(p):
        return np.array([np.clip(p[0], *BOX[0]), np.clip(p[1], *BOX[1])])

    def cloud(method, N_g, eps, seed):
        rng = np.random.default_rng(seed)
        p = np.array(THETA_STAR, float); pts = []
        n_res = max(1, N_g // 2)
        for step in range(K_STEPS):
            meas = sample_obs(p[0], p[1], n_res, rng)
            if method == "rescaled":
                meas = meas - GAMMA * Dvec(p[0], p[1])
            r = meas - target
            lam = None
            if method == "rescaled":
                gn = grad_noisy(p[0], p[1]); gi = grad_ideal(p[0], p[1])
                lam = np.where(np.abs(gi) > 0.1, gn / gi, 1.0)
            grad = np.zeros(2)
            for l in range(2):
                for o in range(3):
                    if method == "fd":
                        e = np.zeros(2); e[l] = eps; nfd = max(1, N_g // 2)
                        gd = (sample_obs(p[0] + e[0], p[1] + e[1], nfd, rng)[o]
                              - sample_obs(p[0] - e[0], p[1] - e[1], nfd, rng)[o]) \
                            / (2 * eps)
                    else:
                        gd = psr_grad(o, l, p[0], p[1], N_g, rng)
                        if method == "rescaled":
                            lo = lam[o, l]
                            if 0.25 <= lo <= 4.0:
                                gd = gd / lo
                    grad[l] += 2 * weights[o] * r[o] * gd
            p = clip(p - ETA * grad)
            if step >= BURN:
                pts.append(p.copy())
        return np.array(pts)

    def stats(cl):
        er = np.hypot(cl[:, 0] - THETA_STAR[0], cl[:, 1] - THETA_STAR[1])
        return dict(rms=float(np.sqrt(np.mean(er ** 2))),
                    mean=[float(cl[:, 0].mean()), float(cl[:, 1].mean())])

    if os.path.exists(cache):
        d = json.load(open(cache)); print("loaded cache — replotting only")
    else:
        d = {"theta_star": list(THETA_STAR), "w": W, "budgets": N_BUDGETS,
             "n_show": N_SHOW, "raw": {}, "fd": {}, "rescaled": {}, "clouds": {}}
        for N_g in N_BUDGETS:
            for method in ("raw", "rescaled"):
                cl = np.vstack([cloud(method, N_g, None, s)
                                for s in range(R_SEEDS)])
                st = stats(cl); d[method][str(N_g)] = st
                if N_g == N_SHOW:
                    d["clouds"][method] = cl.tolist()
                print(f"  N={N_g} {method}: rms {st['rms']:.3f} "
                      f"mean {np.round(st['mean'],3).tolist()}", flush=True)
            best = None
            for eps in EPS_TUNE:
                cl = np.vstack([cloud("fd", N_g, float(eps), s)
                                for s in range(R_SEEDS)])
                st = stats(cl)
                if best is None or st["rms"] < best["rms"]:
                    best = dict(st, eps=float(eps), cloud=cl)
            d["fd"][str(N_g)] = {"rms": best["rms"], "mean": best["mean"],
                                 "eps": best["eps"]}
            if N_g == N_SHOW:
                d["clouds"]["fd"] = best["cloud"].tolist()
            print(f"  N={N_g} fd: rms {best['rms']:.3f} (oracle eps "
                  f"{best['eps']:.2f})", flush=True)
        json.dump(d, open(cache, "w"), default=float)
        print(f"cached: {cache}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.2, 5.0), dpi=150)
    cols = {"raw": "#9e9e9e", "fd": "#7b1fa2", "rescaled": "#00897b"}
    labs = {"raw": "raw PSR", "fd": "oracle-FD", "rescaled": "rescaled PSR"}
    for k in ("raw", "fd", "rescaled"):
        c = np.array(d["clouds"][k])
        axA.scatter(c[:, 0], c[:, 1], s=7, color=cols[k], alpha=0.35,
                    label=labs[k], zorder=2 if k == "rescaled" else 1)
        axA.plot(c[:, 0].mean(), c[:, 1].mean(), "P", color=cols[k], ms=13,
                 mec="w", mew=1.5, zorder=3)
    axA.plot(*THETA_STAR, "*", color="#ffd600", ms=20, mec="k",
             label=r"true $\theta^*$", zorder=4)
    axA.set_xlabel(r"$\theta_1$"); axA.set_ylabel(r"$\theta_2$")
    axA.set_title(f"(A) recovered-parameter cloud (w={d['w']}, N={d['n_show']}"
                  "/grad)\nraw PSR & FD offset to noise-shifted params; "
                  "rescaled on θ*")
    axA.legend(frameon=True, fontsize=8)

    Ns = np.array(d["budgets"])
    for k in ("raw", "fd", "rescaled"):
        axB.loglog(Ns, [d[k][str(n)]["rms"] for n in Ns], "o-", color=cols[k],
                   lw=2.4 if k == "rescaled" else 2.0, label=labs[k])
    axB.set_xlabel("total shots per gradient component $N_g$")
    axB.set_ylabel(r"parameter error $\|\hat\theta-\theta^*\|$")
    axB.set_title("(B) recovery vs shots — raw PSR & FD FLOOR at the noise "
                  "bias;\nonly rescaled PSR converges to θ*")
    axB.legend(frameon=False, fontsize=8.5)
    fig.suptitle("Finite-shot Hamiltonian learning, IDEAL targets "
                 f"(T/T2*=0.15, w={d['w']}): rescaled PSR recovers θ* where raw "
                 "PSR & oracle-FD\nconverge — with more shots — to the "
                 "noise-shifted wrong couplings", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.join(FIGDIR, "rescale_cloud.png")
    fig.savefig(out); print(f"saved: {out}")


if __name__ == "__main__":
    main()
