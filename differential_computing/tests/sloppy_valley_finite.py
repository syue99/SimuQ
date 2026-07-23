"""
sloppy_valley_finite.py — the sloppy_valley.png experiment AT FINITE SHOTS.

Same setup as sloppy_valley.py (self-consistent NOISY targets measured at
theta*, so the noisy-cost minimum IS theta*; FD applied BLACK-BOX to the COST
with FLOORED eps=0.3/0.6; PSR differentiates observables + chain rule), swept
over sloppiness 1/w — but every measurement is SHOT-SAMPLED.  High budget so
the surviving effect is the BIAS (the sloppy_valley point), not variance.

  PSR (chain rule) : unbiased estimator of the noisy-cost gradient (lemma) ->
                     converges to theta* (= noisy-cost min, self-consistent).
  FD-of-cost, floored eps : smoothed-gradient offset -> shifted fixed point,
                     AMPLIFIED by the valley (~1/w) -> wrong couplings with a
                     right-looking cost, now confirmed to survive shot noise.

Panels mirror sloppy_valley.png:
  A: valley contour + mean finite-shot descent trajectories (w=0.03).
  B: parameter error ||theta_hat-theta*|| vs condition ratio 1/w.

Reuses cached noisy field.  Run:
  conda run -n qec_pg python differential_computing/tests/sloppy_valley_finite.py
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

T, T2 = 1.5, 10.0
THETA_STAR = (0.8, 0.65)
W_SWEEP = [0.3, 0.1, 0.03]
W_SHOW = 0.03
N_G = 64000                       # high finite budget → bias-dominated
EPS_FLOORS = [0.3, 0.6]
K_STEPS, R_SEEDS, ETA = 260, 16, 0.12
START = (0.35, 0.8)               # sloppy_valley's verified start
POOL, NS_CAP, CELL = 48, 400, 0.05
DOM = (0.2, 1.4)

I2 = qp.qeye(2); X = qp.sigmax(); Z = qp.sigmaz()
ZZ = qp.tensor(Z, Z); X0 = qp.tensor(X, I2); Z0O = qp.tensor(Z, I2)
XD = qp.tensor(X, I2) + qp.tensor(I2, X)
OBS = [ZZ, X0, Z0O]
PSI0 = qp.tensor(qp.basis(2, 0), qp.basis(2, 0))
C_OPS = [np.sqrt(1.0 / (2 * T2)) * qp.tensor(Z, I2),
         np.sqrt(1.0 / (2 * T2)) * qp.tensor(I2, Z)]
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


NF = np.load(os.path.join(FIGDIR, "sloppy_valley_shots_field.npy"))


def noisy(t1, t2): return _bilerp(NF, t1, t2)


# ── PSR pools (lazy) ──────────────────────────────────────────────────────────
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
        H = (s * zz + float(c2) * xd) if pl == 0 else (float(c1) * zz + s * xd)
        val = c1 if pl == 0 else c2
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
    blocks = get_pool(oi, pl, t1, t2); M = len(blocks)
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
    return 2.0 * rng.binomial(n, 0.5 * (1 + np.clip(noisy(t1, t2), -1, 1))) / n - 1


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    cache = os.path.join(FIGDIR, "sloppy_valley_finite_data.json")
    target = noisy(*THETA_STAR).copy()          # self-consistent noisy targets

    def clipd(p):
        return np.array([np.clip(p[0], *DOM), np.clip(p[1], *DOM)])

    def cost_shot(t1, t2, w, n, rng):
        m = sample_obs(t1, t2, n, rng) - target
        return m[0] ** 2 + w * (m[1] ** 2 + m[2] ** 2)

    def descend(method, w, eps, seed, keep_path=False):
        rng = np.random.default_rng(seed)
        p = np.array(START, float); path = [p.copy()]
        wv = np.array([1.0, w, w])
        for _ in range(K_STEPS):
            if method == "psr":
                nres = max(1, N_G // 2)
                r = sample_obs(p[0], p[1], nres, rng) - target
                grad = np.zeros(2)
                for l in range(2):
                    for o in range(3):
                        grad[l] += 2 * wv[o] * r[o] * psr_grad(o, l, p[0], p[1],
                                                               N_G, rng)
            else:                                # FD of the COST, floored eps
                nfd = max(1, N_G // 6)
                grad = np.zeros(2)
                for l in range(2):
                    e = np.zeros(2); e[l] = eps
                    grad[l] = (cost_shot(p[0] + e[0], p[1] + e[1], w, nfd, rng)
                               - cost_shot(p[0] - e[0], p[1] - e[1], w, nfd, rng)) \
                        / (2 * eps)
            p = clipd(p - ETA * grad)
            if keep_path:
                path.append(p.copy())
        return (np.array(path) if keep_path else p)

    if os.path.exists(cache):
        d = json.load(open(cache)); print("loaded cache — replotting only")
    else:
        methods = [("psr", None), ("fd", EPS_FLOORS[0]), ("fd", EPS_FLOORS[1])]
        keys = ["psr", f"fd{EPS_FLOORS[0]}", f"fd{EPS_FLOORS[1]}"]
        d = {"theta_star": list(THETA_STAR), "start": list(START),
             "N_g": N_G, "w_sweep": W_SWEEP, "w_show": W_SHOW,
             "err": {k: [] for k in keys}, "paths": {}}
        for w in W_SWEEP:
            for (meth, eps), key in zip(methods, keys):
                ends = np.array([descend(meth, w, eps, s)
                                 for s in range(R_SEEDS)])
                er = np.hypot(ends[:, 0] - THETA_STAR[0],
                              ends[:, 1] - THETA_STAR[1])
                d["err"][key].append(float(np.sqrt(np.mean(er ** 2))))
                if abs(w - W_SHOW) < 1e-9:
                    d["paths"][key] = np.mean(
                        [descend(meth, w, eps, s, keep_path=True)
                         for s in range(6)], axis=0).tolist()
                print(f"  w={w} {key}: rms {d['err'][key][-1]:.3f}", flush=True)
        # contour
        gx = np.linspace(DOM[0], DOM[1], 41)
        LC = [[float(np.log10((noisy(a, b)[0] - target[0]) ** 2
                              + W_SHOW * ((noisy(a, b)[1] - target[1]) ** 2
                                         + (noisy(a, b)[2] - target[2]) ** 2)
                              + 1e-8)) for a in gx] for b in gx]
        d["gx"] = list(map(float, gx)); d["LC"] = LC
        json.dump(d, open(cache, "w"), default=float)
        print(f"cached: {cache}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.2, 5.0), dpi=150)
    gx = np.array(d["gx"])
    cs = axA.contourf(gx, gx, np.array(d["LC"]), levels=24, cmap="viridis")
    fig.colorbar(cs, ax=axA, label=r"$\log_{10} C(\theta)$")
    sty = {"psr": ("#00897b", "PSR (chain rule)"),
           f"fd{EPS_FLOORS[0]}": ("#7b1fa2", f"FD-of-cost ε={EPS_FLOORS[0]}"),
           f"fd{EPS_FLOORS[1]}": ("#d62728", f"FD-of-cost ε={EPS_FLOORS[1]}")}
    for k, (c, lab) in sty.items():
        p = np.array(d["paths"][k])
        axA.plot(p[:, 0], p[:, 1], color=c, lw=2, marker="o", ms=2, label=lab)
        axA.plot(*p[-1], "D", color=c, ms=8, mec="w")
    axA.plot(*d["theta_star"], "*", color="#ffd600", ms=18, mec="k",
             label=r"true $\theta^*$")
    axA.plot(*d["start"], "ks", ms=8, mfc="none", mew=2)
    axA.set_xlabel(r"$\theta_1$"); axA.set_ylabel(r"$\theta_2$")
    axA.set_title(f"(A) sloppy valley (w={d['w_show']}, N={d['N_g']}/grad): "
                  "mean\nfinite-shot descent — FD-of-cost to wrong parameters")
    axA.legend(frameon=True, fontsize=8, loc="lower right")

    ws = np.array(d["w_sweep"])
    for k, (c, lab) in sty.items():
        axB.loglog(1 / ws, d["err"][k], "o-", color=c, lw=2.4, label=lab)
    axB.set_xlabel(r"condition ratio $1/w$  (sloppiness)")
    axB.set_ylabel(r"parameter error $\|\hat\theta-\theta^*\|$")
    axB.set_title("(B) small direction bias × ill-conditioning = amplified\n"
                  "parameter error (FINITE shots, N=%d/grad)" % d["N_g"])
    axB.legend(frameon=False, fontsize=8.5)
    fig.suptitle("Hamiltonian-learning with a sloppy direction, FINITE SHOTS "
                 f"(T/T2*=0.15, N={d['N_g']}/grad, self-consistent targets): "
                 "PSR recovers θ*;\nfloored-ε FD-of-cost converges to the "
                 "valley-amplified wrong parameters — survives shot noise",
                 fontsize=9.8)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = os.path.join(FIGDIR, "sloppy_valley_finite.png")
    fig.savefig(out); print(f"saved: {out}")


if __name__ == "__main__":
    main()
