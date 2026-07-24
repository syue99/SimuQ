"""
sloppy_valley_hard.py — recovered-parameter CLOUDS for the sloppy ideal-target
landscape (w=0.05, T/T2*=0.15) at N=1K, N=40K, and the infinite-shot BIAS.

Same setup as rescale_shots_scaling.py (ideal targets, stationary scatter from
theta*, oracle-FD vs raw PSR vs rescaled PSR).  Three parameter-space figures
(no overwrite of existing files):
  sloppy_valley_hard_1k.png     — clouds at N=1000  (variance regime)
  sloppy_valley_hard_40k.png    — clouds at N=40000 (bias regime)
  sloppy_valley_hard_biasinf.png— deterministic infinite-shot fixed points
Each shows oracle-FD / raw pinned at the noise-shifted couplings (bias 0.046),
rescaled PSR on theta*.  Oracle eps taken from the scan (0.34 at 1K, 0.05 at
40K) so no re-sweep.  Reuses cached noisy/ideal/D fields.

Run:  conda run -n qec_pg python differential_computing/tests/sloppy_valley_hard.py
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
GAMMA = 1.0 / (2.0 * T2)
THETA_STAR = (0.8, 0.65)
W = 0.05
CASES = [("1k", 1000, 0.343414), ("40k", 40000, 0.05)]   # (tag, N, oracle eps)
K_STEPS, BURN, R_SEEDS, ETA = 220, 110, 40, 0.16
POOL, NS_CAP, CELL = 48, 400, 0.05
BOX = ((0.5, 1.1), (0.4, 0.9))

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


NF = np.load(os.path.join(FIGDIR, "sloppy_valley_shots_field.npy"))
IF = np.load(os.path.join(FIGDIR, "rescale_cl_ideal.npy"))
DF = np.load(os.path.join(FIGDIR, "rescale_cl_D.npy"))


def noisy(t1, t2): return _bilerp(NF, t1, t2)
def idealv(t1, t2): return _bilerp(IF, t1, t2)
def Dvec(t1, t2): return _bilerp(DF, t1, t2)


def gfield(F, t1, t2, h=_GS):
    return np.stack([(_bilerp(F, t1 + h, t2) - _bilerp(F, t1 - h, t2)) / (2 * h),
                     (_bilerp(F, t1, t2 + h) - _bilerp(F, t1, t2 - h)) / (2 * h)],
                    axis=1)


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
    cache = os.path.join(FIGDIR, "sloppy_valley_hard_data.json")
    target = idealv(*THETA_STAR).copy()
    weights = np.array([1.0, W, W])

    def clip(p):
        return np.array([np.clip(p[0], *BOX[0]), np.clip(p[1], *BOX[1])])

    START = (1.05, 0.86)          # up the valley; loop pulls raw/FD to the bias

    def descend(method, N_g, eps, seed, deterministic=False):
        rng = np.random.default_rng(seed)
        p = np.array(START, float); path = [p.copy()]
        n_res = max(1, (N_g or 2) // 2)
        for step in range(K_STEPS):
            if deterministic:
                meas = noisy(p[0], p[1]).copy()
            else:
                meas = sample_obs(p[0], p[1], n_res, rng)
            if method == "rescaled":
                meas = meas - GAMMA * Dvec(p[0], p[1])
            r = meas - target
            lam = None
            if method == "rescaled":
                gn = gfield(NF, p[0], p[1]); gi = gfield(IF, p[0], p[1])
                lam = np.where(np.abs(gi) > 0.1, gn / gi, 1.0)
            grad = np.zeros(2)
            for l in range(2):
                for o in range(3):
                    if method == "fd":
                        e = np.zeros(2); e[l] = eps
                        if deterministic:
                            gd = (noisy(p[0] + e[0], p[1] + e[1])[o]
                                  - noisy(p[0] - e[0], p[1] - e[1])[o]) / (2 * eps)
                        else:
                            nfd = max(1, N_g // 2)
                            gd = (sample_obs(p[0]+e[0], p[1]+e[1], nfd, rng)[o]
                                  - sample_obs(p[0]-e[0], p[1]-e[1], nfd, rng)[o]) \
                                / (2 * eps)
                    elif deterministic:
                        gd = gfield(NF, p[0], p[1])[o, l]
                        if method == "rescaled" and 0.25 <= lam[o, l] <= 4.0:
                            gd = gd / lam[o, l]
                    else:
                        gd = psr_grad(o, l, p[0], p[1], N_g, rng)
                        if method == "rescaled" and 0.25 <= lam[o, l] <= 4.0:
                            gd = gd / lam[o, l]
                    grad[l] += 2 * weights[o] * r[o] * gd
            p = clip(p - ETA * grad)
            path.append(p.copy())
        return np.array(path)

    def mean_traj(method, N_g, eps, n_seed=8):
        return np.mean([descend(method, N_g, eps, s) for s in range(n_seed)],
                       axis=0)

    # deterministic infinite-shot fixed points (grid min of each cost)
    def cost_of(fv, t1, t2):
        m = fv(t1, t2) - target
        return m[0] ** 2 + W * (m[1] ** 2 + m[2] ** 2)

    def corr(t1, t2):
        return noisy(t1, t2) - GAMMA * Dvec(t1, t2)

    gg = np.linspace(BOX[0][0], BOX[0][1], 161)
    hh = np.linspace(BOX[1][0], BOX[1][1], 161)
    fp = {}
    for name, fv in (("noisy", noisy), ("corrected", corr)):
        best, bp = np.inf, None
        for a in gg:
            for b in hh:
                c = cost_of(fv, a, b)
                if c < best:
                    best, bp = c, (float(a), float(b))
        fp[name] = bp
    print(f"infinite-shot fixed points: noisy(raw/FD) {np.round(fp['noisy'],4)} "
          f"err {np.hypot(*(np.array(fp['noisy'])-THETA_STAR)):.4f}; "
          f"corrected(rescaled) {np.round(fp['corrected'],4)} "
          f"err {np.hypot(*(np.array(fp['corrected'])-THETA_STAR)):.4f}", flush=True)

    # ideal-cost contour (min at theta*): shows the TRUE valley the loop should
    # descend; raw/FD trajectories peel off to the noise bias, rescaled stays.
    gx = np.linspace(0.55, 1.15, 60)
    LC = [[float(np.log10((idealv(a, b)[0] - target[0]) ** 2
                          + W * ((idealv(a, b)[1] - target[1]) ** 2
                                 + (idealv(a, b)[2] - target[2]) ** 2) + 1e-9))
           for a in gx] for b in gx]

    d = {"fp": fp, "gx": list(map(float, gx)), "LC": LC, "start": list(START),
         "traj": {}}
    ALL = CASES + [("biasinf", None, None)]
    for tag, N_g, eps in ALL:
        d["traj"][tag] = {}
        for method in ("raw", "fd", "rescaled"):
            e = (eps if tag != "biasinf" else 0.05) if method == "fd" else None
            if tag == "biasinf":
                tr = descend(method, None, e, 0, deterministic=True)
            else:
                tr = mean_traj(method, N_g, e)
            d["traj"][tag][method] = tr.tolist()
            print(f"  {tag} {method}: end {np.round(tr[-1],3).tolist()} "
                  f"err {np.hypot(*(tr[-1]-np.array(THETA_STAR))):.4f}",
                  flush=True)
    json.dump(d, open(cache, "w"), default=float)

    cols = {"raw": ("#9e9e9e", "raw PSR"), "fd": ("#7b1fa2", "oracle-FD"),
            "rescaled": ("#00897b", "rescaled PSR")}
    gxp = np.array(d["gx"])

    def loop_fig(tag, N_g, eps):
        fig, ax = plt.subplots(figsize=(6.6, 5.8), dpi=150)
        cs = ax.contourf(gxp, gxp, np.array(d["LC"]), levels=24, cmap="viridis")
        fig.colorbar(cs, ax=ax, label=r"$\log_{10} C_{\rm ideal}(\theta)$")
        for m in ("raw", "fd", "rescaled"):
            tr = np.array(d["traj"][tag][m]); c, lab = cols[m]
            if m == "fd" and eps is not None:
                lab += f" (ε={eps:g})"
            ax.plot(tr[:, 0], tr[:, 1], color=c, lw=2, marker="o", ms=2,
                    label=lab)
            ax.plot(*tr[-1], "D", color=c, ms=8, mec="w")
        ax.plot(*THETA_STAR, "*", color="#ffd600", ms=20, mec="k",
                label=r"true $\theta^*$")
        ax.plot(*d["start"], "ks", ms=8, mfc="none", mew=2)
        ax.set_xlabel(r"$\theta_1$"); ax.set_ylabel(r"$\theta_2$")
        ttl = ("infinite shots (deterministic)" if tag == "biasinf"
               else f"N={N_g}/grad")
        ax.set_title(f"Learning-loop descent, {ttl} (sloppy w={W}, ideal "
                     "targets):\nraw PSR & oracle-FD pulled to the noise bias; "
                     "rescaled PSR to θ*", fontsize=9.3)
        ax.legend(frameon=True, fontsize=8, loc="upper right", framealpha=0.9)
        fig.tight_layout()
        out = os.path.join(FIGDIR, f"sloppy_valley_hard_{tag}.png")
        fig.savefig(out); plt.close(fig); print(f"saved: {out}")

    for tag, N_g, eps in CASES:
        loop_fig(tag, N_g, eps)
    loop_fig("biasinf", None, None)


if __name__ == "__main__":
    main()
