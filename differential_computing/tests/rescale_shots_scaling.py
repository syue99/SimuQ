"""
rescale_shots_scaling.py — the closed-loop shots-scaling crossover:
oracle-FD floors at its attenuation BIAS; only rescaled PSR converges.

Ill-conditioned (sloppy, w small) Hamiltonian learning with IDEAL targets, so
the noise attenuation shifts the recovered couplings — an "accurate-gradient-
required" landscape.  Stationary scatter from theta*, three estimators, swept
over total shots from 100 (variance-dominated for everyone, incl. rescaled) up
to ~4e5 (oracle-FD uses small eps -> low variance but hits its BIAS floor):

  oracle-FD    : per-budget best eps (min realized parameter error).  At high
                 shots it picks small eps (NOT eps-dilemma-limited) yet floors
                 at the noise-shifted couplings — a BIAS the variance can't fix.
  raw PSR      : same bias floor (lemma).
  rescaled PSR : first-order-corrected observable + 1/lambda gradient -> the
                 IDEAL cost -> converges to theta* ~N^-1/2, crossing BELOW the
                 oracle-FD floor once shots overcome its (1/lambda-amplified)
                 variance.

Headline: with enough samples the rescale wins because oracle-FD is bias-
limited, not variance-limited.  Reuses cached noisy/ideal/D fields.

Run:  conda run -n qec_pg python differential_computing/tests/rescale_shots_scaling.py
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

T, T2 = 1.5, 10.0                 # T/T2* = 0.15 (validated rescale regime)
GAMMA = 1.0 / (2.0 * T2)
THETA_STAR = (0.8, 0.65)
W = 0.05                          # sloppy → gradient accuracy matters
N_BUDGETS = [int(x) for x in
             os.environ.get("SV_BUDGETS", "100,300,1000,3000,10000").split(",")]
SUF = os.environ.get("SV_SUF", "")
N_SHOW = 6400
EPS_TUNE = np.geomspace(0.05, 0.9, 10)
K_STEPS, BURN, R_SEEDS, ETA = 220, 110, 24, 0.16
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


NF = np.load(os.path.join(FIGDIR, "sloppy_valley_shots_field.npy"))     # T2=10
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
    cache = os.path.join(FIGDIR, f"rescale_shots_scaling{SUF}_data.json")
    target = idealv(*THETA_STAR).copy()
    weights = np.array([1.0, W, W])
    print(f"ideal target {np.round(target,4).tolist()}, noisy(theta*) "
          f"{np.round(noisy(*THETA_STAR),4).tolist()}, corrected "
          f"{np.round(noisy(*THETA_STAR)-GAMMA*Dvec(*THETA_STAR),4).tolist()}")

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
                gn = gfield(NF, p[0], p[1]); gi = gfield(IF, p[0], p[1])
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

    def rms(cl):
        er = np.hypot(cl[:, 0] - THETA_STAR[0], cl[:, 1] - THETA_STAR[1])
        return float(np.sqrt(np.mean(er ** 2)))

    # ── BIAS first: deterministic (infinite-shot) fixed points ──
    # raw PSR / oracle-FD (eps->0) descend the NOISY cost -> its minimum;
    # rescaled descends the CORRECTED (~ideal) cost -> its minimum.  The
    # parameter error of each minimum vs theta* is the shot-independent BIAS
    # each method floors at.
    def cost_of(fieldval, t1, t2):
        m = fieldval(t1, t2) - target
        return m[0] ** 2 + W * (m[1] ** 2 + m[2] ** 2)

    def corr(t1, t2):
        return noisy(t1, t2) - GAMMA * Dvec(t1, t2)

    gg = np.linspace(BOX[0][0], BOX[0][1], 121)
    hh = np.linspace(BOX[1][0], BOX[1][1], 121)
    bias = {}
    for name, fv in (("noisy", noisy), ("corrected", corr)):
        best, bp = np.inf, None
        for a in gg:
            for b in hh:
                c = cost_of(fv, a, b)
                if c < best:
                    best, bp = c, (a, b)
        bias[name] = float(np.hypot(bp[0] - THETA_STAR[0], bp[1] - THETA_STAR[1]))
    print(f"BIAS (infinite-shot floors): raw PSR / oracle-FD = {bias['noisy']:.4f}"
          f"  (noisy-cost min); rescaled = {bias['corrected']:.4f} "
          f"(corrected-cost min).  ratio {bias['noisy']/max(bias['corrected'],1e-6):.1f}x")

    if os.path.exists(cache):
        d = json.load(open(cache)); print("loaded cache")
        d["bias"] = bias
    else:
        d = {"theta_star": list(THETA_STAR), "w": W, "budgets": N_BUDGETS,
             "bias": bias, "raw": {}, "fd": {}, "fd_eps": {}, "rescaled": {}}

    def save():
        json.dump(d, open(cache, "w"), default=float)

    for N_g in N_BUDGETS:
        for method in ("raw", "rescaled"):
            if str(N_g) in d[method]:
                continue
            cl = np.vstack([cloud(method, N_g, None, s)
                            for s in range(R_SEEDS)])
            d[method][str(N_g)] = rms(cl); save()
            print(f"  N={N_g} {method}: rms {d[method][str(N_g)]:.4f}", flush=True)
        if str(N_g) not in d["fd"]:
            best = None
            for eps in EPS_TUNE:
                cl = np.vstack([cloud("fd", N_g, float(eps), s)
                                for s in range(R_SEEDS)])
                rr = rms(cl)
                if best is None or rr < best[0]:
                    best = (rr, float(eps))
            d["fd"][str(N_g)] = best[0]; d["fd_eps"][str(N_g)] = best[1]; save()
            print(f"  N={N_g} oracle-fd: rms {best[0]:.4f} (eps {best[1]:g})",
                  flush=True)
    print(f"cached: {cache}")

    Ns = np.array(d["budgets"])
    fig, ax = plt.subplots(figsize=(7.8, 5.6), dpi=150)
    ax.loglog(Ns, [d["fd"][str(n)] for n in Ns], "s-", color="#7b1fa2", lw=2.2,
              label="oracle-FD (best ε per N)")
    ax.loglog(Ns, [d["raw"][str(n)] for n in Ns], "o--", color="#9e9e9e",
              lw=1.8, label="raw PSR")
    ax.loglog(Ns, [d["rescaled"][str(n)] for n in Ns], "o-", color="#00897b",
              lw=2.8, label="rescaled PSR")
    ax.axhline(d["bias"]["noisy"], color="#7b1fa2", ls=":", lw=1.4, alpha=0.8,
               label=f"raw/FD bias floor ({d['bias']['noisy']:.3f}, ∞ shots)")
    ax.axhline(d["bias"]["corrected"], color="#00897b", ls=":", lw=1.4,
               alpha=0.8,
               label=f"rescaled bias floor ({d['bias']['corrected']:.3f})")
    ax.loglog(Ns, d["rescaled"][str(Ns[0])] * (Ns / Ns[0]) ** -0.5, "-.",
              color="#555", lw=0.9, alpha=0.6, label=r"$N^{-1/2}$")
    for n in Ns:
        ax.annotate(f"ε={d['fd_eps'][str(n)]:g}", (n, d["fd"][str(n)]),
                    textcoords="offset points", xytext=(3, 6), fontsize=6,
                    color="#7b1fa2")
    ax.set_xlabel("total shots per gradient component  $N_g$")
    ax.set_ylabel(r"parameter error $\|\hat\theta - \theta^*\|$  (RMS)")
    ax.set_title(f"Closed-loop learning, sloppy landscape (w={W}, T/T2*={T/T2:.2g}, "
                 "ideal targets):\noracle-FD floors at its attenuation BIAS "
                 "(even with small ε); only rescaled PSR converges")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(True, which="both", alpha=0.12)
    fig.tight_layout()
    out = os.path.join(FIGDIR, f"rescale_shots_scaling{SUF}.png")
    fig.savefig(out); print(f"saved: {out}")


if __name__ == "__main__":
    main()
