"""
sloppy_valley_shots.py — FINITE-SHOT sloppy-valley Hamiltonian learning:
does PSR reliably recover theta* while oracle-tuned FD scatters + drifts?

Capstone of the multivariable thread.  Same IQS task as sloppy_valley.py
(recover theta*=(0.8,0.65) by matching device observables
C = (<Z0Z1>-a*)^2 + w[(<X0>-b*)^2 + (<Z0>-d*)^2]) but with SHOT NOISE.

Design (fairness-critical):
  - Cost gradient dC/dθ_l = Σ_O 2 r_O ∂_l<O>.  The residuals r_O are MEASURED
    (shot-sampled) and SHARED between methods — so the only difference is how
    ∂_l<O> is estimated: PSR vs FD.  Isolates the estimator.
  - Fair shot budget: per (observable, parameter) gradient, total N_g shots.
    FD: N_g/2 per eval × 2 evals (θ±ε).  PSR: N_g split over 2·n_sample
    branches (n_per = N_g/(2 n_sample)) — the standard fair protocol.
  - ORACLE FD: per budget, ε chosen from a grid to MINIMIZE the realized RMS
    parameter error (maximally generous — an oracle that knows theta*).
  - PSR mean = exact noisy-cost gradient (lemma, validated); PSR shot noise
    is DIRECTLY simulated from a per-cell branch pool (not modeled).

Stationary scatter: start AT theta*, run K SGD steps at budget N, take the
post-burn-in cloud.  PSR cloud centers on theta* (bias-free fixed point);
oracle-FD cloud is OFFSET (smoothed-gradient fixed point) and wider (1/ε
variance) — both elongated along the sloppy valley (~1/w).

Panel A: parameter-space clouds at (w_show, N_show).
Panel B: RMS ||theta_hat - theta*|| vs total shots, PSR vs oracle-FD, split
         into bias (||mean-theta*||) and scatter (std).

Run:  conda run -n qec_pg python differential_computing/tests/sloppy_valley_shots.py
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
W_SHOW = 0.03
N_BUDGETS = [500, 2000, 8000, 32000]   # total shots per (obs,param) gradient
N_SHOW = 2000
EPS_TUNE = np.geomspace(0.1, 0.9, 9)
K_STEPS, BURN, R_SEEDS = 160, 80, 32
ETA = 0.25
POOL, NS_CAP = 48, 400
CELL = 0.05                            # pool snap resolution
BOX = ((0.45, 1.15), (0.35, 0.95))     # local grid for PSR pools + descent clip

I2 = qp.qeye(2); X = qp.sigmax(); Z = qp.sigmaz()
ZZ = qp.tensor(Z, Z); X0 = qp.tensor(X, I2); Z0O = qp.tensor(Z, I2)
XD = qp.tensor(X, I2) + qp.tensor(I2, X)
OBS = [ZZ, X0, Z0O]                     # [strong, weak, weak]
PSI0 = qp.tensor(qp.basis(2, 0), qp.basis(2, 0))
C_OPS = [np.sqrt(1.0 / (2 * T2)) * qp.tensor(Z, I2),
         np.sqrt(1.0 / (2 * T2)) * qp.tensor(I2, Z)]


def Hq(t1, t2):
    return t1 * ZZ + t2 * XD


# Exact observable field on a fine grid → bilinear interpolation everywhere
# (residuals + FD endpoints); avoids per-point mesolve during descent.
_GRID_LO, _GRID_HI, _GRID_STEP = -0.1, 1.7, 0.02
_gx = np.arange(_GRID_LO, _GRID_HI + _GRID_STEP, _GRID_STEP)
_FIELD = None


def _build_field():
    global _FIELD
    figdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                          "figures"))
    fcache = os.path.join(figdir, "sloppy_valley_shots_field.npy")
    if os.path.exists(fcache):
        _FIELD = np.load(fcache)
        return
    n = len(_gx)
    F = np.zeros((n, n, 3))
    rho0 = PSI0 * PSI0.dag()
    for i, a in enumerate(_gx):
        for j, b in enumerate(_gx):
            r = qp.mesolve(Hq(a, b), rho0, [0.0, T], c_ops=C_OPS).states[-1]
            F[i, j] = [float(qp.expect(o, r).real) for o in OBS]
        if i % 15 == 0:
            print(f"  field row {i}/{n}", flush=True)
    os.makedirs(figdir, exist_ok=True)
    np.save(fcache, F)
    _FIELD = F


def obs_exact(t1, t2):
    """Exact noisy <O> vector via bilinear interpolation of the fine grid."""
    if _FIELD is None:
        _build_field()
    fx = (np.clip(t1, _GRID_LO, _GRID_HI) - _GRID_LO) / _GRID_STEP
    fy = (np.clip(t2, _GRID_LO, _GRID_HI) - _GRID_LO) / _GRID_STEP
    i0 = int(np.floor(fx)); j0 = int(np.floor(fy))
    i0 = min(i0, len(_gx) - 2); j0 = min(j0, len(_gx) - 2)
    dx = fx - i0; dy = fy - j0
    F = _FIELD
    return ((1 - dx) * (1 - dy) * F[i0, j0] + dx * (1 - dy) * F[i0 + 1, j0]
            + (1 - dx) * dy * F[i0, j0 + 1] + dx * dy * F[i0 + 1, j0 + 1])


def obs_grad_exact(t1, t2, h=1e-3):
    """Exact noisy-landscape gradient d<O>/dθ_l (= PSR mean, lemma). [O][l]."""
    gp1 = obs_exact(t1 + h, t2); gm1 = obs_exact(t1 - h, t2)
    gp2 = obs_exact(t1, t2 + h); gm2 = obs_exact(t1, t2 - h)
    return np.stack([(gp1 - gm1) / (2 * h), (gp2 - gm2) / (2 * h)], axis=1)


# ── PSR branch pools (per cell, lazy) ─────────────────────────────────────────
_pools = {}


def _pool_expfn():
    def expfn(H_list, obs):
        rho = PSI0 * PSI0.dag()
        for k, (Hs, dur) in enumerate(H_list):
            Hqo = Hs.to_qutip_qobj()
            if k == 1:
                U = (-1j * Hqo * float(dur)).expm()
                rho = U * rho * U.dag()
            else:
                rho = qp.mesolve(Hqo, rho, [0.0, float(dur)],
                                 c_ops=C_OPS).states[-1]
        return float(qp.expect(obs, rho).real)
    return expfn


def _build_param_H(obs_idx, param_idx, t1, t2):
    """Single-symbol H for ∂θ_{param_idx}; obs is measured, not in H."""
    qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    s = sp.Symbol("v")
    zz = q[0].Z * q[1].Z; xd = q[0].X + q[1].X
    if param_idx == 0:
        H = s * zz + float(t2) * xd; val = t1
    else:
        H = float(t1) * zz + s * xd; val = t2
    return H, val


def get_pool(obs_idx, param_idx, t1, t2):
    """Cached (em, ep, ug) branch pool at snapped cell for (obs, param)."""
    c1 = round(round(t1 / CELL) * CELL, 3)
    c2 = round(round(t2 / CELL) * CELL, 3)
    key = (obs_idx, param_idx, c1, c2)
    if key not in _pools:
        H, val = _build_param_H(obs_idx, param_idx, c1, c2)
        orig = np.random.rand
        np.random.rand = lambda k: (np.arange(k) + 0.5) / k
        try:
            progs = observable_program_generator(H, T, n_sample=POOL,
                                                 n_repetition=1, diff_var="v",
                                                 value=float(val))
        finally:
            np.random.rand = orig
        expfn = _pool_expfn()
        blocks = []
        for H_tot, ug, _ in progs:
            b = len(H_tot) // 2
            em = np.array([expfn(H_tot[2 * i], OBS[obs_idx]) for i in range(b)])
            ep = np.array([expfn(H_tot[2 * i + 1], OBS[obs_idx])
                           for i in range(b)])
            blocks.append((em, ep, float(ug)))
        _pools[key] = blocks
    return _pools[key]


def psr_obs_grad(obs_idx, param_idx, t1, t2, N_g, rng):
    """One shot-noisy PSR estimate of ∂θ_{param}<O_obs> at budget N_g."""
    blocks = get_pool(obs_idx, param_idx, t1, t2)
    M = len(blocks)
    n_sample = int(min(NS_CAP, max(1, N_g // (2 * M))))
    n_per = int(max(1, round(N_g / (2 * M * n_sample))))
    g = 0.0
    for (em, ep, ug) in blocks:
        idx = rng.integers(0, len(em), size=n_sample)
        fm = 2.0 * rng.binomial(n_per, 0.5 * (1 + np.clip(em[idx], -1, 1))) \
            / n_per - 1
        fp = 2.0 * rng.binomial(n_per, 0.5 * (1 + np.clip(ep[idx], -1, 1))) \
            / n_per - 1
        g += (T / n_sample) * ug * np.sum(fm - fp)
    return g


def sample_obs(t1, t2, n, rng):
    """Shot-sampled <O> vector (n shots per observable)."""
    ex = obs_exact(t1, t2)
    return 2.0 * rng.binomial(n, 0.5 * (1 + np.clip(ex, -1, 1))) / n - 1.0


def main():
    figdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    os.makedirs(figdir, exist_ok=True)
    cache = os.path.join(figdir, "sloppy_valley_shots_data.json")

    star = obs_exact(*THETA_STAR).copy()
    weights = np.array([1.0, W_SHOW, W_SHOW])

    def clip(p):
        return np.array([np.clip(p[0], *BOX[0]), np.clip(p[1], *BOX[1])])

    def sgd_cloud(method, N_g, eps, seed):
        """Stationary cloud from theta*; returns post-burn-in points."""
        rng = np.random.default_rng(seed)
        p = np.array(THETA_STAR, float)
        pts = []
        n_res = max(1, N_g // 2)               # residual shots per observable
        for step in range(K_STEPS):
            r = sample_obs(p[0], p[1], n_res, rng) - star    # residuals
            grad = np.zeros(2)
            for l in range(2):
                for o in range(3):
                    if method == "psr":
                        gd = psr_obs_grad(o, l, p[0], p[1], N_g, rng)
                    else:                       # FD, oracle eps
                        e = np.zeros(2); e[l] = eps
                        n_fd = max(1, N_g // 2)
                        fp = sample_obs(p[0] + e[0], p[1] + e[1], n_fd, rng)[o]
                        fm = sample_obs(p[0] - e[0], p[1] - e[1], n_fd, rng)[o]
                        gd = (fp - fm) / (2 * eps)
                    grad[l] += 2 * weights[o] * r[o] * gd
            p = clip(p - ETA * grad)
            if step >= BURN:
                pts.append(p.copy())
        return np.array(pts)

    if os.path.exists(cache):
        d = json.load(open(cache))
        print("loaded cache — replotting only")
    else:
        d = {"theta_star": list(THETA_STAR), "w": W_SHOW, "budgets": N_BUDGETS,
             "n_show": N_SHOW, "psr": {}, "fd": {}, "clouds": {}}
        for N_g in N_BUDGETS:
            # PSR
            cloud = np.vstack([sgd_cloud("psr", N_g, None, s)
                               for s in range(R_SEEDS)])
            err = np.hypot(cloud[:, 0] - THETA_STAR[0],
                           cloud[:, 1] - THETA_STAR[1])
            bias = np.hypot(cloud[:, 0].mean() - THETA_STAR[0],
                            cloud[:, 1].mean() - THETA_STAR[1])
            d["psr"][str(N_g)] = dict(rms=float(np.sqrt(np.mean(err ** 2))),
                                      bias=float(bias),
                                      scatter=float(np.std(err)))
            if N_g == N_SHOW:
                d["clouds"]["psr"] = cloud.tolist()
            print(f"  PSR N={N_g}: rms {d['psr'][str(N_g)]['rms']:.3f} "
                  f"(bias {bias:.3f})", flush=True)
            # FD oracle-eps
            best = None
            for eps in EPS_TUNE:
                cl = np.vstack([sgd_cloud("fd", N_g, float(eps), s)
                                for s in range(R_SEEDS)])
                er = np.hypot(cl[:, 0] - THETA_STAR[0],
                              cl[:, 1] - THETA_STAR[1])
                rms = float(np.sqrt(np.mean(er ** 2)))
                if best is None or rms < best["rms"]:
                    bi = np.hypot(cl[:, 0].mean() - THETA_STAR[0],
                                  cl[:, 1].mean() - THETA_STAR[1])
                    best = dict(rms=rms, bias=float(bi),
                                scatter=float(np.std(er)), eps=float(eps),
                                cloud=cl)
            d["fd"][str(N_g)] = {k: best[k] for k in
                                 ("rms", "bias", "scatter", "eps")}
            if N_g == N_SHOW:
                d["clouds"]["fd"] = best["cloud"].tolist()
            print(f"  FD  N={N_g}: rms {best['rms']:.3f} (bias {best['bias']:.3f}"
                  f", oracle eps {best['eps']:.2f})", flush=True)
        json.dump(d, open(cache, "w"), default=float)
        print(f"cached: {cache}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.2, 5.0), dpi=150)
    cp = np.array(d["clouds"]["psr"]); cf = np.array(d["clouds"]["fd"])
    axA.scatter(cp[:, 0], cp[:, 1], s=8, color="#00897b", alpha=0.35,
                label="PSR", zorder=1)
    axA.scatter(cf[:, 0], cf[:, 1], s=8, color="#7b1fa2", alpha=0.5,
                label=f"oracle-FD (ε={d['fd'][str(d['n_show'])]['eps']:.2f})",
                zorder=2)
    axA.plot(cp[:, 0].mean(), cp[:, 1].mean(), "P", color="#00897b", ms=13,
             mec="w", mew=1.5, zorder=3)
    axA.plot(cf[:, 0].mean(), cf[:, 1].mean(), "P", color="#7b1fa2", ms=13,
             mec="w", mew=1.5, zorder=3)
    axA.plot(*THETA_STAR, "*", color="#ffd600", ms=20, mec="k",
             label=r"true $\theta^*$", zorder=5)
    axA.set_xlabel(r"$\theta_1$"); axA.set_ylabel(r"$\theta_2$")
    axA.set_title(f"(A) recovered-parameter cloud (w={d['w']}, N={d['n_show']}"
                  "/grad)\nboth center on θ*; oracle-FD is TIGHTER "
                  "(lower gradient variance)")
    axA.legend(frameon=True, fontsize=8)

    Ns = np.array(d["budgets"])
    axB.loglog(Ns, [d["psr"][str(n)]["rms"] for n in Ns], "o-", color="#00897b",
               lw=2.4, label="PSR — RMS error")
    axB.loglog(Ns, [d["psr"][str(n)]["bias"] for n in Ns], "o:", color="#00897b",
               lw=1.4, alpha=0.7, label="PSR — bias")
    axB.loglog(Ns, [d["fd"][str(n)]["rms"] for n in Ns], "s-", color="#7b1fa2",
               lw=2.4, label="oracle-FD — RMS error")
    axB.loglog(Ns, [d["fd"][str(n)]["bias"] for n in Ns], "s:", color="#7b1fa2",
               lw=1.4, alpha=0.7, label="oracle-FD — bias (floor)")
    axB.set_xlabel("total shots per gradient component  $N_g$")
    axB.set_ylabel(r"parameter error $\|\hat\theta-\theta^*\|$")
    axB.set_title("(B) recovery vs shots — BOTH converge ~$N^{-1/2}$, both "
                  "unbiased;\noracle-FD lower variance at this benign point")
    axB.legend(frameon=False, fontsize=8)
    fig.suptitle("Finite-shot sloppy-valley learning, MAXIMALLY-FAIR FD "
                 f"(oracle ε + chain-rule + shared residuals, T/T2*=0.15, "
                 f"w={d['w']}):\nboth recover θ* — a VARIANCE contest oracle-FD "
                 "wins.  (The ∞-shot FD failure needs black-box FD-of-cost + "
                 "floored ε.)", fontsize=9.8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.join(figdir, "sloppy_valley_shots.png")
    fig.savefig(out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
