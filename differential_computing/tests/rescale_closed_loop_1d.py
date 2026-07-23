"""
rescale_closed_loop_1d.py — the RESCALE in a robust 1-D closed learning loop.

Recover a single coupling theta*=0.8 that reproduces the TRUE (noiseless)
property A* = ideal <Z0Z1>(theta*).  Device measures NOISY <Z0Z1>.
Cost C(theta) = (Abar(theta) - A*)^2, minimized by gradient descent.

  raw PSR / FD : Abar = measured noisy <Z0Z1>; descend the noisy cost ->
                 fixed point where noisy<Z0Z1>(theta) = ideal target -> a
                 SHIFTED theta (attenuation moved it) = WRONG coupling.
  rescaled PSR : Abar = measured - Gamma*D (first-order corrected observable
                 ~ ideal <Z0Z1>), grad = noisy gradient / lambda -> theta*.

1-D is monotone near theta* -> no basin/spurious-minimum fragility; the ONLY
thing that separates the methods is the noise bias the rescale removes.
"More shots" is the point: raw PSR & FD pin CONFIDENTLY at the wrong coupling;
rescaled PSR converges to theta*.

Run:  conda run -n qec_pg python differential_computing/tests/rescale_closed_loop_1d.py
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
GAMMA = 1.0 / (2.0 * T2)
# theta*=1.15: <Z0Z1> is monotone with real slope here (scan), the dephasing
# attenuation is a clean sizeable bias, and the first-order correction recovers
# theta* — well-conditioned window (theta*=0.8 sat near the observable's max).
THETA_STAR = 1.15
DLO, DHI = 0.65, 1.45          # monotone flank; descent clipped here
N_BUDGETS = [500, 2000, 8000, 32000, 128000, 512000]
N_SHOW = 32000
EPS_TUNE = np.geomspace(0.05, 0.6, 10)
K_STEPS, R_SEEDS, ETA, START = 140, 40, 0.6, 1.42
GLO, GHI, GS = 0.2, 1.6, 0.01

I2 = qp.qeye(2); X = qp.sigmax(); Z = qp.sigmaz()
ZZ = qp.tensor(Z, Z); XD = qp.tensor(X, I2) + qp.tensor(I2, X)
PSI0 = qp.tensor(qp.basis(2, 0), qp.basis(2, 0))
C_OPS = [np.sqrt(GAMMA) * qp.tensor(Z, I2), np.sqrt(GAMMA) * qp.tensor(I2, Z)]
FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
_gx = np.arange(GLO, GHI + GS, GS)


def Hq(t):
    return t * ZZ + XD


def _interp(F, t):
    fx = (np.clip(t, GLO, GHI) - GLO) / GS
    i0 = min(int(np.floor(fx)), len(_gx) - 2)
    dx = fx - i0
    return (1 - dx) * F[i0] + dx * F[i0 + 1]


def _fields():
    cache = os.path.join(FIGDIR, "rescale_cl_1d_fields.npz")
    if os.path.exists(cache):
        z = np.load(cache); return z["noisy"], z["ideal"], z["D"]
    n = len(_gx)
    noisy = np.zeros(n); ideal = np.zeros(n); D = np.zeros(n)
    rho0 = PSI0 * PSI0.dag()
    for i, t in enumerate(_gx):
        r = qp.mesolve(Hq(t), rho0, [0, T], c_ops=C_OPS).states[-1]
        noisy[i] = float(qp.expect(ZZ, r).real)
        ideal[i] = float(qp.expect(ZZ, (-1j * Hq(t) * T).expm() * PSI0).real)
        D[i] = ar.dO_dGamma(Hq(t), ZZ, PSI0, T, 2, z_sites=range(2), n_grid=80)
    os.makedirs(FIGDIR, exist_ok=True)
    np.savez(cache, noisy=noisy, ideal=ideal, D=D)
    return noisy, ideal, D


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    cache = os.path.join(FIGDIR, "rescale_closed_loop_1d_data.json")
    NF, IF, DF = _fields()
    h = GS

    def noisy(t): return _interp(NF, t)
    def ideal(t): return _interp(IF, t)
    def Dv(t): return _interp(DF, t)
    def corrected(t): return noisy(t) - GAMMA * Dv(t)

    def dnoisy(t): return (noisy(t + h) - noisy(t - h)) / (2 * h)
    def dideal(t): return (ideal(t + h) - ideal(t - h)) / (2 * h)

    target = ideal(THETA_STAR)
    print(f"theta*={THETA_STAR}: ideal target A*={target:.4f}, "
          f"noisy(theta*)={noisy(THETA_STAR):.4f}, "
          f"corrected(theta*)={corrected(THETA_STAR):.4f} "
          f"(1st-order err {abs(corrected(THETA_STAR)-target):.4f})")
    # deterministic fixed points (infinite shots): where does each cost min sit?
    tg = np.linspace(GLO + 0.05, GHI - 0.05, 400)
    t_noisy_fp = tg[np.argmin([(noisy(t) - target) ** 2 for t in tg])]
    t_corr_fp = tg[np.argmin([(corrected(t) - target) ** 2 for t in tg])]
    print(f"infinite-shot fixed points: noisy-cost min theta={t_noisy_fp:.3f} "
          f"(err {abs(t_noisy_fp-THETA_STAR):.3f}), corrected-cost min "
          f"theta={t_corr_fp:.3f} (err {abs(t_corr_fp-THETA_STAR):.3f})")

    def sample(fn, t, nsh, rng):
        return 2.0 * rng.binomial(nsh, 0.5 * (1 + np.clip(fn(t), -1, 1))) / nsh - 1

    def descend(method, N_g, eps, seed):
        rng = np.random.default_rng(seed)
        t = START
        n_res = max(1, N_g // 2)
        for _ in range(K_STEPS):
            meas = sample(noisy, t, n_res, rng)
            if method == "rescaled":
                meas = meas - GAMMA * Dv(t)
            r = meas - target
            if method == "fd":
                nfd = max(1, N_g // 2)
                gd = (sample(noisy, t + eps, nfd, rng)
                      - sample(noisy, t - eps, nfd, rng)) / (2 * eps)
            else:
                gn = dnoisy(t)
                sig = T / np.sqrt(N_g)
                gd = gn + rng.normal(0, sig)
                if method == "rescaled":
                    lam = gn / dideal(t) if abs(dideal(t)) > 0.1 else 1.0
                    if 0.25 <= lam <= 4.0:
                        gd = gd / lam
            t = float(np.clip(t - ETA * 2 * r * gd, DLO, DHI))
        return t

    if os.path.exists(cache):
        d = json.load(open(cache)); print("loaded cache — replotting only")
    else:
        d = {"theta_star": THETA_STAR, "target": float(target),
             "budgets": N_BUDGETS, "n_show": N_SHOW,
             "noisy_fp": float(t_noisy_fp), "corr_fp": float(t_corr_fp),
             "fd": {}, "raw": {}, "rescaled": {}, "ends": {}}
        for N_g in N_BUDGETS:
            for method in ("fd", "raw", "rescaled"):
                if method == "fd":
                    best = None
                    for eps in EPS_TUNE:
                        e = np.array([descend("fd", N_g, float(eps), s)
                                      for s in range(R_SEEDS)])
                        rms = float(np.sqrt(np.mean((e - THETA_STAR) ** 2)))
                        if best is None or rms < best[0]:
                            best = (rms, float(eps), e)
                    d["fd"][str(N_g)] = best[0]; ends = best[2]
                    tag = f"(eps {best[1]:.2f})"
                else:
                    ends = np.array([descend(method, N_g, None, s)
                                     for s in range(R_SEEDS)])
                    d[method][str(N_g)] = float(np.sqrt(np.mean(
                        (ends - THETA_STAR) ** 2))); tag = ""
                if N_g == N_SHOW:
                    d["ends"][method] = ends.tolist()
                print(f"  N={N_g} {method}: rms {d[method][str(N_g)]:.4f} {tag}",
                      flush=True)
        json.dump(d, open(cache, "w"), default=float)
        print(f"cached: {cache}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=150)
    cols = {"fd": "#7b1fa2", "raw": "#9e9e9e", "rescaled": "#00897b"}
    labs = {"fd": "oracle-FD", "raw": "raw PSR", "rescaled": "rescaled PSR"}
    # panel A: recovered-coupling histograms at N_show
    for k in ("raw", "fd", "rescaled"):
        e = np.array(d["ends"][k])
        axA.hist(e, bins=22, color=cols[k], alpha=0.5, label=labs[k])
    axA.axvline(THETA_STAR, color="#ffd600", lw=2.5, label=r"true $\theta^*$")
    axA.axvline(d["noisy_fp"], color="#7b1fa2", ls="--", lw=1.5,
                label="noisy-cost min (raw/FD target)")
    axA.set_xlabel(r"recovered coupling $\hat\theta$")
    axA.set_ylabel("count")
    axA.set_title(f"(A) recovered coupling (N={d['n_show']}): raw PSR & FD pin "
                  "at\nthe noise-shifted value; rescaled PSR at θ*")
    axA.legend(frameon=False, fontsize=7.8)

    Ns = np.array(d["budgets"])
    for k in ("fd", "raw", "rescaled"):
        axB.loglog(Ns, [max(d[k][str(n)], 1e-4) for n in Ns], "o-",
                   color=cols[k], lw=2.4 if k == "rescaled" else 2.0,
                   label=labs[k])
    axB.axhline(abs(d["noisy_fp"] - THETA_STAR), color="#7b1fa2", ls=":",
                lw=1.2, alpha=0.7, label="noise bias floor")
    axB.set_xlabel("total shots per gradient  $N_g$")
    axB.set_ylabel(r"coupling error $|\hat\theta - \theta^*|$")
    axB.set_title("(B) more shots: raw PSR & FD FLOOR at the noise bias;\n"
                  "only rescaled PSR converges to θ*")
    axB.legend(frameon=False, fontsize=8)
    fig.suptitle("Closed-loop Hamiltonian learning, ideal target (1-D, robust; "
                 f"T/T2*=0.15): rescaled PSR recovers the TRUE coupling θ*={d['theta_star']}"
                 "\nwhere raw PSR & oracle-FD converge — with more shots — to "
                 "the noise-shifted wrong value", fontsize=9.8)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = os.path.join(FIGDIR, "rescale_closed_loop_1d.png")
    fig.savefig(out); print(f"saved: {out}")


if __name__ == "__main__":
    main()
