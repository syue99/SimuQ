"""
rescale_closed_loop.py — the RESCALE in a closed learning loop, vs shots.

IQS goal: recover couplings theta*=(0.8,0.65) that reproduce the TRUE
(noiseless) physical properties.  Targets = IDEAL <O>(theta*).  The device
only measures NOISY <O>.  Cost C(theta) = (Abar-A*)^2 + w[(Bbar-B*)^2 +
(Dbar-Dd*)^2], A*=ideal <Z0Z1>(theta*), etc.

Three estimators in the SAME closed-loop gradient descent:
  FD (noisy)   : Abar = measured noisy <O>, grad = finite-diff of noisy field.
  raw PSR      : Abar = measured noisy <O>, grad = noisy gradient (lemma).
                 Both descend the NOISY cost -> fixed point where
                 noisy<O>(theta) = ideal target -> SHIFTED theta (wrong physics).
  RESCALED PSR : Abar = measured - Gamma*D_O (first-order corrected observable
                 ~ ideal <O>), grad = noisy gradient / lambda (rescaled).
                 Descends the IDEAL cost -> fixed point at theta*.

D_O = dO/dGamma from the ideal trajectory (analytic_rescale.dO_dGamma);
Gamma = 1/(2 T2).  lambda per (observable,param) = noisy_grad/ideal_grad.

"More shots" is the point: raw PSR & FD converge tightly to the noise-shifted
WRONG couplings (confidently wrong); rescaled PSR converges to theta*.  So
||theta_hat - theta*|| FLOORS for raw/FD and ->0 for rescaled as shots grow.

Reuses the noisy field cache from sloppy_valley_shots (same H/theta*/T2).

Run:  conda run -n qec_pg python differential_computing/tests/rescale_closed_loop.py
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
THETA_STAR = (0.8, 0.65)
W = 0.1
N_BUDGETS = [1000, 4000, 16000, 64000, 256000]
N_SHOW = 64000
EPS_TUNE = np.geomspace(0.1, 0.9, 9)
K_STEPS, R_SEEDS, ETA = 220, 24, 0.30
START = (1.1, 0.9)

I2 = qp.qeye(2); X = qp.sigmax(); Z = qp.sigmaz()
ZZ = qp.tensor(Z, Z); X0 = qp.tensor(X, I2); Z0O = qp.tensor(Z, I2)
XD = qp.tensor(X, I2) + qp.tensor(I2, X)
OBS = [ZZ, X0, Z0O]
PSI0 = qp.tensor(qp.basis(2, 0), qp.basis(2, 0))
C_OPS = [np.sqrt(GAMMA) * qp.tensor(Z, I2), np.sqrt(GAMMA) * qp.tensor(I2, Z)]

_GLO, _GHI, _GS = -0.1, 1.7, 0.02
_gx = np.arange(_GLO, _GHI + _GS, _GS)
FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))


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


def _build_fields():
    """noisy, ideal, and D=dO/dGamma fields on the grid (cached)."""
    fn = os.path.join(FIGDIR, "sloppy_valley_shots_field.npy")    # noisy (reuse)
    fi = os.path.join(FIGDIR, "rescale_cl_ideal.npy")
    fd = os.path.join(FIGDIR, "rescale_cl_D.npy")
    n = len(_gx)
    NF = np.load(fn) if os.path.exists(fn) else None
    if NF is None:
        NF = np.zeros((n, n, 3)); rho0 = PSI0 * PSI0.dag()
        for i, a in enumerate(_gx):
            for j, b in enumerate(_gx):
                r = qp.mesolve(Hq(a, b), rho0, [0, T], c_ops=C_OPS).states[-1]
                NF[i, j] = [float(qp.expect(o, r).real) for o in OBS]
            if i % 15 == 0:
                print(f"  noisy row {i}/{n}", flush=True)
        np.save(fn, NF)
    if os.path.exists(fi) and os.path.exists(fd):
        return NF, np.load(fi), np.load(fd)
    IF = np.zeros((n, n, 3)); DF = np.zeros((n, n, 3))
    for i, a in enumerate(_gx):
        for j, b in enumerate(_gx):
            psi = (-1j * Hq(a, b) * T).expm() * PSI0
            IF[i, j] = [float(qp.expect(o, psi).real) for o in OBS]
            for oi, o in enumerate(OBS):
                DF[i, j, oi] = ar.dO_dGamma(Hq(a, b), o, PSI0, T, 2,
                                            z_sites=range(2), n_grid=60)
        if i % 10 == 0:
            print(f"  ideal/D row {i}/{n}", flush=True)
    np.save(fi, IF); np.save(fd, DF)
    return NF, IF, DF


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    cache = os.path.join(FIGDIR, "rescale_closed_loop_data.json")

    NF, IF, DF = _build_fields()
    h = _GS

    def noisy(t1, t2): return _bilerp(NF, t1, t2)
    def ideal(t1, t2): return _bilerp(IF, t1, t2)
    def Dvec(t1, t2): return _bilerp(DF, t1, t2)
    def corrected(t1, t2): return noisy(t1, t2) - GAMMA * Dvec(t1, t2)

    def grad_field(fn_, t1, t2):     # [O][l] central diff of a field-function
        return np.stack([(fn_(t1 + h, t2) - fn_(t1 - h, t2)) / (2 * h),
                         (fn_(t1, t2 + h) - fn_(t1, t2 - h)) / (2 * h)], axis=1)

    target = ideal(*THETA_STAR).copy()
    weights = np.array([1.0, W, W])
    # sanity: corrected observable ~ ideal (first order)
    cerr = np.abs(corrected(*THETA_STAR) - target).max()
    print(f"targets(ideal)={np.round(target,4).tolist()}  "
          f"noisy(theta*)={np.round(noisy(*THETA_STAR),4).tolist()}  "
          f"corrected-vs-ideal max err {cerr:.4f}")

    def sample(fn_, t1, t2, nsh, rng):
        return 2.0 * rng.binomial(nsh, 0.5 * (1 + np.clip(fn_(t1, t2), -1, 1))) \
            / nsh - 1.0

    def descend(method, N_g, eps, seed):
        rng = np.random.default_rng(seed)
        p = np.array(START, float)
        n_res = max(1, N_g // 2)
        for _ in range(K_STEPS):
            if method == "rescaled":
                obsfn, gfld = corrected, noisy       # corrected obs; grad rescaled below
            else:
                obsfn, gfld = noisy, noisy
            # residual (measured obs; corrected shifts the measured mean)
            meas = sample(noisy, p[0], p[1], n_res, rng)
            if method == "rescaled":
                meas = meas - GAMMA * Dvec(p[0], p[1])   # correct the observable
            r = meas - target
            grad = np.zeros(2)
            gn = grad_field(noisy, p[0], p[1])           # noisy obs-gradient [O][l]
            gi = grad_field(ideal, p[0], p[1])
            for l in range(2):
                for o in range(3):
                    if method == "fd":
                        e = np.zeros(2); e[l] = eps; nfd = max(1, N_g // 2)
                        gd = (sample(noisy, p[0] + e[0], p[1] + e[1], nfd, rng)[o]
                              - sample(noisy, p[0] - e[0], p[1] - e[1], nfd, rng)[o]) \
                            / (2 * eps)
                    else:
                        # PSR obs-gradient: noisy gradient + shot noise (lemma);
                        # variance ~ (T^2/N_g) per component (validated scaling)
                        sig = T * np.sqrt(max(1e-9, 1 - gn[o, l] ** 2 * 0)) \
                            / np.sqrt(N_g)
                        gd = gn[o, l] + rng.normal(0, sig)
                        if method == "rescaled":
                            lam = gn[o, l] / gi[o, l] if abs(gi[o, l]) > 0.1 else 1.0
                            if 0.25 <= lam <= 4.0:
                                gd = gd / lam            # rescale to ideal grad
                    grad[l] += 2 * weights[o] * r[o] * gd
            p = np.clip(p - ETA * grad, 0.2, 1.4)
        return p

    if os.path.exists(cache):
        d = json.load(open(cache)); print("loaded cache — replotting only")
    else:
        d = {"theta_star": list(THETA_STAR), "start": list(START), "w": W,
             "budgets": N_BUDGETS, "n_show": N_SHOW,
             "fd": {}, "raw": {}, "rescaled": {}, "ends": {}}
        for N_g in N_BUDGETS:
            for method, key, tuned in (("fd", "fd", True),
                                       ("raw", "raw", False),
                                       ("rescaled", "rescaled", False)):
                if tuned:
                    best = None
                    for eps in EPS_TUNE:
                        ends = np.array([descend("fd", N_g, float(eps), s)
                                         for s in range(R_SEEDS)])
                        er = np.hypot(ends[:, 0] - THETA_STAR[0],
                                      ends[:, 1] - THETA_STAR[1])
                        rms = float(np.sqrt(np.mean(er ** 2)))
                        if best is None or rms < best[0]:
                            best = (rms, float(eps), ends)
                    d[key][str(N_g)] = best[0]
                    ends = best[2]
                    print(f"  N={N_g} fd: rms {best[0]:.3f} (oracle eps "
                          f"{best[1]:.2f})", flush=True)
                else:
                    ends = np.array([descend(method, N_g, None, s)
                                     for s in range(R_SEEDS)])
                    er = np.hypot(ends[:, 0] - THETA_STAR[0],
                                  ends[:, 1] - THETA_STAR[1])
                    d[key][str(N_g)] = float(np.sqrt(np.mean(er ** 2)))
                    print(f"  N={N_g} {key}: rms {d[key][str(N_g)]:.3f}",
                          flush=True)
                if N_g == N_SHOW:
                    d["ends"][key] = ends.tolist()
        json.dump(d, open(cache, "w"), default=float)
        print(f"cached: {cache}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 5.0), dpi=150)
    cols = {"fd": "#7b1fa2", "raw": "#9e9e9e", "rescaled": "#00897b"}
    labs = {"fd": "oracle-FD", "raw": "raw PSR", "rescaled": "rescaled PSR"}
    for k in ("raw", "fd", "rescaled"):
        e = np.array(d["ends"][k])
        axA.scatter(e[:, 0], e[:, 1], s=22, color=cols[k], alpha=0.6,
                    label=labs[k], zorder=2 if k == "rescaled" else 1)
        axA.plot(e[:, 0].mean(), e[:, 1].mean(), "P", color=cols[k], ms=13,
                 mec="w", mew=1.5, zorder=3)
    axA.plot(*THETA_STAR, "*", color="#ffd600", ms=20, mec="k",
             label=r"true $\theta^*$", zorder=4)
    axA.plot(*d["start"], "ks", ms=8, mfc="none", mew=2)
    axA.set_xlabel(r"$\theta_1$"); axA.set_ylabel(r"$\theta_2$")
    axA.set_title(f"(A) recovered couplings (N={d['n_show']}/grad): raw PSR & "
                  "FD\nland at the noise-SHIFTED params; rescaled PSR hits θ*")
    axA.legend(frameon=True, fontsize=8)

    Ns = np.array(d["budgets"])
    for k in ("fd", "raw", "rescaled"):
        axB.loglog(Ns, [d[k][str(n)] for n in Ns], "o-", color=cols[k],
                   lw=2.4 if k == "rescaled" else 2.0, label=labs[k])
    axB.set_xlabel("total shots per gradient component $N_g$")
    axB.set_ylabel(r"parameter error $\|\hat\theta - \theta^*\|$")
    axB.set_title("(B) more shots: raw PSR & FD FLOOR at the noise bias;\n"
                  "only rescaled PSR converges to θ*")
    axB.legend(frameon=False, fontsize=8.5)
    fig.suptitle("Closed-loop Hamiltonian learning with IDEAL targets "
                 f"(T/T2*=0.15, w={d['w']}): the rescale recovers the TRUE "
                 "couplings\nwhere raw PSR and oracle-FD converge — with more "
                 "shots — to the wrong (noise-shifted) physics", fontsize=9.8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.join(FIGDIR, "rescale_closed_loop.png")
    fig.savefig(out); print(f"saved: {out}")


if __name__ == "__main__":
    main()
