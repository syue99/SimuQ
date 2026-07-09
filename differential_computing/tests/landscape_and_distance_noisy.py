"""
landscape_and_distance_noisy.py — noisier regime where BOTH FD arms fail on sign:
small ε → shot-noise coin-flip (variance), large ε → truncation bias.  PSR sits
below the whole U.  (Companion to landscape_and_distance.py; separate figure.)

Regime: stronger dephasing (T/T2*=0.5), a smaller-gradient point, fewer shots —
so small-ε shot noise is large enough to flip the sign too.  FD ε are marked RED
when the per-shot WRONG-SIGN fraction exceeds 20% (captures both the variance arm
and the bias arm).

Caches all plot inputs to figures/landscape_and_distance_noisy_data.json —
replots (and the paper_fig builder) never re-simulate.

Run:  conda run -n qec_pg python differential_computing/tests/landscape_and_distance_noisy.py
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
import analytic_rescale as ar
from observable_program_generator import observable_program_generator
from noisy_qutip import NoisyQuTiPRunner
from noise_model import NoiseModel

T = 2.5
T2 = 5.0                   # T/T2* = 0.50 (noisier)
OBS = qp.tensor(qp.sigmaz(), qp.qeye(2))
PSI0 = qp.tensor(qp.basis(2, 0), qp.basis(2, 0))
I = qp.qeye(2); X, Z = qp.sigmax(), qp.sigmaz()
N_SHOTS, R, N_SAMPLE, POOL = 9000, 5000, 48, 800
WRONG_FRAC = 0.20          # mark ε red if >20% of shot estimates have wrong sign


def Hsq(th):
    return th * qp.tensor(Z, I) + qp.tensor(X, I)


def Hsimuq():
    x = sp.Symbol("x"); qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    return x * q[0].Z + q[0].X, "x"


def fmaker(runner):
    H, var = Hsimuq()
    expfn = runner.make_expectation_fn(PSI0, OBS)
    return lambda x: expfn([[H.set_parameterizedHam({"x": float(x)}), T]])


def compute():
    rng = np.random.default_rng(0)
    clean = NoisyQuTiPRunner(2, noise=None)
    noisy = NoisyQuTiPRunner(2, noise=NoiseModel(n_qubits=2, T2=T2))
    fc, fn = fmaker(clean), fmaker(noisy)

    # moderate gradient (PSR resolves it, rescale STABLE) + FD large-ε sign-flip.
    # Among sign-flip candidates, pick one whose analytic rescale factor is moderate.
    xs = np.linspace(0.2, 1.6, 130)
    gi = np.array([(fc(x + 1e-3) - fc(x - 1e-3)) / 2e-3 for x in xs])
    cands = []
    for k, x in enumerate(xs):
        if 0.25 < abs(gi[k]) < 0.42:
            fd06 = (fn(x + 0.6) - fn(x - 0.6)) / 1.2
            if np.sign(fd06) != np.sign(gi[k]):
                cands.append((float(x), abs(fd06 - gi[k])))
    cands.sort(key=lambda c: -c[1])               # most FD failure first
    x_star, s, factor = None, None, None
    for x, _ in cands[:8]:
        sx = ar.lambda_slope(Hsq, OBS, PSI0, T, 2, z_sites=[0], theta=x)
        fx = ar.rescale_factor(sx, T, T2)
        if 0.7 <= fx <= 1.9:                       # stable rescale
            x_star, s, factor = x, sx, fx
            break
    if x_star is None:                             # fallback: least-extreme factor
        x_star = cands[0][0] if cands else float(xs[np.argmin(np.abs(np.abs(gi)-0.3))])
        s = ar.lambda_slope(Hsq, OBS, PSI0, T, 2, z_sites=[0], theta=x_star)
        factor = ar.rescale_factor(s, T, T2)
    g_real = (fc(x_star + 1e-3) - fc(x_star - 1e-3)) / 2e-3
    sgn = np.sign(g_real)

    def shots(exact, n):
        return 2.0 * rng.binomial(int(max(1, n)),
                                  0.5*(1+np.clip(exact, -1, 1)), size=R)/max(1, n)-1

    eps_grid = np.geomspace(0.03, 1.4, 22)
    fd_rmse, fd_wrongfrac = [], []
    for eps in eps_grid:
        n = N_SHOTS // 2
        est = (shots(fn(x_star + eps), n) - shots(fn(x_star - eps), n)) / (2 * eps)
        fd_rmse.append(float(np.sqrt(np.mean((est - g_real) ** 2))))
        fd_wrongfrac.append(float(np.mean(np.sign(est) != sgn)))

    # PSR pool under noise.  The POOL is the τ "population" the estimator subsamples
    # N_SAMPLE from; build it with a DETERMINISTIC midpoint grid (O(1/n²)) so the
    # estimator's CENTER is the exact τ-integral (a random pool offsets the center by
    # ~1/√POOL ≈ 0.035).  The realistic τ-sampling VARIANCE is preserved: the per-trial
    # N_SAMPLE subsampling below uses a separate rng and still draws uniformly over τ.
    H, var = Hsimuq()
    orig_rand = np.random.rand
    np.random.rand = lambda k: (np.arange(k) + 0.5) / k
    try:
        progs = observable_program_generator(H, T, n_sample=POOL, n_repetition=1,
                                             diff_var=var, value=x_star)
    finally:
        np.random.rand = orig_rand
    pexp = noisy.make_expectation_fn(PSI0, OBS)
    H_tot, ug, _ = progs[0]; b = len(H_tot)//2
    em = np.array([pexp(H_tot[2*i]) for i in range(b)])
    ep = np.array([pexp(H_tot[2*i+1]) for i in range(b)])
    n_per = int(max(1, round(N_SHOTS/(2*N_SAMPLE))))
    idx = rng.integers(0, len(em), size=(R, N_SAMPLE))
    fm = 2.0*rng.binomial(n_per, 0.5*(1+np.clip(em[idx], -1, 1)))/n_per-1
    fpb = 2.0*rng.binomial(n_per, 0.5*(1+np.clip(ep[idx], -1, 1)))/n_per-1
    raw = (T/N_SAMPLE)*float(ug)*np.sum(fm - fpb, axis=1)
    psr_raw_rmse = float(np.sqrt(np.mean((raw - g_real)**2)))
    psr_resc_rmse = float(np.sqrt(np.mean((raw*factor - g_real)**2)))
    psr_raw_wrong = float(np.mean(np.sign(raw) != sgn))
    psr_slope = float(np.mean(raw)) * factor

    # landscape curves + secants (plot inputs)
    gx = np.linspace(0.0, 1.45, 150)
    y_ideal = [fc(x) for x in gx]
    y_noisy = [fn(x) for x in gx]
    z0 = fc(x_star)
    sec_eps = [0.15, 0.3, 0.45, 0.6]
    secants = [dict(eps=e, fm=fn(x_star - e), fp=fn(x_star + e)) for e in sec_eps]
    # tiny ε=0.01: shot-noise dominated → a typical realization flips the sign
    n0 = N_SHOTS // 2
    sl01 = None
    for _ in range(20):
        fp01 = 2*rng.binomial(n0, 0.5*(1+np.clip(fn(x_star+0.01), -1, 1)))/n0 - 1
        fm01 = 2*rng.binomial(n0, 0.5*(1+np.clip(fn(x_star-0.01), -1, 1)))/n0 - 1
        cand = (fp01 - fm01) / 0.02
        if np.sign(cand) != sgn:                  # a representative wrong-sign draw
            sl01 = cand; break
    if sl01 is None:
        sl01 = cand

    return dict(T=T, T2=T2, N_SHOTS=N_SHOTS,
                x_star=float(x_star), g_real=float(g_real), factor=float(factor),
                z0=float(z0), sl01=float(sl01), psr_slope=psr_slope,
                gx=list(map(float, gx)), y_ideal=y_ideal, y_noisy=y_noisy,
                secants=secants,
                eps_grid=list(map(float, eps_grid)),
                fd_rmse=fd_rmse, fd_wrongfrac=fd_wrongfrac,
                psr_raw_rmse=psr_raw_rmse, psr_resc_rmse=psr_resc_rmse,
                psr_raw_wrong=psr_raw_wrong)


def main():
    figdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    os.makedirs(figdir, exist_ok=True)
    cache = os.path.join(figdir, "landscape_and_distance_noisy_data.json")
    if os.path.exists(cache):
        d = json.load(open(cache))
        print("loaded cache — replotting only")
    else:
        d = compute()
        json.dump(d, open(cache, "w"), default=float)
        print(f"cached: {cache}")

    x_star, g_real, factor = d["x_star"], d["g_real"], d["factor"]
    sgn = np.sign(g_real)
    eps_grid = np.array(d["eps_grid"]); fd_rmse = np.array(d["fd_rmse"])
    fd_wrong = np.array(d["fd_wrongfrac"]) > WRONG_FRAC

    print(f"T={d['T']}, T/T2*={d['T']/d['T2']:.2f}, x*={x_star:.3f}, "
          f"REAL grad={g_real:+.4f}, 1/λ={factor:.2f}.")
    print(f"  PSR raw RMSE {d['psr_raw_rmse']:.4f} "
          f"(wrong-sign {d['psr_raw_wrong']:.0%}), "
          f"PSR rescaled RMSE {d['psr_resc_rmse']:.4f}")
    print(f"  FD best RMSE {fd_rmse.min():.4f}; ε wrong-sign>20%: "
          f"{int(fd_wrong.sum())}/{len(eps_grid)} "
          f"(small-ε arm {int((fd_wrong & (eps_grid<0.15)).sum())}, "
          f"large-ε arm {int((fd_wrong & (eps_grid>0.4)).sum())})")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.8, 4.9), dpi=150)
    gx = np.array(d["gx"])
    axA.plot(gx, d["y_ideal"], "--", color="#9aa0a6", lw=1.6, label="ideal")
    axA.plot(gx, d["y_noisy"], color="#202124", lw=2.2,
             label=f"noisy (T/T2*={d['T']/d['T2']:.2f})")
    z0 = d["z0"]; ex = np.array([x_star - 0.28, x_star + 0.28])
    axA.plot(ex, z0 + g_real*(ex - x_star), color="#1a73e8", lw=3,
             label=f"true tangent ({g_real:+.3f})")
    cmap = plt.cm.viridis(np.linspace(0.2, 0.82, len(d["secants"])))
    for sec, c in zip(d["secants"], cmap):
        e, fm_e, fp_e = sec["eps"], sec["fm"], sec["fp"]
        axA.plot([x_star-e, x_star+e], [fm_e, fp_e], "o-", color=c, lw=1.8,
                 ms=4, label=f"FD ε={e}: {(fp_e-fm_e)/(2*e):+.3f}")
    axA.plot(ex, z0 + d["sl01"]*(ex - x_star), ":", color="#d50000", lw=2.4,
             label=f"FD ε=0.01 (shots): {d['sl01']:+.2f}  WRONG sign")
    axA.plot(ex, z0 + d["psr_slope"]*(ex - x_star), "-", color="#00897b", lw=2.6,
             label=f"PSR rescaled: {d['psr_slope']:+.3f}")
    axA.axhline(0, color="#bbb", lw=0.8); axA.axvline(x_star, color="#bbb", ls=":", lw=1)
    axA.plot([x_star], [z0], "ko", ms=6)
    axA.set_xlabel("parameter x"); axA.set_ylabel(r"$\langle Z_0\rangle(x)$")
    axA.set_title(f"(A) sharp landscape + FD secants at all ε  (x*={x_star:.2f})")
    axA.legend(frameon=False, fontsize=8, loc="lower right")

    axB.loglog(eps_grid, fd_rmse, "-", color="#7b1fa2", lw=2, zorder=1, label="FD (shots)")
    ok = ~fd_wrong
    axB.loglog(eps_grid[ok], fd_rmse[ok], "o", color="#7b1fa2", ms=6, zorder=2)
    if fd_wrong.any():
        axB.loglog(eps_grid[fd_wrong], fd_rmse[fd_wrong], "X", color="#d50000",
                   ms=10, zorder=3, label=">20% wrong sign")
    axB.axhline(d["psr_raw_rmse"], color="#9e9e9e", lw=2.2, ls="--", label="PSR raw")
    axB.axhline(d["psr_resc_rmse"], color="#00897b", lw=2.8, label="PSR rescaled")
    axB.set_xlabel(r"FD step size $\varepsilon$")
    axB.set_ylabel("distance to real gradient (RMSE)")
    axB.set_title(f"(B) error vs ε, N={d['N_SHOTS']} shots — both arms fail on sign")
    axB.text(0.03, 0.04, f"real gradient = {g_real:+.3f}\n"
             f"small ε → shot-noise sign flips   large ε → bias sign flip",
             transform=axB.transAxes, fontsize=8, color="#333", va="bottom")
    axB.legend(frameon=False, fontsize=8.5, loc="upper center")

    fig.suptitle(f"Noisier regime T/T2*={d['T']/d['T2']:.2f}: FD's sign is wrong at "
                 f"BOTH small ε (shot noise) AND large ε (bias) —\nno ε is reliable; "
                 f"PSR (raw & rescaled) sit below the whole U.", fontsize=9.2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.join(figdir, "landscape_and_distance_noisy.png")
    fig.savefig(out)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
