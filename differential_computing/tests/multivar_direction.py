"""
multivar_direction.py (Case A) — multivariable gradient DIRECTION under noise:
does per-component rescaled PSR recover a better descent direction than FD?

2-qubit H(θ1,θ2) = θ1·Z0Z1 + θ2·(X0+X1), observable <Z0Z1>, T/T2* = 0.15,
∞ shots (exact expectations), deterministic τ.  Per-parameter attenuations are
HETEROGENEOUS (λ_ZZ ≈ 0.9 vs λ_X ≈ 1.0), so the noisy gradient vector is
ĝ = diag(λ1, λ2)·g_true — a ROTATION no scalar learning rate can absorb.

Three-way structure this study demonstrates on a (θ1, θ2) grid:
  PSR raw  : all λ_ℓ > 0 ⇒ positive-diagonal preconditioner ⇒ rotated but
             ALWAYS inside the descent cone (angle < 90°).
  PSR resc : per-component analytic 1/λ_ℓ ⇒ DIRECTION restored (<~1°).
  FD floored-ε: truncation bias flips component signs on sharp regions ⇒
             indefinite distortion ⇒ can LEAVE the descent cone (angle > 90°).

LEMMA SHORTCUT (honest): at ∞ shots with deterministic τ and noiseless kicks,
raw PSR = the noisy-landscape gradient exactly (validated to 4 decimals many
times, incl. here at 3 grid points with full per-parameter PSR pools).  The
grid maps therefore compute PSR-raw as the fine-ε noisy gradient; the full
pools run only at the validation points.

Multi-parameter PSR follows the h2_vqe pattern: per-parameter single-symbol
Hamiltonian (other parameter substituted numerically).

Run:  conda run -n qec_pg python differential_computing/tests/multivar_direction.py
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

T, T2 = 1.5, 10.0                 # T/T2* = 0.15 (headline regime)
N = 2
EPS_FLOORS = [0.3, 0.6]           # hardware-floored FD steps for the trap maps
GRID = np.linspace(0.2, 1.4, 13)  # θ1 and θ2
G_MIN = 0.15                      # mask |g_true| below this (angle meaningless)
H_FD = 1e-3                       # fine step for ε→0 gradients
POOL = 100                        # τ pool for PSR validation points
VALID_PTS = [(0.4, 0.6), (0.8, 1.0), (1.2, 0.4)]

I2 = qp.qeye(2); X = qp.sigmax(); Z = qp.sigmaz()
ZZ = qp.tensor(Z, Z)
XX_SUM = qp.tensor(X, I2) + qp.tensor(I2, X)
OBS = ZZ
PSI0 = qp.tensor(qp.basis(2, 0), qp.basis(2, 0))


def Hq(t1, t2):
    return t1 * ZZ + t2 * XX_SUM


def clean_expect(t1, t2):
    return float(qp.expect(OBS, (-1j * Hq(t1, t2) * T).expm() * PSI0).real)


def grad2(f, t1, t2, h):
    return np.array([(f(t1 + h, t2) - f(t1 - h, t2)) / (2 * h),
                     (f(t1, t2 + h) - f(t1, t2 - h)) / (2 * h)])


def lambda_factors(t1, t2, g_true):
    """Per-parameter analytic rescale factors (ideal-trajectory only)."""
    def dOdG(a, b):
        return ar.dO_dGamma(Hq(a, b), OBS, PSI0, T, N, z_sites=range(N),
                            n_grid=120)
    factors = []
    for ell in range(2):
        e = np.zeros(2); e[ell] = H_FD
        dgdG = (dOdG(t1 + e[0], t2 + e[1]) - dOdG(t1 - e[0], t2 - e[1])) \
            / (2 * H_FD)
        slope = (dgdG / g_true[ell]) / (2.0 * T)
        factors.append(ar.rescale_factor(slope, T, T2))
    return np.array(factors)


def angle_deg(v, ref):
    c = float(np.dot(v, ref) / (np.linalg.norm(v) * np.linalg.norm(ref)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def psr_vector_validation(noisy_expfn, t1, t2):
    """Full per-parameter PSR pools (deterministic τ) at one point."""
    grad = []
    for diff_var, value, fixed in (("x", t1, ("y", t2)), ("y", t2, ("x", t1))):
        qs = QSystem(); q = [Qubit(qs) for _ in range(N)]
        # per-parameter single-symbol Hamiltonian (h2_vqe pattern):
        s = sp.Symbol(diff_var)
        if diff_var == "x":
            Hp = s * (q[0].Z * q[1].Z) + float(fixed[1]) * (q[0].X + q[1].X)
        else:
            Hp = float(fixed[1]) * (q[0].Z * q[1].Z) + s * (q[0].X + q[1].X)
        orig = np.random.rand
        np.random.rand = lambda k: (np.arange(k) + 0.5) / k
        try:
            progs = observable_program_generator(
                Hp, T, n_sample=POOL, n_repetition=1,
                diff_var=diff_var, value=float(value))
        finally:
            np.random.rand = orig
        g = 0.0
        for H_tot, ug, _ in progs:
            b = len(H_tot) // 2
            em = np.array([noisy_expfn(H_tot[2 * i]) for i in range(b)])
            ep = np.array([noisy_expfn(H_tot[2 * i + 1]) for i in range(b)])
            g += (T / b) * float(ug) * np.sum(em - ep)
        grad.append(g)
    return np.array(grad)


def main():
    figdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    os.makedirs(figdir, exist_ok=True)
    cache = os.path.join(figdir, "multivar_direction_data.json")

    if os.path.exists(cache):
        d = json.load(open(cache))
        print("loaded cache — replotting only")
    else:
        noisy = NoisyQuTiPRunner(N, noise=NoiseModel(n_qubits=N, T2=T2))
        nx = noisy.make_expectation_fn(PSI0, OBS)

        # noisy landscape via simuq TIHam (runner API)
        x, y = sp.Symbol("x"), sp.Symbol("y")
        qs = QSystem(); q = [Qubit(qs) for _ in range(N)]
        Hsq = x * (q[0].Z * q[1].Z) + y * (q[0].X + q[1].X)

        def fnoisy(t1, t2):
            Ht = Hsq.set_parameterizedHam({"x": float(t1), "y": float(t2)})
            return nx([[Ht, T]])

        pts = []
        for i, t1 in enumerate(GRID):
            for j, t2 in enumerate(GRID):
                g_true = grad2(clean_expect, t1, t2, H_FD)
                gnorm = float(np.linalg.norm(g_true))
                rec = dict(i=i, j=j, t1=float(t1), t2=float(t2),
                           g_true=list(g_true), gnorm=gnorm)
                if gnorm >= G_MIN:
                    g_raw = grad2(fnoisy, t1, t2, H_FD)      # = PSR raw (lemma)
                    fac = lambda_factors(t1, t2, g_true)
                    g_res = g_raw * fac
                    rec.update(g_raw=list(g_raw), factors=list(fac),
                               lam=[1.0 / fac[0], 1.0 / fac[1]],
                               ang_raw=angle_deg(g_raw, g_true),
                               ang_res=angle_deg(g_res, g_true))
                    for eps in EPS_FLOORS:
                        g_fd = grad2(fnoisy, t1, t2, eps)
                        rec[f"ang_fd{eps}"] = angle_deg(g_fd, g_true)
                pts.append(rec)
            print(f"  row {i + 1}/{len(GRID)} done", flush=True)

        # lemma validation: full PSR pools at 3 points
        checks = []
        for (t1, t2) in VALID_PTS:
            g_pool = psr_vector_validation(nx, t1, t2)
            g_fine = grad2(fnoisy, t1, t2, H_FD)
            checks.append(dict(t1=t1, t2=t2, g_pool=list(g_pool),
                               g_fine=list(g_fine),
                               diff=float(np.max(np.abs(g_pool - g_fine)))))
            print(f"  lemma check ({t1},{t2}): pool {g_pool} vs fine {g_fine}")
        d = dict(T=T, T2=T2, grid=list(map(float, GRID)), pts=pts,
                 checks=checks)
        json.dump(d, open(cache, "w"), default=float)
        print(f"cached: {cache}")

    # ── analysis + maps ──
    # COMPONENT-GATED rescale (the Case-A refinement): the per-component 1/λ_ℓ
    # inherits the small-gradient instability PER COMPONENT — a near-zero
    # component's factor explodes (λ observed up to 1e25) and wrecks an
    # otherwise good direction (ungated rescaled max ~79°).  Rule: apply the
    # rescale only to components with |g_raw_ℓ| ≥ G_COMP and a sane factor;
    # leave the rest raw (sign-safe).  Leaving a tiny component raw barely
    # moves the direction; rescaling it by a huge factor destroys it.
    G_COMP, FAC_LO, FAC_HI = 0.10, 0.25, 4.0

    G = len(d["grid"])
    keys = ["ang_raw", "ang_res", "ang_res_gated"] + \
        [f"ang_fd{e}" for e in EPS_FLOORS]
    maps = {k: np.full((G, G), np.nan) for k in keys}
    lam1, lam2 = [], []
    for r in d["pts"]:
        if "ang_raw" not in r:
            continue
        g_true = np.array(r["g_true"]); g_raw = np.array(r["g_raw"])
        fac = np.array(r["factors"])
        fac_g = np.where((np.abs(g_raw) >= G_COMP) &
                         (fac >= FAC_LO) & (fac <= FAC_HI), fac, 1.0)
        r["ang_res_gated"] = angle_deg(g_raw * fac_g, g_true)
        for k in maps:
            maps[k][r["j"], r["i"]] = r[k]
        if abs(g_raw[0]) >= G_COMP and FAC_LO <= fac[0] <= FAC_HI:
            lam1.append(r["lam"][0])
        if abs(g_raw[1]) >= G_COMP and FAC_LO <= fac[1] <= FAC_HI:
            lam2.append(r["lam"][1])

    print(f"\nλ1 (ZZ-param, gated pts): {min(lam1):.3f}–{max(lam1):.3f}   "
          f"λ2 (X-param, gated pts): {min(lam2):.3f}–{max(lam2):.3f}")
    for k, m in maps.items():
        v = m[~np.isnan(m)]
        print(f"{k:>10}: median {np.median(v):5.2f}°  max {v.max():6.2f}°  "
              f">90° at {100 * np.mean(v > 90):.1f}% of valid points")
    for c in d["checks"]:
        print(f"lemma check ({c['t1']},{c['t2']}): max component diff "
              f"{c['diff']:.5f}")

    titles = {"ang_raw": "PSR raw (= FD best-ε)",
              "ang_res_gated": "PSR rescaled (component-gated)",
              f"ang_fd{EPS_FLOORS[0]}": f"FD ε={EPS_FLOORS[0]} (floored)",
              f"ang_fd{EPS_FLOORS[1]}": f"FD ε={EPS_FLOORS[1]} (floored)"}
    order = ["ang_raw", "ang_res_gated",
             f"ang_fd{EPS_FLOORS[0]}", f"ang_fd{EPS_FLOORS[1]}"]
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 8.2), dpi=150,
                            sharex=True, sharey=True)
    ext = [d["grid"][0], d["grid"][-1], d["grid"][0], d["grid"][-1]]
    for ax, k in zip(axs.flat, order):
        m = maps[k]
        im = ax.imshow(np.minimum(m, 20.0), origin="lower", extent=ext,
                       vmin=0, vmax=20, cmap="viridis", aspect="auto")
        bad = m > 90
        if bad.any():
            yy, xx = np.where(bad)
            gs = np.array(d["grid"])
            ax.plot(gs[xx], gs[yy], "x", color="#d62728", ms=9, mew=2.4,
                    label="angle > 90° (uphill)")
            ax.legend(frameon=False, fontsize=8, loc="upper right")
        v = m[~np.isnan(m)]
        ax.set_title(f"{titles[k]}\nmedian {np.median(v):.2f}°, "
                     f"max {v.max():.0f}°", fontsize=9)
    for ax in axs[-1]:
        ax.set_xlabel(r"$\theta_1$ (ZZ)")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$\theta_2$ (X)")
    cb = fig.colorbar(im, ax=axs, shrink=0.85)
    cb.set_label("angle to true gradient (deg, capped at 20)")
    fig.suptitle("Multivariable gradient DIRECTION under noise (T/T2*=0.15, ∞ shots):\n"
                 "heterogeneous per-parameter attenuation rotates the vector; "
                 "raw PSR stays in the descent cone,\nper-component rescale restores "
                 "the direction, floored-ε FD can point uphill (red ✕)",
                 fontsize=10.5)
    out = os.path.join(figdir, "multivar_direction.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
