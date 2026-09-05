"""
build_Floop_direction.py — F-loop (Fig A-c), gradient-DIRECTION map.

The descent-trap is Goldilocks-hard (mild landscape self-corrects; strongly multi-modal one is
non-identifiable for everyone). The robust demonstration that FD cannot reliably reach the
optimum is the DIRECTION map: at each point, the angle of the estimator's gradient to the true
noisy-landscape gradient ∇C_noisy (the realized objective's gradient — descending it closes the
loop). A gradient more than 90° from ∇C_noisy points UPHILL: the loop would ascend/diverge there
— a trap zone. FD at a floored ε points uphill over whole regions; the ε-free strategies stay in
the descent cone (<90°) everywhere.

Sec-6-clean (vs the ML transfer-map paper): reference is ∇C_noisy (NOT ∇C_ideal), estimators are
finite-shot PSR and NSR (M=∞) — NO rescale (the sound strategies win by being ε-free). 2q TFIM
H=θ1·Z0Z1+θ2·(X0+X1), observable ⟨Z0Z1⟩, T/T2*=0.15, compiled to machine-native segments,
emulated under T4 (diagonal-readout shots).
Panels: PSR | NSR (M=∞) | FD ε=0.3 (floored) | FD ε=0.6 (floored). Uphill (>90°) marked red ×.
Run: conda run -n qec_pg python differential_computing/tests/build_Floop_direction.py
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
from noisy_qutip import NoisyQuTiPRunner
from noise_model import NoiseModel
from observable_program_generator import observable_program_generator
from nyquist_shift import tangent_hamiltonian, bandwidth_K

T, T2 = 1.5, 10.0                    # T/T2* = 0.15
DELTA = 0.02
B_BUDGET = int(os.environ.get("B_BUDGET", "8000"))    # shots/gradient (cheap: scales binomial n)
M_PSR = 10
MAXN = 14
GRID = np.linspace(0.2, 1.4, int(os.environ.get("NG", "13")))
G_MIN = 0.12                         # mask |∇C_noisy| below this (direction meaningless when flat)
H_FINE = 1e-3
EPS_FLOORS = [0.3, 0.6]
FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))

I2, X, Z = qp.qeye(2), qp.sigmax(), qp.sigmaz()
ZZ = qp.tensor(Z, Z)
PSI0 = qp.tensor(qp.basis(2, 0), qp.basis(2, 0))
ZZV = np.array([1.0, -1.0, -1.0, 1.0])
RUNNER = NoisyQuTiPRunner(2, noise=NoiseModel(n_qubits=2, T2=T2))
PROBS = RUNNER.make_probs_fn(PSI0)
H_SYM, NAMES = None, ("x", "y")


def build_sym():
    x, y = sp.Symbol("x"), sp.Symbol("y")
    qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    return x * (q[0].Z * q[1].Z) + y * (q[0].X + q[1].X), ("x", "y")


H_SYM, NAMES = build_sym()


def probs_at(th, no_clip=False):
    return PROBS([[H_SYM.set_parameterizedHam({"x": float(th[0]), "y": float(th[1])}), T]])


def Oexact(th):
    return float(ZZV @ probs_at(th))


def gnoisy(th, h=H_FINE):                       # ∇C_noisy (fine central FD of the noisy landscape)
    return np.array([(Oexact([th[0] + h, th[1]]) - Oexact([th[0] - h, th[1]])) / (2 * h),
                     (Oexact([th[0], th[1] + h]) - Oexact([th[0], th[1] - h])) / (2 * h)])


def shot_O(p, n, rng):
    idx = rng.choice(4, size=int(max(1, n)), p=p)
    return float(np.mean(ZZV[idx]))


def Hp_for(ell, th):
    return H_SYM.set_parameterizedHam({"y": float(th[1])}) if ell == 0 \
        else H_SYM.set_parameterizedHam({"x": float(th[0])})


def psr_g(th, rng):
    g = np.zeros(2)
    for ell in range(2):
        Hp = Hp_for(ell, th)
        orig = np.random.rand; np.random.rand = lambda k: (np.arange(k) + 0.5) / k
        try:
            pr = observable_program_generator(Hp, T, n_sample=M_PSR, n_repetition=1,
                                              diff_var=NAMES[ell], value=float(th[ell]))
        finally:
            np.random.rand = orig
        # one program set per Pauli term of the coefficient (y multiplies X0 + X1 → two);
        # weight T/M per τ-sample per term.  Reading pr[0] only halved ∂/∂y (fixed 2026-09-04).
        nb_tot = sum(len(H_tot) // 2 for H_tot, _, _ in pr)
        nper = int(max(1, round((B_BUDGET / 2) / (2 * nb_tot))))
        for H_tot, ug, _ in pr:
            nb = len(H_tot) // 2
            fm = np.array([shot_O(PROBS(H_tot[2 * i]), nper, rng) for i in range(nb)])
            fp = np.array([shot_O(PROBS(H_tot[2 * i + 1]), nper, rng) for i in range(nb)])
            g[ell] += (T / M_PSR) * float(ug) * float(np.sum(fm - fp))
    return g


ns = np.arange(MAXN); pw = 1.0 / (ns + 0.5) ** 2; pw /= pw.sum()


def nsr_g(th, rng):
    g = np.zeros(2); nshot = int(max(1, round(B_BUDGET / 2)))
    for ell in range(2):
        Hp = Hp_for(ell, th)
        _, A = tangent_hamiltonian(Hp, NAMES[ell], float(th[ell]))
        K = bandwidth_K(A, T); L1 = 2 * np.pi * K
        cache = {}
        for sg in (-1.0, 1.0):
            for n in range(MAXN):
                s = sg * (n + 0.5) / (2 * K)
                tt = th.copy(); tt[ell] += s                 # M=∞: probe not clamped
                cache[(n, sg)] = probs_at(tt)
        nd = rng.choice(ns, size=nshot, p=pw); sig = rng.choice([-1.0, 1.0], size=nshot)
        vals = np.array([shot_O(cache[(int(a), b)], 1, rng) for a, b in zip(nd, sig)])
        g[ell] = float(np.mean(L1 * ((-1.0) ** nd) * sig * vals))
    return g


def fd_g(th, eps, rng):
    g = np.zeros(2); nper = max(1, B_BUDGET // 4)
    for ell in range(2):
        dp, dm = rng.normal(0, DELTA), rng.normal(0, DELTA)
        ep = th.copy(); ep[ell] += eps + dp
        em = th.copy(); em[ell] -= eps - dm
        g[ell] = (shot_O(probs_at(ep), nper, rng) - shot_O(probs_at(em), nper, rng)) / (2 * eps)
    return g


def angle_deg(v, ref):
    nv, nr = np.linalg.norm(v), np.linalg.norm(ref)
    if nv < 1e-12 or nr < 1e-12:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(v, ref) / (nv * nr), -1, 1))))


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    cache = os.path.join(FIGDIR, "F_loop_direction_data.json")
    keys = ["PSR", "NSR"] + [f"FD{e}" for e in EPS_FLOORS]
    if os.path.exists(cache) and os.environ.get("REPLOT"):
        d = json.load(open(cache)); maps = {k: np.array(d["maps"][k]) for k in keys}
        print("loaded cache — replot", flush=True)
    else:
        maps = {k: np.full((len(GRID), len(GRID)), np.nan) for k in keys}
        for i, t1 in enumerate(GRID):
            for j, t2 in enumerate(GRID):
                th = np.array([t1, t2]); ref = gnoisy(th)
                if np.linalg.norm(ref) < G_MIN:
                    continue
                rng = np.random.default_rng(17 * i + 5 * j + 1)
                maps["PSR"][j, i] = angle_deg(psr_g(th, rng), ref)
                maps["NSR"][j, i] = angle_deg(nsr_g(th, rng), ref)
                for e in EPS_FLOORS:
                    maps[f"FD{e}"][j, i] = angle_deg(fd_g(th, e, rng), ref)
            print(f"  row {i+1}/{len(GRID)} done", flush=True)
        json.dump({"grid": GRID.tolist(), "maps": {k: maps[k].tolist() for k in keys},
                   "G_min": G_MIN, "B": B_BUDGET, "eps_floors": EPS_FLOORS},
                  open(cache, "w"), default=float)

    titles = {"PSR": "PSR (ε-free)", "NSR": r"NSR ($M=\infty$, ε-free)",
              f"FD{EPS_FLOORS[0]}": f"FD ε={EPS_FLOORS[0]} (floored)",
              f"FD{EPS_FLOORS[1]}": f"FD ε={EPS_FLOORS[1]} (floored)"}
    order = ["PSR", "NSR", f"FD{EPS_FLOORS[0]}", f"FD{EPS_FLOORS[1]}"]
    fig, axs = plt.subplots(2, 2, figsize=(9.2, 8.0), sharex=True, sharey=True)
    ext = [GRID[0], GRID[-1], GRID[0], GRID[-1]]
    im = None
    for ax, k in zip(axs.flat, order):
        m = maps[k]
        im = ax.imshow(np.minimum(m, 20.0), origin="lower", extent=ext, vmin=0, vmax=20,
                       cmap="viridis", aspect="auto")
        bad = m > 90
        if bad.any():
            yy, xx = np.where(bad)
            ax.plot(GRID[xx], GRID[yy], "x", color="#d62728", ms=9, mew=2.4,
                    label="uphill (>90°): trap zone")
            ax.legend(frameon=False, fontsize=7.5, loc="upper right")
        v = m[~np.isnan(m)]
        ax.set_title(f"{titles[k]}\nmedian {np.nanmedian(v):.1f}°, max {np.nanmax(v):.0f}°, "
                     f">90° at {100*np.mean(v>90):.0f}%", fontsize=8.5)
    for ax in axs[-1]:
        ax.set_xlabel(r"$\theta_1$ (ZZ coupling)")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$\theta_2$ (X field)")
    cb = fig.colorbar(im, ax=axs, shrink=0.85)
    cb.set_label(r"angle of gradient to $\nabla C_{\rm noisy}$ (deg, capped at 20)")
    fig.suptitle("F-loop — gradient DIRECTION vs the realized objective's gradient "
                 r"$\nabla C_{\rm noisy}$"
                 "\nε-free PSR/NSR stay in the descent cone; floored-ε FD points uphill (trap zones, "
                 "red ×)\n(TFIM 2q; compiled to machine-native segments, emulated under T4; "
                 r"$T/T_2^*$=0.15)", fontsize=9.5)
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"F_loop_direction.{e}"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote F_loop_direction.pdf/.png/.json")
    for k in order:
        v = maps[k][~np.isnan(maps[k])]
        print(f"  {k:8s}: median {np.median(v):5.1f}°  max {v.max():6.1f}°  "
              f">90° (uphill) at {100*np.mean(v>90):.0f}% of valid points")


if __name__ == "__main__":
    main()
