"""
psr_vs_fd_device_gradient.py — raw PSR vs oracle-FD as estimators of the EXACT
device gradient  ∇C_noisy  (DiffSimuQ, device-target, NO rescale).

Background: analog PSR computes ∇C_noisy EXACTLY even under non-unitary Lindblad
noise — the ±kick shift identity  K_+ρK_+† − K_-ρK_-† = −i[H_j,ρ]  is algebraic
(holds for any mixed ρ), so the surrounding channel may be arbitrary.  PSR uses
no finite step ε, so hardware control error δ never gets 1/ε-amplified as in FD.
Cost = ⟨Z0Z1⟩ under dephasing;  ansatz H(θ)=θ1 Z0Z1 + θ2 (X0+X1);  T=1.5.

Part A (∞-shot, gradient accuracy vs control resolution r, RELATIVE error, over
  several GENERIC points and two noise levels):
    raw PSR = exact device gradient (~1e-4, theorem);  oracle-FD's δ/ε error is
    a CONTROL-resolution effect — γ-INDEPENDENT (the two γ curves overlap).
    -> figures/psr_fd_device_gradient_multipt.png
Part B (finite shots at fixed r):
    PSR converges ~N^{-1/2} to the device gradient; oracle-FD FLOORS at the δ/ε
    control-error bias (not shot-reducible).  -> figures/psr_fd_device_finite_shot.png

Control resolution r floors FD's realizable ε AND sets the setpoint error δ~N(0,r)
(same hardware quantity), so at the floor δ/ε ~ O(1) unavoidably.  PSR has neither.

Cache: figures/psr_fd_device_gradient_data.json (re-plots without re-running sims).
Run:  conda run -n qec_pg python differential_computing/tests/psr_vs_fd_device_gradient.py
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

T = 1.5
POOL = 48
NS_CAP = 400
I2 = qp.qeye(2); X = qp.sigmax(); Z = qp.sigmaz()
ZZ = qp.tensor(Z, Z); XD = qp.tensor(X, I2) + qp.tensor(I2, X); O = ZZ
PSI0 = qp.tensor(qp.basis(2, 0), qp.basis(2, 0)); RHO0 = PSI0 * PSI0.dag()
FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
CACHE = os.path.join(FIGDIR, "psr_fd_device_gradient_data.json")

# Part A: generic points (away from gradient zeros) + two noise levels
PTS = [(0.8, 0.65), (0.4, 0.7), (1.0, 0.4), (0.7, 0.85), (0.5, 0.35)]
RS = [0.01, 0.03, 0.06, 0.1, 0.15]
GAM = [("T/T2*=0.15", 10.0, "#1f77b4"), ("T/T2*=0.5", 3.0, "#d62728")]
ND = 8
# Part B: finite-shot at one generic point, moderate control resolution
TH_B = (0.4, 0.7); T2_B = 10.0; R_B = 0.05
NS_SHOT = [100, 300, 1000, 3000, 10000, 30000, 100000]
EPS_B = [0.05, 0.1, 0.15, 0.2, 0.3]
SEEDS_B = 24


def Hq(a, b):
    return a * ZZ + b * XD


def make(T2):
    """Evaluators for a given dephasing time T2 (fresh mesolve cache)."""
    G = 1.0 / (2.0 * T2)
    C = [np.sqrt(G) * qp.tensor(Z, I2), np.sqrt(G) * qp.tensor(I2, Z)]
    cache = {}

    def nobs(a, b):                                    # C_noisy = <ZZ> under dephasing
        k = (round(a, 6), round(b, 6))
        if k not in cache:
            r = qp.mesolve(Hq(a, b), RHO0, [0, T], c_ops=C).states[-1]
            cache[k] = float(qp.expect(O, r).real)
        return cache[k]

    def grad_true(a, b, h=1e-4):                       # perfect FD of C_noisy = ∇C_noisy
        return np.array([(nobs(a + h, b) - nobs(a - h, b)) / (2 * h),
                         (nobs(a, b + h) - nobs(a, b - h)) / (2 * h)])

    def bexp(H_list):                                  # 3-seg branch: noisy base + ideal kick
        rho = RHO0
        for kk, (Hs, dur) in enumerate(H_list):
            Hqo = Hs.to_qutip_qobj()
            if kk == 1:
                U = (-1j * Hqo * float(dur)).expm(); rho = U * rho * U.dag()
            else:
                rho = qp.mesolve(Hqo, rho, [0, float(dur)], c_ops=C).states[-1]
        return float(qp.expect(O, rho).real)

    def blocks(a, b):                                  # PSR branch expectations per param
        out = []
        for pl in range(2):
            qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
            s = sp.Symbol("v"); zz = q[0].Z * q[1].Z; xd = q[0].X + q[1].X
            H = (s * zz + b * xd) if pl == 0 else (a * zz + s * xd)
            val = a if pl == 0 else b
            orig = np.random.rand
            np.random.rand = lambda kk: (np.arange(kk) + 0.5) / kk   # deterministic midpoint τ
            try:
                progs = observable_program_generator(H, T, n_sample=POOL,
                                                     n_repetition=1, diff_var="v",
                                                     value=float(val))
            finally:
                np.random.rand = orig
            bl = []
            for H_tot, ug, _ in progs:
                bb = len(H_tot) // 2
                em = np.array([bexp(H_tot[2 * i]) for i in range(bb)])
                ep = np.array([bexp(H_tot[2 * i + 1]) for i in range(bb)])
                bl.append((em, ep, float(ug)))
            out.append(bl)
        return out

    def psr_det(a, b):                                 # deterministic PSR grad = ∇C_noisy
        bl = blocks(a, b)
        return np.array([sum((T / len(em)) * ug * np.sum(em - ep) for (em, ep, ug) in bl[l])
                         for l in range(2)]), bl

    def fd_delta(a, b, eps, r, rng):                   # ∞-shot FD with control error on shifts
        g = np.zeros(2)
        for l in range(2):
            e = np.zeros(2); e[l] = eps; dp = rng.normal(0, r, 2); dm = rng.normal(0, r, 2)
            g[l] = (nobs(a + e[0] + dp[0], b + e[1] + dp[1])
                    - nobs(a - e[0] + dm[0], b - e[1] + dm[1])) / (2 * eps)
        return g

    return nobs, grad_true, psr_det, fd_delta


def psr_shot(bl, N, rng):
    """Finite-shot PSR gradient from precomputed branch expectations (budget N/param)."""
    g = np.zeros(2)
    for l in range(2):
        blk = bl[l]; M = len(blk)
        ns = int(min(NS_CAP, max(1, N // (2 * M))))
        npr = int(max(1, round(N / (2 * M * ns))))
        for (em, ep, ug) in blk:
            idx = rng.integers(0, len(em), size=ns)
            fm = 2.0 * rng.binomial(npr, 0.5 * (1 + np.clip(em[idx], -1, 1))) / npr - 1
            fp = 2.0 * rng.binomial(npr, 0.5 * (1 + np.clip(ep[idx], -1, 1))) / npr - 1
            g[l] += (T / ns) * ug * np.sum(fm - fp)
    return g


def fd_shot(vplus, vminus, eps, N, rng):
    """Finite-shot FD from precomputed δ-shifted deterministic values (budget N/param)."""
    n = max(1, N // 2); g = np.zeros(2)
    for l in range(2):
        fp = 2.0 * rng.binomial(n, 0.5 * (1 + np.clip(vplus[l], -1, 1))) / n - 1
        fm = 2.0 * rng.binomial(n, 0.5 * (1 + np.clip(vminus[l], -1, 1))) / n - 1
        g[l] = (fp - fm) / (2 * eps)
    return g


def compute():
    d = {"partA": {}, "partB": {}}
    # ---- Part A : ∞-shot relative gradient error over points × γ ----
    for label, T2, _ in GAM:
        nobs, grad_true, psr_det, fd_delta = make(T2)
        rows = {"psr_rel": [], "fd_rel": []}
        for (a, b) in PTS:
            g0 = grad_true(a, b); ng = float(np.linalg.norm(g0))
            gp, _ = psr_det(a, b)
            rows["psr_rel"].append(float(np.linalg.norm(gp - g0) / ng))
            rr = []
            for r in RS:
                best = min(float(np.sqrt(np.mean([
                    (np.linalg.norm(fd_delta(a, b, eps, r, np.random.default_rng(s)) - g0) / ng) ** 2
                    for s in range(ND)])))
                    for eps in np.geomspace(r, 0.6, 6))
                rr.append(best)
            rows["fd_rel"].append(rr)
            print(f"  A {label} pt=({a},{b}) |g|={ng:.2f} PSR_rel={rows['psr_rel'][-1]:.1e} "
                  f"FD_rel(r=.15)={rr[-1]:.2f}", flush=True)
        d["partA"][label] = rows
    # ---- Part B : finite-shot at one point ----
    nobs, grad_true, psr_det, _ = make(T2_B)
    gt, bl = psr_det(*TH_B); ng = float(np.linalg.norm(gt))
    print(f"  B point={TH_B} |∇C_noisy|={ng:.3f}  det-PSR err vs perfect-FD="
          f"{np.linalg.norm(gt-grad_true(*TH_B))/ng:.1e}", flush=True)
    seed_fd = []                                        # per-run δ-shifted deterministic values
    for s in range(SEEDS_B):
        rng = np.random.default_rng(1000 + s)
        dp = rng.normal(0, R_B, 2); dm = rng.normal(0, R_B, 2)
        per = {}
        for eps in EPS_B:
            vp = np.array([nobs(TH_B[0] + eps * (l == 0) + dp[0],
                                TH_B[1] + eps * (l == 1) + dp[1]) for l in range(2)])
            vm = np.array([nobs(TH_B[0] - eps * (l == 0) + dm[0],
                                TH_B[1] - eps * (l == 1) + dm[1]) for l in range(2)])
            per[eps] = (vp, vm)
        seed_fd.append(per)
    psr_rmse, fd_rmse, fd_eps = [], [], []
    for N in NS_SHOT:
        pe = [np.linalg.norm(psr_shot(bl, N, np.random.default_rng(7000 + s)) - gt) / ng
              for s in range(SEEDS_B)]
        psr_rmse.append(float(np.sqrt(np.mean(np.square(pe)))))
        best = None
        for eps in EPS_B:
            fe = [np.linalg.norm(fd_shot(seed_fd[s][eps][0], seed_fd[s][eps][1], eps, N,
                                         np.random.default_rng(9000 + s)) - gt) / ng
                  for s in range(SEEDS_B)]
            rms = float(np.sqrt(np.mean(np.square(fe))))
            if best is None or rms < best[0]:
                best = (rms, eps)
        fd_rmse.append(best[0]); fd_eps.append(best[1])
        print(f"  B N={N:<7} PSR rmse={psr_rmse[-1]:.4f}  oracle-FD rmse={best[0]:.4f} "
              f"(eps={best[1]})", flush=True)
    d["partB"] = {"point": list(TH_B), "r": R_B, "T2": T2_B, "N": NS_SHOT,
                  "psr_rmse": psr_rmse, "fd_rmse": fd_rmse, "fd_eps": fd_eps,
                  "psr_floor": float(np.linalg.norm(gt - grad_true(*TH_B)) / ng)}
    json.dump(d, open(CACHE, "w"), default=float)
    return d


def plot(d):
    # Fig A
    fig, ax = plt.subplots(figsize=(7.8, 5.4), dpi=150); psr_all = []
    for label, _, col in GAM:
        rows = d["partA"][label]; psr_all += rows["psr_rel"]
        for rr in rows["fd_rel"]:
            ax.plot(RS, rr, color=col, lw=0.7, alpha=0.35)
        ax.plot(RS, np.median(rows["fd_rel"], 0), color=col, lw=2.6, marker="s",
                label=f"oracle-FD, {label} (median, {len(PTS)} pts)")
    ax.axhline(np.median(psr_all), color="#00897b", lw=2.6,
               label=f"raw PSR (both γ, all pts): ~{np.median(psr_all):.0e}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("control resolution  r   (floors ε, sets δ)")
    ax.set_ylabel(r"relative gradient error  $\|\hat g-\nabla C_{\rm noisy}\|/\|\nabla C_{\rm noisy}\|$")
    ax.set_title("Device gradient ∇C_noisy (relative error, generic points):\nFD's δ/ε "
                 "disadvantage is CONTROL-resolution — γ-independent; PSR exact", fontsize=9.2)
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.18); fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "psr_fd_device_gradient_multipt.png")); plt.close(fig)
    # Fig B
    b = d["partB"]; N = np.array(b["N"])
    fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=150)
    ax.loglog(N, b["psr_rmse"], "o-", color="#00897b", lw=2.6, label="raw PSR (exact device gradient)")
    ax.loglog(N, b["fd_rmse"], "s-", color="#7b1fa2", lw=2.4, label=f"oracle-FD (best ε, control error δ~r={b['r']})")
    ax.loglog(N, b["psr_rmse"][0] * (N / N[0]) ** -0.5, "--", color="#555", lw=1,
              alpha=0.7, label=r"$N^{-1/2}$")
    fdfloor = np.median(b["fd_rmse"][-2:])
    ax.axhline(fdfloor, color="#7b1fa2", ls=":", lw=1.3, alpha=0.8,
               label=f"FD δ/ε bias floor (~{fdfloor:.2f}, not shot-reducible)")
    ax.set_xlabel("total shots per gradient component  N")
    ax.set_ylabel(r"relative gradient RMSE vs $\nabla C_{\rm noisy}$")
    ax.set_title(f"Finite shots, device gradient (point={tuple(b['point'])}, r={b['r']}):\n"
                 "PSR converges to the exact device gradient; oracle-FD FLOORS at δ/ε", fontsize=9.3)
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.18); fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "psr_fd_device_finite_shot.png")); plt.close(fig)
    print("saved: psr_fd_device_gradient_multipt.png, psr_fd_device_finite_shot.png")


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    if os.path.exists(CACHE):
        d = json.load(open(CACHE)); print("loaded cache — replotting only")
    else:
        d = compute()
    plot(d)


if __name__ == "__main__":
    main()
