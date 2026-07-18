"""
multivar_direction_cases.py (Case A extension) — tuned-FD + more cases, for the
multivariable gradient-direction comparison.

At ∞ shots FD's best ε is ε→0 = the noisy-landscape gradient = raw PSR (lemma),
so "tuned FD" is only a distinct estimator at FINITE shots.  This study:

1. THREE CASES with different parameter structure (per-parameter single-symbol
   Hamiltonians, h2_vqe pattern; T/T2* = 0.15, det-τ):
     C1  2q:  θ1·Z0Z1 + θ2·(X0+X1),           obs Z0Z1   (baseline, M=1/2)
     C2  3q:  θ1·(Z0Z1+Z1Z2) + θ2·ΣX,         obs Z0Z1   (chain, M=2/3)
     C3  2q:  θ1·Z0 + θ2·Z0Z1 + 1.0·(X0+X1),  obs Z0     (field vs coupling)

2. ∞-SHOT grid stats per case (9×9): median/max angle + uphill% for PSR raw,
   PSR rescaled (component-gated), FD floored ε=0.3/0.6.

3. FINITE-SHOT comparison at a moderate-|g| operating point per case, equal
   total budget N split evenly over parameters:
     FD ORACLE-ε — per-component, per-budget ε chosen to minimize that
       component's actual RMSE (maximally generous to FD);
     PSR raw — τ-pool subsampling, optimal split (n_per=1, max τ-samples);
     PSR rescaled — analytic per-component factors with the gating rule
       (|ĝ_ℓ| ≥ 0.1, factor ∈ [0.25, 4]) applied per realization.
   Metrics over R realizations: mean angle to g_true, P(angle > 90°).

Run:  conda run -n qec_pg python differential_computing/tests/multivar_direction_cases.py
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

T, T2 = 1.5, 10.0
H_FD = 1e-3
GRID = np.linspace(0.2, 1.4, 9)
G_MIN, G_COMP, FAC_LO, FAC_HI = 0.15, 0.10, 0.25, 4.0
EPS_FLOORS = [0.3, 0.6]
EPS_TUNE = np.geomspace(0.03, 1.0, 12)
BUDGETS = [1000, 10000, 100000]
POOL, NS_CAP, R = 300, 1500, 1500

I2 = qp.qeye(2); X = qp.sigmax(); Z = qp.sigmaz()


def emb(op, i, n):
    return qp.tensor([op if k == i else I2 for k in range(n)])


def _case(name, n, terms, fixed, obs):
    """terms: list per parameter of (qutip_op, simuq_builder); fixed: qutip op."""
    return dict(name=name, n=n, terms=terms, fixed=fixed, obs=obs,
                psi0=qp.tensor([qp.basis(2, 0)] * n))


def build_cases():
    # C1
    n = 2
    zz = emb(Z, 0, n) * emb(Z, 1, n); xs = emb(X, 0, n) + emb(X, 1, n)
    C1 = _case("C1: θ1·ZZ + θ2·ΣX (2q)", n, [zz, xs], 0.0 * zz, zz)
    # C2
    n = 3
    zz2 = emb(Z, 0, n) * emb(Z, 1, n) + emb(Z, 1, n) * emb(Z, 2, n)
    xs3 = sum(emb(X, i, n) for i in range(n))
    C2 = _case("C2: θ1·(ZZ+ZZ) + θ2·ΣX (3q)", n, [zz2, xs3],
               0.0 * xs3, emb(Z, 0, n) * emb(Z, 1, n))
    # C3
    n = 2
    z0 = emb(Z, 0, n); zz = emb(Z, 0, n) * emb(Z, 1, n)
    xfix = 1.0 * (emb(X, 0, n) + emb(X, 1, n))
    C3 = _case("C3: θ1·Z0 + θ2·ZZ + X-drive (2q)", n, [z0, zz], xfix,
               emb(Z, 0, n))
    return [C1, C2, C3]


def simuq_param_H(case, ell, theta):
    """Single-symbol simuq Hamiltonian for parameter ell at point theta."""
    n = case["n"]
    qs = QSystem(); q = [Qubit(qs) for _ in range(n)]
    s = sp.Symbol("v")

    def sq_term(idx):
        if case["name"].startswith("C1"):
            return [q[0].Z * q[1].Z, q[0].X + q[1].X][idx]
        if case["name"].startswith("C2"):
            return [q[0].Z * q[1].Z + q[1].Z * q[2].Z,
                    q[0].X + q[1].X + q[2].X][idx]
        return [q[0].Z * 1.0, q[0].Z * q[1].Z][idx]

    H = s * sq_term(ell)
    other = 1 - ell
    H = H + float(theta[other]) * sq_term(other)
    if case["name"].startswith("C3"):
        H = H + 1.0 * (q[0].X + q[1].X)
    return H


def main():
    figdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    os.makedirs(figdir, exist_ok=True)
    cache = os.path.join(figdir, "multivar_direction_cases_data.json")

    def Hq(case, t1, t2):
        return t1 * case["terms"][0] + t2 * case["terms"][1] + case["fixed"]

    def angle_deg(v, ref):
        c = float(np.dot(v, ref) / (np.linalg.norm(v) * np.linalg.norm(ref)))
        return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

    if os.path.exists(cache):
        d = json.load(open(cache))
        print("loaded cache — replotting only")
    else:
        rng = np.random.default_rng(0)
        d = {"cases": []}
        for case in build_cases():
            n, obs, psi0 = case["n"], case["obs"], case["psi0"]
            noisy = NoisyQuTiPRunner(n, noise=NoiseModel(n_qubits=n, T2=T2))
            nx = noisy.make_expectation_fn(psi0, obs)
            print(f"\n=== {case['name']} ===", flush=True)

            def fclean(t1, t2):
                return float(qp.expect(
                    obs, (-1j * Hq(case, t1, t2) * T).expm() * psi0).real)

            # noisy landscape via density matrix (qutip mesolve on qobj H)
            c_ops = [np.sqrt(1.0 / (2 * T2)) * emb(Z, i, n) for i in range(n)]

            def fnoisy(t1, t2):
                rho = psi0 * psi0.dag()
                res = qp.mesolve(Hq(case, t1, t2), rho, [0.0, T], c_ops=c_ops)
                return float(qp.expect(obs, res.states[-1]).real)

            def grad2(f, t1, t2, h):
                return np.array([(f(t1 + h, t2) - f(t1 - h, t2)) / (2 * h),
                                 (f(t1, t2 + h) - f(t1, t2 - h)) / (2 * h)])

            def factors_at(t1, t2, g_true):
                out = []
                for ell in range(2):
                    e = np.zeros(2); e[ell] = H_FD
                    dgdG = (ar.dO_dGamma(Hq(case, t1 + e[0], t2 + e[1]), obs,
                                         psi0, T, n, range(n), n_grid=100)
                            - ar.dO_dGamma(Hq(case, t1 - e[0], t2 - e[1]), obs,
                                           psi0, T, n, range(n), n_grid=100)) \
                        / (2 * H_FD)
                    slope = (dgdG / g_true[ell]) / (2.0 * T)
                    out.append(ar.rescale_factor(slope, T, T2))
                return np.array(out)

            # ── ∞-shot grid stats ──
            stats = {k: [] for k in
                     ["raw", "resc"] + [f"fd{e}" for e in EPS_FLOORS]}
            best_pt, best_min = None, -1
            for t1 in GRID:
                for t2 in GRID:
                    g_true = grad2(fclean, t1, t2, H_FD)
                    if np.linalg.norm(g_true) < G_MIN:
                        continue
                    g_raw = grad2(fnoisy, t1, t2, H_FD)
                    fac = factors_at(t1, t2, g_true)
                    fac_g = np.where((np.abs(g_raw) >= G_COMP) &
                                     (fac >= FAC_LO) & (fac <= FAC_HI),
                                     fac, 1.0)
                    stats["raw"].append(angle_deg(g_raw, g_true))
                    stats["resc"].append(angle_deg(g_raw * fac_g, g_true))
                    for e in EPS_FLOORS:
                        stats[f"fd{e}"].append(
                            angle_deg(grad2(fnoisy, t1, t2, e), g_true))
                    m = min(abs(g_true[0]), abs(g_true[1]))
                    if m > best_min:
                        best_min, best_pt = m, (float(t1), float(t2))
                print(f"  grid row t1={t1:.2f} done", flush=True)
            summary = {k: dict(median=float(np.median(v)),
                               max=float(np.max(v)),
                               uphill=float(np.mean(np.array(v) > 90)))
                       for k, v in stats.items()}

            # ── finite-shot comparison at best_pt ──
            t1, t2 = best_pt
            g_true = grad2(fclean, t1, t2, H_FD)
            fac = factors_at(t1, t2, g_true)
            print(f"  operating point {best_pt}, g_true={g_true}, fac={fac}")

            # PSR pools per parameter (deterministic τ)
            pools = []
            for ell in range(2):
                Hp = simuq_param_H(case, ell, (t1, t2))
                orig = np.random.rand
                np.random.rand = lambda k: (np.arange(k) + 0.5) / k
                try:
                    progs = observable_program_generator(
                        Hp, T, n_sample=POOL, n_repetition=1, diff_var="v",
                        value=float((t1, t2)[ell]))
                finally:
                    np.random.rand = orig
                pp = []
                for H_tot, ug, _ in progs:
                    b = len(H_tot) // 2
                    em = np.array([nx(H_tot[2 * i]) for i in range(b)])
                    ep = np.array([nx(H_tot[2 * i + 1]) for i in range(b)])
                    pp.append((em, ep, float(ug)))
                pools.append(pp)
                print(f"  pool for θ{ell + 1} built ({len(pp)} programs)",
                      flush=True)

            # FD exact endpoints per tuning ε per component
            fd_ends = {float(e): [(fnoisy(t1 + e, t2), fnoisy(t1 - e, t2)),
                                  (fnoisy(t1, t2 + e), fnoisy(t1, t2 - e))]
                       for e in EPS_TUNE}

            def shot(p_exact, nsh, size):
                p = 0.5 * (1 + np.clip(p_exact, -1, 1))
                return 2.0 * rng.binomial(int(max(1, nsh)), p, size=size) \
                    / max(1, nsh) - 1

            fin = []
            for N in BUDGETS:
                Np = N // 2                       # per parameter
                # PSR vectors
                gs = np.zeros((R, 2))
                for ell in range(2):
                    M = len(pools[ell])
                    ns = int(min(NS_CAP, max(1, Np // (2 * M))))
                    n_per = int(max(1, round(Np / (2 * M * ns))))
                    for (em, ep, ug) in pools[ell]:
                        idx = rng.integers(0, len(em), size=(R, ns))
                        fm = 2.0 * rng.binomial(n_per, 0.5 * (1 + np.clip(em[idx], -1, 1))) / n_per - 1
                        fp = 2.0 * rng.binomial(n_per, 0.5 * (1 + np.clip(ep[idx], -1, 1))) / n_per - 1
                        gs[:, ell] += (T / ns) * ug * np.sum(fm - fp, axis=1)
                # gated rescale per realization
                fac_g = np.where((np.abs(gs) >= G_COMP) &
                                 (fac >= FAC_LO) & (fac <= FAC_HI), fac, 1.0)
                gres = gs * fac_g
                # FD oracle-ε per component
                nfd = int(max(1, Np // 2))
                fd_comp = np.zeros((R, 2))
                for ell in range(2):
                    best_rmse, best_est = np.inf, None
                    for e in EPS_TUNE:
                        fpx, fmx = fd_ends[float(e)][ell]
                        est = (shot(fpx, nfd, R) - shot(fmx, nfd, R)) / (2 * e)
                        rmse = float(np.sqrt(np.mean((est - g_true[ell]) ** 2)))
                        if rmse < best_rmse:
                            best_rmse, best_est = rmse, est
                    fd_comp[:, ell] = best_est
                row = dict(N=int(N))
                for nm, arr in (("psr_raw", gs), ("psr_resc", gres),
                                ("fd_oracle", fd_comp)):
                    angs = np.array([angle_deg(arr[r_], g_true)
                                     for r_ in range(R)])
                    row[nm] = dict(mean_angle=float(np.mean(angs)),
                                   uphill=float(np.mean(angs > 90)))
                fin.append(row)
                print(f"  N={N}: " + "  ".join(
                    f"{nm} {row[nm]['mean_angle']:.1f}°/{row[nm]['uphill']:.1%}"
                    for nm in ("fd_oracle", "psr_raw", "psr_resc")), flush=True)

            d["cases"].append(dict(name=case["name"], summary=summary,
                                   point=list(best_pt),
                                   g_true=list(map(float, g_true)),
                                   factors=list(map(float, fac)), finite=fin))
        json.dump(d, open(cache, "w"), default=float)
        print(f"\ncached: {cache}")

    # ── report + figure ──
    print("\n── ∞-shot grid stats (median° / max° / uphill%) ──")
    for c in d["cases"]:
        print(f"{c['name']}:")
        for k, s in c["summary"].items():
            print(f"   {k:>6}: {s['median']:6.2f}° / {s['max']:6.1f}° / "
                  f"{100 * s['uphill']:4.1f}%")

    fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.4), dpi=150, sharey=True)
    colors = {"fd_oracle": "#7b1fa2", "psr_raw": "#9e9e9e",
              "psr_resc": "#00897b"}
    labels = {"fd_oracle": "FD oracle-ε (per-component tuned)",
              "psr_raw": "PSR raw", "psr_resc": "PSR rescaled (gated)"}
    for ax, c in zip(axs, d["cases"]):
        Ns = [r["N"] for r in c["finite"]]
        for nm in ("fd_oracle", "psr_raw", "psr_resc"):
            ax.semilogx(Ns, [r[nm]["mean_angle"] for r in c["finite"]],
                        "o-", color=colors[nm], lw=2, label=labels[nm])
            for r in c["finite"]:
                if r[nm]["uphill"] > 0.01:
                    ax.annotate(f"{r[nm]['uphill']:.0%}↑",
                                (r["N"], r[nm]["mean_angle"]),
                                textcoords="offset points", xytext=(4, 5),
                                fontsize=7, color=colors[nm])
        ax.set_title(c["name"], fontsize=9.5)
        ax.set_xlabel("total shots N (split over parameters)")
        ax.grid(True, which="both", axis="y", alpha=0.15)
    axs[0].set_ylabel("mean angle to true gradient (deg)")
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Finite-shot gradient DIRECTION at equal budget (T/T2*=0.15): "
                 "oracle-tuned FD vs PSR — annotations = P(uphill)",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(figdir, "multivar_direction_cases.png")
    fig.savefig(out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
