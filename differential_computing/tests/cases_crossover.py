"""
cases_crossover.py — PSR (rescaled) vs FD (best ε) across several TFIM-like cases:
where is the CROSSOVER shot budget for each?

Cases vary the chain-rule M (#terms a parameter affects) and qubit count n.  For
each: pick a robust point, compute the analytic rescale (full-system slope), build
the multi-term noisy PSR pool (gradient SUMS over all M programs), and sweep total
shots N to find where PSR-rescaled RMSE drops below FD-best-ε RMSE.

Prediction: the crossover N grows with M (PSR's variance ∝ M from splitting the
budget over 2·M·n_sample branches; FD is always 2 evals).

Run:  conda run -n qec_pg python differential_computing/tests/cases_crossover.py
"""

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

T, T2 = 1.5, 20.0         # moderate attenuation (λ~0.7-0.85) so the rescale matters
R, POOL, NS_CAP = 2000, 400, 1000
BUDGETS = np.array([200, 1000, 5000, 20000, 80000, 300000])
I = qp.qeye(2); X, Z = qp.sigmax(), qp.sigmaz()


def emb(op, i, n):
    return qp.tensor([op if k == i else I for k in range(n)])


def zz(i, j, n): return emb(Z, i, n) * emb(Z, j, n)


# ── cases: (name, n, simuq H builder, qutip H(th) builder, qutip observable) ──
def make_cases():
    cases = []

    def tfim_zz(n):                      # θ on ΣZZ; obs Z0Z1; M=n-1
        def sq():
            x = sp.Symbol("x"); qs = QSystem(); q = [Qubit(qs) for _ in range(n)]
            H = x * sum((q[i].Z*q[i+1].Z for i in range(n-1)), 0*q[0].Z)
            for i in range(n): H = H + q[i].X
            return H, "x"
        def hq(th):
            H = th*sum(zz(i, i+1, n) for i in range(n-1))
            for i in range(n): H = H + emb(X, i, n)
            return H
        return sq, hq, zz(0, 1, n)

    for n in (2, 3, 4):
        sq, hq, ob = tfim_zz(n)
        cases.append((f"{n}q ZZ-param (M={n-1})", n, sq, hq, ob, n-1))

    # 4q with the parameter on the X field (M=4)
    n = 4
    def sq4x():
        x = sp.Symbol("x"); qs = QSystem(); q = [Qubit(qs) for _ in range(n)]
        H = sum((q[i].Z*q[i+1].Z for i in range(n-1)), 0*q[0].Z)
        for i in range(n): H = H + x*q[i].X
        return H, "x"
    def hq4x(th):
        H = sum(zz(i, i+1, n) for i in range(n-1))
        for i in range(n): H = H + th*emb(X, i, n)
        return H
    cases.append(("4q X-param (M=4)", n, sq4x, hq4x, zz(0, 1, n), 4))
    return cases


def run_case(name, nq, sq_fn, hq_fn, obs, rng):
    PSIn = qp.tensor([qp.basis(2, 0)] * nq)
    clean = NoisyQuTiPRunner(nq, noise=None)
    noisy = NoisyQuTiPRunner(nq, noise=NoiseModel(n_qubits=nq, T2=T2))
    H, var = sq_fn()

    def fc(th):
        return float(qp.expect(obs, (-1j*hq_fn(th)*T).expm()*PSIn).real)
    fnz = noisy.make_expectation_fn(PSIn, obs)
    def fn(th):
        return fnz([[H.set_parameterizedHam({var: float(th)}), T]])

    xs = np.linspace(0.2, 2.0, 48)
    gi = np.array([(fc(x+1e-3)-fc(x-1e-3))/2e-3 for x in xs])
    x_star, best = None, -1
    for k, x in enumerate(xs):
        if abs(gi[k]) > 0.3:
            lam = (fn(x+1e-2)-fn(x-1e-2))/2e-2 / gi[k]
            if 0.55 < lam < 0.95 and abs(gi[k]) > best:
                x_star, best = float(x), abs(gi[k])
    if x_star is None:
        x_star = float(xs[np.argmax(np.abs(gi))])
    g_real = (fc(x_star+1e-3)-fc(x_star-1e-3))/2e-3
    s = ar.lambda_slope(hq_fn, obs, PSIn, T, nq, z_sites=range(nq), theta=x_star,
                        n_grid=120)
    factor = ar.rescale_factor(s, T, T2)

    np.random.seed(123)
    progs = observable_program_generator(H, T, n_sample=POOL, n_repetition=1,
                                         diff_var=var, value=x_star)
    pexp = noisy.make_expectation_fn(PSIn, obs)
    pools = []
    for H_tot, ug_j, _ in progs:
        bj = len(H_tot)//2
        e = np.array([pexp(H_tot[2*i]) for i in range(bj)])
        p = np.array([pexp(H_tot[2*i+1]) for i in range(bj)])
        pools.append((e, p, float(ug_j)))
    M = len(pools)
    psr_mean = sum((T/len(e))*u*np.sum(e-p) for e, p, u in pools)

    eps_grid = np.geomspace(0.03, 1.0, 11)
    fd_pm = {ee: (fn(x_star+ee), fn(x_star-ee)) for ee in eps_grid}
    fd_best, psr_res = [], []
    for N in BUDGETS:
        nfd = N//2; bst = np.inf
        for ee in eps_grid:
            fp, fm = fd_pm[ee]
            a = 2.0*rng.binomial(nfd, 0.5*(1+np.clip(fp, -1, 1)), size=R)/nfd-1
            bb = 2.0*rng.binomial(nfd, 0.5*(1+np.clip(fm, -1, 1)), size=R)/nfd-1
            bst = min(bst, float(np.sqrt(np.mean(((a-bb)/(2*ee)-g_real)**2))))
        fd_best.append(bst)
        ns = int(min(N//(2*M), NS_CAP)); ns = max(1, ns)
        n_per = int(max(1, round(N/(2*M*ns))))
        tot = np.zeros(R)
        for e, p, u in pools:
            idx = rng.integers(0, len(e), size=(R, ns))
            fm = 2.0*rng.binomial(n_per, 0.5*(1+np.clip(e[idx], -1, 1)))/n_per-1
            fp = 2.0*rng.binomial(n_per, 0.5*(1+np.clip(p[idx], -1, 1)))/n_per-1
            tot += (T/ns)*u*np.sum(fm-fp, axis=1)
        psr_res.append(float(np.sqrt(np.mean((tot*factor-g_real)**2))))
    fd_best, psr_res = np.array(fd_best), np.array(psr_res)

    # crossover: first N where PSR < FD (log-log interpolation)
    d = psr_res - fd_best
    cross = None
    for i in range(len(BUDGETS)-1):
        if d[i] > 0 and d[i+1] <= 0:
            lo, hi = np.log(BUDGETS[i]), np.log(BUDGETS[i+1])
            t = d[i]/(d[i]-d[i+1])
            cross = float(np.exp(lo + t*(hi-lo))); break
    if cross is None:
        cross = BUDGETS[0] if d[0] <= 0 else np.inf
    return dict(name=name, M=M, lam=psr_mean/g_real, factor=factor,
                fd=fd_best, psr=psr_res, cross=cross)


def main():
    rng = np.random.default_rng(0)
    res = [run_case(*c[:5], rng) for c in make_cases()]
    print(f"{'case':>22}{'M':>3}{'λ_act':>8}{'1/λ':>7}{'crossover N':>13}"
          f"{'FD@max':>9}{'PSR@max':>9}")
    for r in res:
        cr = f"{r['cross']:.0f}" if np.isfinite(r['cross']) else "never"
        print(f"{r['name']:>22}{r['M']:>3}{r['lam']:>8.3f}{r['factor']:>7.2f}"
              f"{cr:>13}{r['fd'][-1]:>9.4f}{r['psr'][-1]:>9.4f}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=150)
    cols = plt.cm.viridis(np.linspace(0.1, 0.85, len(res)))
    for r, c in zip(res, cols):
        axA.loglog(BUDGETS, r["fd"], "s--", color=c, lw=1.5, alpha=0.7)
        axA.loglog(BUDGETS, r["psr"], "o-", color=c, lw=2.2, label=r["name"])
        if np.isfinite(r["cross"]):
            axA.axvline(r["cross"], color=c, ls=":", lw=1, alpha=0.5)
    axA.set_xlabel("total shots N"); axA.set_ylabel("distance to real gradient (RMSE)")
    axA.set_title("(A) PSR rescaled (solid) vs FD best ε (dashed) per case")
    axA.legend(frameon=False, fontsize=7.5)

    Ms = [r["M"] for r in res]; crs = [r["cross"] for r in res]
    axB.semilogy(Ms, crs, "o", color="#1f77b4", ms=10)
    for r in res:
        axB.annotate(r["name"].split()[0], (r["M"], r["cross"]), fontsize=7,
                     xytext=(4, 0), textcoords="offset points", va="center")
    axB.set_xlabel("chain-rule terms  M"); axB.set_ylabel("crossover shot budget N")
    axB.set_title("(B) crossover grows with M (PSR variance ∝ M)")
    axB.set_xticks(sorted(set(Ms)))

    fig.suptitle("PSR rescaled vs FD best-ε across TFIM-like cases: PSR wins above "
                 "a crossover shot budget that\ngrows with the chain-rule term count "
                 "M (more terms → PSR pays more variance per gradient)", fontsize=9.2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "figures", "cases_crossover.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
