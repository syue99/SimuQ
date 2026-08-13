"""
build_F6.py — SEC6 P1-A: F6, floor + amplification (two panels, one story).

Program: TFIM  H(θ) = θ·Z0Z1 + g·(X0+X1),  differentiate the coupling θ (generator
Z0Z1, a single Pauli).  Regime T/T2*=0.15 (headline).  Hamiltonian-level under T4.

Panel L: RMSE vs total shots N (1e2..1e5) per gradient component, target = ∇C_noisy
  (fine-ε FD of the emulated dephased landscape, exact expectation, no shots — logged).
  Curves: raw PSR (kick, finite shots), raw NSR (Nyquist stochastic sampler, finite
  shots), FD at its best-tuned ε (tuned ONCE at N=1e4, then FROZEN — documented).
  Reference: N^{-1/2}; the FD δ/ε floor.  NO oracle-FD.
Panel R: RMSE vs FD step ε at fixed N=1e4, control error δ=0.02.  FD V-shape (+ sign-
  flip X markers), PSR flat, NSR flat.
20 seeds, shade IQR.  δ, T2*, gate errors = T4 best-guess defaults (flagged).
Run: conda run -n qec_pg python differential_computing/tests/build_F6.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sympy as sp
import qutip as qp
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simuq import QSystem, Qubit
from noisy_qutip import NoisyQuTiPRunner
from noise_model import NoiseModel
from observable_program_generator import observable_program_generator
from nyquist_shift import tangent_hamiltonian, bandwidth_K

# ── T4 defaults (best-guess; see T4.csv) ──
G_FIELD, T = 1.0, 1.5
T2 = T / 0.15                       # T/T2* = 0.15 headline
R_CTRL = 0.02                       # control setpoint error δ (T4 best-guess)
# T4's kick gate error is EXCLUDED from F6: it is a PSR-only bias (the kick is a
# digital op with its own error → biases PSR by ~0.028; NSR/waveform-shift is
# immune). That is a separate Sec-5.2 gate-infidelity finding (see data note), not
# F6's shot-floor + δ/ε story. F6 noise = dressed T2* dephasing + control δ.
GATE_2Q = None
N_TARGET = 10000                   # fixed N for panel R + FD-ε tuning
NGRID = [100, 316, 1000, 3162, 10000, 31623, 100000]
R_SEED = 20
OBS = qp.tensor(qp.sigmaz(), qp.sigmaz())
PSI0 = qp.tensor(qp.basis(2, 0), qp.basis(2, 0))
FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
C_FD, C_PSR, C_NSR = "#D55E00", "#0072B2", "#009E73"


def Htfim():
    th = sp.Symbol("th"); qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    return th * q[0].Z * q[1].Z + G_FIELD * (q[0].X + q[1].X), "th"


def shots(val, n, rng):
    n = int(max(1, n))
    return 2.0 * rng.binomial(n, 0.5 * (1 + np.clip(val, -1, 1)), size=None) / n - 1.0


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    noisy = NoisyQuTiPRunner(2, noise=NoiseModel(n_qubits=2, T2=T2, gate_error_2q=GATE_2Q))
    # NOTE: GATE_2Q=None (excluded, see top); with T4's 1e-3 gate error raw PSR floors
    # at a ~0.028 kick-gate bias while NSR stays unbiased — logged as a finding.
    ex = noisy.make_expectation_fn(PSI0, OBS)
    H, var = Htfim()
    C = lambda th: ex([[H.set_parameterizedHam({"th": float(th)}), T]])

    # θ0: a point with a clear noisy gradient
    scan = np.linspace(0.2, 2.2, 60); h = 1e-3
    grads = np.array([(C(t + h) - C(t - h)) / (2 * h) for t in scan])
    th0 = float(scan[np.argmax(np.abs(grads))])
    grad_true = float((C(th0 + h) - C(th0 - h)) / (2 * h))       # TARGET ∇C_noisy (exact)
    C2 = float((C(th0 + 1e-2) - 2 * C(th0) + C(th0 - 1e-2)) / 1e-4)
    _, A = tangent_hamiltonian(H, var, th0); K = bandwidth_K(A, T)
    print(f"TFIM θ0={th0:.3f}  ∇C_noisy={grad_true:+.4f}  C''={C2:+.3f}  K={K:.3f}  (target = exact fine-FD)")

    # grid for landscape samples (FD & NSR shifts)
    s_max = 24.5 / (2 * K)
    grid = np.linspace(th0 - s_max - 0.3, th0 + s_max + 0.3, 1400)
    Cint = interp1d(grid, [C(t) for t in grid], kind="cubic")

    # PSR kick branches at θ0 (deterministic-τ pool), exact noisy values
    # short_kick=True → symmetric ±π/4 kick: branch-symmetric, cancels the Z-type
    # kick gate error (T4) to O(η²) so raw PSR stays unbiased for ∇C_noisy.
    orig = np.random.rand; np.random.rand = lambda k: (np.arange(k) + 0.5) / k
    try:
        progs = observable_program_generator(H, T, n_sample=48, n_repetition=1,
                                             diff_var=var, value=th0, short_kick=True)
    finally:
        np.random.rand = orig
    H_tot, ug, _ = progs[0]; nb = len(H_tot) // 2
    pm = np.array([ex(H_tot[2 * i]) for i in range(nb)])         # f⁻
    pp = np.array([ex(H_tot[2 * i + 1]) for i in range(nb)])     # f⁺
    NSAMP = nb

    def psr_est(Ntot, rng):
        # use ALL nb deterministic-τ pool samples (no resampling); split N over 2·nb branches
        nper = int(max(1, round(Ntot / (2 * NSAMP))))
        fm = 2 * rng.binomial(nper, 0.5 * (1 + np.clip(pm, -1, 1))) / nper - 1
        fp = 2 * rng.binomial(nper, 0.5 * (1 + np.clip(pp, -1, 1))) / nper - 1
        return (T / NSAMP) * float(ug) * np.sum(fm - fp)

    # NSR stochastic sampler: shifts n∝|w_n|, weight 2πK(-1)^nσ, 1 shot each
    MAXN = 24; ns = np.arange(MAXN); pw = 1.0 / (ns + 0.5) ** 2; pw /= pw.sum(); L1 = 2 * np.pi * K

    def nsr_est(Ntot, rng):
        n = rng.choice(ns, size=int(Ntot), p=pw); sig = rng.choice([-1.0, 1.0], size=int(Ntot))
        sft = sig * (n + 0.5) / (2 * K)
        val = Cint(np.clip(th0 + sft, grid[0], grid[-1]))
        sh = 2 * rng.binomial(1, 0.5 * (1 + np.clip(val, -1, 1))) - 1
        return float(np.mean(L1 * ((-1.0) ** n) * sig * sh))

    def fd_est(eps, Ntot, rng):
        nper = Ntot // 2
        dp, dm = rng.normal(0, R_CTRL), rng.normal(0, R_CTRL)
        vp = Cint(np.clip(th0 + eps + dp, grid[0], grid[-1])); vm = Cint(np.clip(th0 - eps + dm, grid[0], grid[-1]))
        fp = 2 * rng.binomial(nper, 0.5 * (1 + np.clip(vp, -1, 1))) / nper - 1
        fm = 2 * rng.binomial(nper, 0.5 * (1 + np.clip(vm, -1, 1))) / nper - 1
        return (fp - fm) / (2 * eps)

    # tune FD ε once at N_TARGET (freeze)
    eps_grid = np.geomspace(0.02, 1.2, 22)
    rng0 = np.random.default_rng(0)
    fd_tune = [np.sqrt(np.mean([(fd_est(e, N_TARGET, rng0) - grad_true) ** 2 for _ in range(60)])) for e in eps_grid]
    eps_star = float(eps_grid[int(np.argmin(fd_tune))])
    print(f"FD ε* tuned at N={N_TARGET}: ε*={eps_star:.3f} (frozen for all N)")

    # Panel L: RMSE vs N, 20 seeds, median + IQR
    def sweepN(estfn):
        med, lo, hi = [], [], []
        for N in NGRID:
            errs = []
            for s in range(R_SEED):
                rng = np.random.default_rng(1000 + s)
                errs.append(abs(estfn(N, rng) - grad_true))
            q = np.percentile(errs, [25, 50, 75]); med.append(q[1]); lo.append(q[0]); hi.append(q[2])
        return np.array(med), np.array(lo), np.array(hi)

    psrL = sweepN(lambda N, r: psr_est(N, r))
    nsrL = sweepN(lambda N, r: nsr_est(N, r))
    fdL = sweepN(lambda N, r: fd_est(eps_star, N, r))

    # Panel R: RMSE vs ε at N_TARGET (FD V), PSR/NSR flat, sign-flips
    epsR = np.geomspace(0.02, 1.2, 24); fd_r, fd_wrong = [], []
    for e in epsR:
        errs, wrong = [], 0
        for s in range(R_SEED):
            rng = np.random.default_rng(500 + s)
            g = fd_est(e, N_TARGET, rng); errs.append((g - grad_true) ** 2)
            wrong += (np.sign(g) != np.sign(grad_true))
        fd_r.append(np.sqrt(np.mean(errs))); fd_wrong.append(wrong / R_SEED)
    fd_r = np.array(fd_r); fd_wrong = np.array(fd_wrong)
    psr_flat = psrL[0][NGRID.index(N_TARGET)]; nsr_flat = nsrL[0][NGRID.index(N_TARGET)]

    # ── plot ──
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    N = np.array(NGRID)
    for (m, lo, hi), c, lab in [(psrL, C_PSR, "raw PSR"), (nsrL, C_NSR, "raw NSR (stochastic)"),
                                (fdL, C_FD, rf"FD (frozen $\varepsilon^*$={eps_star:.2f})")]:
        axL.loglog(N, m, "o-", color=c, ms=5, label=lab)
        axL.fill_between(N, lo, hi, color=c, alpha=0.15)
    axL.loglog(N, psrL[0][0] * (N / N[0]) ** -0.5, ":", color="#999", lw=1, label="$N^{-1/2}$")
    axL.axhline(min(fdL[0]), color=C_FD, lw=0.8, ls="-.")
    axL.text(N[-1] * 0.5, min(fdL[0]) * 1.15, r"FD $\delta/\varepsilon$ floor", fontsize=7, color="#a0451a", ha="right")
    axL.set_xlabel("total shots $N$ / component"); axL.set_ylabel(r"RMSE vs $\nabla C_{\rm noisy}$")
    axL.set_title(f"(L) floor: PSR/NSR → $\\nabla C_{{\\rm noisy}}$, FD floors  ($T/T_2^*$=0.15)", fontsize=8.5)
    axL.legend(fontsize=7); axL.grid(True, which="both", alpha=0.15)

    # sign-flip marker at ≥20% wrong-sign (marginal here: this θ0 is smooth+steep, so
    # δ/ε amplifies RMSE without flipping sign; flips are dramatic only on sharp features)
    wr = fd_wrong >= 0.2
    axR.loglog(epsR, fd_r, "-", color=C_FD, lw=1.6, label="FD (shots + δ)")
    axR.loglog(epsR[~wr], fd_r[~wr], "o", color=C_FD, ms=3.5)
    axR.loglog(epsR[wr], fd_r[wr], "X", color="#1a1a1a", ms=7,
               label="FD ≥20% wrong-sign" if wr.any() else None)
    axR.axhline(psr_flat, color=C_PSR, lw=2.0, label="raw PSR (flat)")
    axR.axhline(nsr_flat, color=C_NSR, lw=2.0, ls="--", label="raw NSR (flat)")
    axR.axvline(R_CTRL, color="#999", lw=0.8, ls=":")
    axR.text(R_CTRL * 1.1, fd_r.max() * 0.6, r"$\varepsilon=\delta$", fontsize=7, color="#666")
    axR.set_xlabel(r"FD step $\varepsilon$  ($\delta$=%.2f)" % R_CTRL); axR.set_ylabel(r"RMSE vs $\nabla C_{\rm noisy}$")
    axR.set_title(f"(R) amplification: FD V-shape vs PSR/NSR flat  ($N$={N_TARGET})", fontsize=8.5)
    axR.legend(fontsize=7); axR.grid(True, which="both", alpha=0.15)
    fig.suptitle("F6 — TFIM coupling gradient: shot floor (L) + control-error amplification (R); "
                 "Hamiltonian-level under T4 (δ=%.2f best-guess)" % R_CTRL, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"F6_floor_amplification.{e}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    json.dump(dict(th0=th0, grad_true=grad_true, K=K, eps_star=eps_star, T_over_T2=0.15, delta=R_CTRL,
                   N=NGRID, psr=psrL[0].tolist(), nsr=nsrL[0].tolist(), fd=fdL[0].tolist(),
                   epsR=epsR.tolist(), fd_r=fd_r.tolist(), fd_wrong=fd_wrong.tolist()),
              open(os.path.join(FIGDIR, "F6_floor_amplification.json"), "w"), indent=2, default=float)
    print(f"wrote F6_floor_amplification.pdf/.png/.json")
    print(f"DATA NOTE: TFIM 2q (θ·Z0Z1 + {G_FIELD}·ΣX), θ0={th0:.3f}, target ∇C_noisy="
          f"{grad_true:+.4f} (exact fine-FD of dephased landscape, no shots); 20 seeds median+IQR. "
          f"Noise = dressed T2* (T/T2*=0.15) + control δ={R_CTRL} (T4 best-guess). "
          f"FINDING: T4's kick gate error (1e-3) is EXCLUDED — it biases raw PSR by ~0.028 (the kick "
          f"is a digital op with its own error) while NSR is immune; separate Sec-5.2 gate-infidelity "
          f"point, not F6's shot/δ-ε story. PSR uses short_kick (symmetric).")


if __name__ == "__main__":
    main()
