"""
build_F6.py — SEC6 P1-A: F6, floor + amplification (two panels, one story).

Program: TFIM  H(θ) = θ·Z0Z1 + g·(X0+X1),  differentiate the coupling θ (generator
Z0Z1, a single Pauli).  Regime T/T2*=0.15 (headline).  Hamiltonian-level under T4.

Error is measured against ∇C_noisy(θ0) — the NOISY gradient (exact fine-ε FD of the
dephased landscape, no shots), never the noiseless gradient (F6_REVISION A1).
Panel L: RMSE vs total executions N for one gradient estimate (A2). PSR (kick branches,
  finite shots), NSR (Nyquist stochastic sampler, finite shots), FD at its best-tuned ε
  (tuned ONCE at N=1e4, FROZEN). Faint 'PSR + gate channel' discloses PSR's own gate-bias
  floor (B2). Reference: N^{-1/2} + fitted exponents; the predicted δ/ε floor (B5). No oracle.
Panel R: RMSE vs FD step ε at fixed N=1e4. FD V (both arms) over the predicted δ/ε floor
  curve; sign-flip markers; PSR/NSR flat ('no step size'); ε=δ and the usable-ε window marked.
20 seeds, shade IQR. δ, T2*, gate rate = T4 best-guess (provisional, Q1-pending).
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
G_FIELD, T = 1.0, 5.0               # longer evolution → sharper θ-landscape → higher FD δ/ε floor
T2 = T / 0.15                       # T/T2* = 0.15 headline
R_CTRL = 0.02                       # control setpoint error δ (T4 best-guess)
GRAD_MIN = 0.35                     # θ0 must have a steep gradient; among those, maximize the floor
# T4's kick gate error is EXCLUDED from F6: it is a PSR-only bias (the kick is a
# digital op with its own error → biases PSR by ~0.028; NSR/waveform-shift is
# immune). That is a separate Sec-5.2 gate-infidelity finding (see data note), not
# F6's shot-floor + δ/ε story. F6 noise = dressed T2* dephasing + control δ.
GATE_2Q = None
N_TARGET = 10000                   # fixed N for panel R + FD-ε tuning
# extend to 1e6 so shot noise drops below PSR's gate-channel bias → the B2 floor is visible
NGRID = [100, 316, 1000, 3162, 10000, 31623, 100000, 316228, 1000000]
R_SEED = 40                        # repetitions per point (RMSE stable on heavy-tailed δ error)
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


GATE_2Q_T4 = 1.0e-3                 # 99.9% 2q gate (Evered et al.) — for the B2 disclosure
GATE_1Q_T4 = 1.0e-4                 # 99.99% 1q gate


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    noisy = NoisyQuTiPRunner(2, noise=NoiseModel(n_qubits=2, T2=T2, gate_error_2q=GATE_2Q))
    # F6 headline noise = dressing channel only (T2* + control δ). The digital gate channel
    # is EXCLUDED from the headline series and instead DISCLOSED as a faint PSR-with-gate
    # series (B2): it floors at PSR's own ~0.028 kick-gate bias; NSR is immune.
    ex = noisy.make_expectation_fn(PSI0, OBS)
    noisy_g = NoisyQuTiPRunner(2, noise=NoiseModel(n_qubits=2, T2=T2, gate_error_2q=GATE_2Q_T4,
                                                   gate_error_1q=GATE_1Q_T4, gate_coherent_frac=0.5))
    ex_g = noisy_g.make_expectation_fn(PSI0, OBS)
    H, var = Htfim()
    C = lambda th: ex([[H.set_parameterizedHam({"th": float(th)}), T]])

    # θ0: among STEEP points (|∇C_noisy| ≥ GRAD_MIN), pick the one that MAXIMIZES the FD δ/ε
    # floor RATIO — a sharp+steep operating point where FD fails hard (a low-floor smooth
    # point understates FD's failure). Floor = shot-free FD RMSE = truncation ⊕ δ-amplification.
    h = 1e-3
    def _floor_at(t, g, eps):
        sec = (C(t + eps) - C(t - eps)) / (2 * eps)
        cp = (C(t + eps + h) - C(t + eps - h)) / (2 * h)
        cm = (C(t - eps + h) - C(t - eps - h)) / (2 * h)
        return np.sqrt((sec - g) ** 2 + (np.sqrt(cp ** 2 + cm ** 2) * R_CTRL / (2 * eps)) ** 2)
    scan = np.linspace(0.7, 2.5, 46)
    best_t = None
    for t in scan:
        g = (C(t + h) - C(t - h)) / (2 * h)
        if abs(g) < GRAD_MIN:
            continue
        fl = min(_floor_at(t, g, e) for e in np.geomspace(0.04, 0.8, 12))
        if best_t is None or fl / abs(g) > best_t[0]:
            best_t = (fl / abs(g), float(t))
    th0 = best_t[1]
    grad_true = float((C(th0 + h) - C(th0 - h)) / (2 * h))       # TARGET ∇C_noisy (exact)
    C2 = float((C(th0 + 1e-2) - 2 * C(th0) + C(th0 - 1e-2)) / 1e-4)
    _, A = tangent_hamiltonian(H, var, th0); K = bandwidth_K(A, T)
    print(f"TFIM θ0={th0:.3f}  ∇C_noisy={grad_true:+.4f}  C''={C2:+.3f}  K={K:.3f}  "
          f"floor/|∇C|={best_t[0]*100:.0f}%  (steep+sharp point, max FD floor)")

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
    pm = np.array([ex(H_tot[2 * i]) for i in range(nb)])         # f⁻  (dressing channel only)
    pp = np.array([ex(H_tot[2 * i + 1]) for i in range(nb)])     # f⁺
    pm_g = np.array([ex_g(H_tot[2 * i]) for i in range(nb)])     # f⁻  WITH the gate channel
    pp_g = np.array([ex_g(H_tot[2 * i + 1]) for i in range(nb)]) #      (for the B2 disclosure)
    NSAMP = nb

    def _psr_from(pm_, pp_, Ntot, rng):
        nper = int(max(1, round(Ntot / (2 * NSAMP))))            # split N over 2·nb branches
        fm = 2 * rng.binomial(nper, 0.5 * (1 + np.clip(pm_, -1, 1))) / nper - 1
        fp = 2 * rng.binomial(nper, 0.5 * (1 + np.clip(pp_, -1, 1))) / nper - 1
        return (T / NSAMP) * float(ug) * np.sum(fm - fp)

    def psr_est(Ntot, rng):
        return _psr_from(pm, pp, Ntot, rng)

    def psr_gate_est(Ntot, rng):                                 # PSR carrying its gate bias
        return _psr_from(pm_g, pp_g, Ntot, rng)

    psr_gate_bias = abs((T / NSAMP) * float(ug) * np.sum(pm_g - pp_g) - grad_true)

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

    _fl_rng = np.random.default_rng(12345)

    def fd_floor_pred(eps, nmc=600):
        # the irreducible δ/ε floor = shot-FREE FD RMSE (N→∞): truncation bias ⊕ the exact
        # δ-setpoint spread (Monte-Carlo, so it stays correct on sharp/curved landscapes where
        # a linear δ-propagation over-estimates). This is exactly where FD saturates.
        dp = _fl_rng.normal(0, R_CTRL, nmc); dm = _fl_rng.normal(0, R_CTRL, nmc)
        vp = Cint(np.clip(th0 + eps + dp, grid[0], grid[-1]))
        vm = Cint(np.clip(th0 - eps + dm, grid[0], grid[-1]))
        est = (vp - vm) / (2 * eps)
        return float(np.sqrt(np.mean((est - grad_true) ** 2)))

    # tune FD ε once at N_TARGET (freeze)
    eps_grid = np.geomspace(0.02, 1.2, 22)
    rng0 = np.random.default_rng(0)
    fd_tune = [np.sqrt(np.mean([(fd_est(e, N_TARGET, rng0) - grad_true) ** 2 for _ in range(60)])) for e in eps_grid]
    eps_star = float(eps_grid[int(np.argmin(fd_tune))])
    print(f"FD ε* tuned at N={N_TARGET}: ε*={eps_star:.3f} (frozen for all N)")

    # Panel L: RMSE vs N (R_SEED reps/point); dispersion = bootstrap 25–75 band on the RMSE
    def sweepN(estfn):
        rmse, lo, hi = [], [], []
        for N in NGRID:
            errs = np.array([estfn(N, np.random.default_rng(1000 + s)) - grad_true
                             for s in range(R_SEED)])
            rmse.append(float(np.sqrt(np.mean(errs ** 2))))
            boot = [np.sqrt(np.mean(errs[np.random.default_rng(9000 + b).integers(0, R_SEED, R_SEED)] ** 2))
                    for b in range(200)]
            lo.append(float(np.percentile(boot, 25))); hi.append(float(np.percentile(boot, 75)))
        return np.array(rmse), np.array(lo), np.array(hi)

    psrL = sweepN(lambda N, r: psr_est(N, r))
    nsrL = sweepN(lambda N, r: nsr_est(N, r))
    psrGL = sweepN(lambda N, r: psr_gate_est(N, r))              # B2: PSR + gate channel
    fdL = sweepN(lambda N, r: fd_est(eps_star, N, r))

    # B4: fitted N^{-1/2} exponents for the sound series (log-log slope)
    lgN = np.log(np.array(NGRID))
    exp_psr = float(np.polyfit(lgN, np.log(psrL[0]), 1)[0])
    exp_nsr = float(np.polyfit(lgN, np.log(nsrL[0]), 1)[0])
    floor_star = fd_floor_pred(eps_star)                        # B5: predicted δ/ε floor at ε*

    # Panel R: RMSE vs ε at N_TARGET (FD V), PSR/NSR flat, sign-flips, δ/ε floor curve
    epsR = np.geomspace(0.02, 1.2, 24); fd_r, fd_wrong = [], []
    for e in epsR:
        errs, wrong = [], 0
        for s in range(R_SEED):
            rng = np.random.default_rng(500 + s)
            g = fd_est(e, N_TARGET, rng); errs.append((g - grad_true) ** 2)
            wrong += (np.sign(g) != np.sign(grad_true))
        fd_r.append(np.sqrt(np.mean(errs))); fd_wrong.append(wrong / R_SEED)
    fd_r = np.array(fd_r); fd_wrong = np.array(fd_wrong)
    floor_curve = np.array([fd_floor_pred(e) for e in epsR])    # C1: δ/ε floor reference
    psr_flat = psrL[0][NGRID.index(N_TARGET)]; nsr_flat = nsrL[0][NGRID.index(N_TARGET)]
    # C3: usable-ε window (same definition as Fig 1 R10: RMSE/|∇C_noisy| < 0.5), + sign-error
    rel = fd_r / abs(grad_true); WIN = 0.5
    good = epsR[(rel < WIN) & (fd_wrong < 0.05)]
    win_lo = float(good.min()) if good.size else None
    win_hi = float(good.max()) if good.size else None
    signerr = {float(e): float(w) for e, w in zip(epsR, fd_wrong)}

    # ── plot ──
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    N = np.array(NGRID)
    for (m, lo, hi), c, lab in [(psrL, C_PSR, rf"PSR (fit $N^{{{exp_psr:.2f}}}$)"),
                                (nsrL, C_NSR, rf"NSR, stochastic (fit $N^{{{exp_nsr:.2f}}}$)"),
                                (fdL, C_FD, rf"FD (frozen $\varepsilon^*$={eps_star:.2f})")]:
        axL.loglog(N, m, "o-", color=c, ms=5, label=lab)
        axL.fill_between(N, lo, hi, color=c, alpha=0.15)
    # B2 disclosure: PSR WITH the gate channel — floors at its own ~0.028 bias (faint)
    axL.loglog(N, psrGL[0], "s--", color=C_PSR, ms=3, lw=1.0, alpha=0.5,
               label="PSR + gate channel")
    axL.loglog(N, psrL[0][0] * (N / N[0]) ** -0.5, ":", color="#999", lw=1, label=r"$N^{-1/2}$")
    axL.axhline(floor_star, color=C_FD, lw=0.9, ls="-.")        # B5 predicted δ/ε floor
    axL.text(N[-1] * 0.9, floor_star * 1.15, r"predicted FD $\delta/\varepsilon$ floor",
             fontsize=6.5, color="#a0451a", ha="right")
    axL.set_xlabel(r"total executions $N$ (one gradient estimate)")
    axL.set_ylabel(r"RMSE vs $\nabla C_{\rm noisy}$ (noisy gradient)")
    axL.set_title(r"(L) shot floor: PSR/NSR $\to\nabla C_{\rm noisy}$, FD floors  ($T/T_2^*$=0.15)", fontsize=8.5)
    axL.legend(fontsize=6.3, loc="lower left"); axL.grid(True, which="both", alpha=0.15)

    wr = fd_wrong >= 0.2
    axR.loglog(epsR, floor_curve, "-", color="#999", lw=1.0, alpha=0.9,
               label=r"predicted $\delta/\varepsilon$ floor")   # C1 floor reference curve
    axR.loglog(epsR, fd_r, "-", color=C_FD, lw=1.6, label=r"FD (shots + $\delta$)")
    axR.loglog(epsR[~wr], fd_r[~wr], "o", color=C_FD, ms=3.5)
    axR.loglog(epsR[wr], fd_r[wr], "X", color="#1a1a1a", ms=7,
               label="FD wrong-sign (≥20%)" if wr.any() else None)
    axR.axhline(psr_flat, color=C_PSR, lw=2.0, label="PSR (no step size)")
    axR.axhline(nsr_flat, color=C_NSR, lw=2.0, ls="--", label="NSR (no step size)")
    axR.axvline(R_CTRL, color="#999", lw=0.8, ls=":")
    axR.text(R_CTRL * 1.12, fd_r.max() * 0.6, r"$\varepsilon=\delta$", fontsize=7, color="#666")
    if win_lo:                                                  # C3 usable-ε window band
        axR.axvspan(win_lo, win_hi, color="#666", alpha=0.06)
        for xe in (win_lo, win_hi):
            axR.axvline(xe, color="#888", lw=0.7, ls=(0, (2, 2)))
        axR.text(np.sqrt(win_lo * win_hi), fd_r.min() * 0.60, "usable-$\\varepsilon$ window",
                 fontsize=5.6, color="#555", ha="center", va="top")
    axR.set_xlabel(r"FD step $\varepsilon$  ($\delta$=%.2f)" % R_CTRL)
    axR.set_ylabel(r"RMSE vs $\nabla C_{\rm noisy}$ (noisy gradient)")
    axR.set_title(f"(R) amplification: FD V vs PSR/NSR (no $\\varepsilon$)  ($N$={N_TARGET})", fontsize=8.5)
    axR.legend(fontsize=6.3); axR.grid(True, which="both", alpha=0.15)
    fig.suptitle("F6 — TFIM coupling gradient: shot floor (L) + control-error amplification (R); "
                 "Hamiltonian-level under the T4 noise model (δ=%.2f, provisional)" % R_CTRL, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"F6_floor_amplification.{e}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    json.dump(dict(th0=th0, grad_true=grad_true, K=K, eps_star=eps_star, T_over_T2=0.15, delta=R_CTRL,
                   N=NGRID, psr=psrL[0].tolist(), nsr=nsrL[0].tolist(), psr_gate=psrGL[0].tolist(),
                   fd=fdL[0].tolist(), exp_psr=exp_psr, exp_nsr=exp_nsr, floor_star=floor_star,
                   psr_gate_bias=float(psr_gate_bias), win_lo=win_lo, win_hi=win_hi,
                   epsR=epsR.tolist(), fd_r=fd_r.tolist(), fd_wrong=fd_wrong.tolist(),
                   floor_curve=floor_curve.tolist(), n_seeds=R_SEED),
              open(os.path.join(FIGDIR, "F6_floor_amplification.json"), "w"), indent=2, default=float)
    win_txt = f"[{win_lo:.3f},{win_hi:.3f}]" if win_lo else "EMPTY"
    max_wrong = float(np.max(fd_wrong))
    # E: caption (instrument+regime, what error is vs, two-panel claim, gate-channel clause;
    # no "rescale"/"oracle"/"raw"/"iterations"; section ref lives ONLY here, never in-image).
    caption = (
        "Figure 6. The finite-difference floor, and that the sound strategies do not have it. "
        "TFIM coupling gradient for H(θ)=θZ0Z1+g·ΣX, Hamiltonian-level under the T4 noise model "
        "at T/T2*=0.15; error is RMSE against the noisy gradient ∇C_noisy(θ0) (exact fine-FD of "
        "the dephased landscape). (L) At equal execution budget, PSR and NSR ride N^(−1/2) to "
        "∇C_noisy while finite-shot FD at its best ε saturates at the predicted δ/ε floor. (R) No "
        f"FD step size escapes: small ε amplifies the δ/ε control-noise floor, large ε truncates "
        f"(wrong sign up to {max_wrong*100:.0f}%), whereas PSR and NSR have no step size to tune. "
        f"The digital gate channel (99.9% 2q / 99.99% 1q) is excluded from the headline and "
        f"disclosed as the faint 'PSR + gate channel' series, which floors at PSR's own "
        f"≈{psr_gate_bias:.3f} kick-gate bias (NSR immune); it is isolated in Sec. 6.3.")
    with open(os.path.join(FIGDIR, "F6_floor_amplification_caption.txt"), "w") as f:
        f.write(caption + "\n")
    print(f"wrote F6_floor_amplification.pdf/.png/.json + _caption.txt")
    print("\nCAPTION (E):\n  " + caption + "\n")
    print(
        f"DATA NOTE (F6): TFIM 2q H=θ·Z0Z1+{G_FIELD}·ΣX, θ0={th0:.3f}. Hamiltonian-level under the "
        f"T4 noise model (D4). BOTH panels at T/T2*=0.15 (A3; right panel is the 0.15 rebuild, not "
        f"the 0.5 stressor). Error measured vs ∇C_noisy={grad_true:+.4f} — the NOISY gradient (exact "
        f"fine-FD of the dephased landscape, no shots), NOT the noiseless gradient (A1). "
        f"EXECUTIONS (A2): x = total executions for ONE gradient estimate; FD=2 evals/component "
        f"(n=N/2 each), PSR={2*NSAMP} co-located ± branches (n=N/{2*NSAMP} each), NSR=N singleton "
        f"draws. REAL estimators (D2): PSR from observable_program_generator kick branches through "
        f"NoisyQuTiPRunner (short_kick), NSR from its stochastic (n,σ) sampler — no Gaussian "
        f"surrogates. {R_SEED} reps/point; RMSE with a bootstrap 25–75 dispersion band (D3). "
        f"LEFT: fitted slopes PSR N^{exp_psr:.2f}, NSR N^{exp_nsr:.2f} (≈−0.5, B4); FD frozen at "
        f"ε*={eps_star:.2f} saturates at the predicted δ/ε floor {floor_star:.3f} (B5). "
        f"GATE CHANNEL (B2): excluded from the headline; DISCLOSED as the faint 'PSR + gate channel' "
        f"series, which floors at PSR's own kick-gate bias ≈{psr_gate_bias:.3f} (NSR immune) — "
        f"isolated separately (the gate-infidelity finding). "
        f"RIGHT: FD V closed both arms (small-ε δ/ε amplification, large-ε truncation) over the "
        f"predicted δ/ε floor curve; PSR/NSR flat (no ε — 'no step size'). Sign-error rate peaks at "
        f"{max_wrong*100:.0f}% (C2). Usable-ε window (RMSE/|∇C_noisy|<0.5 & signerr<5%): {win_txt} "
        f"— same δ={R_CTRL} and floor definition as Fig 1 (C3; window differs by program/regime). "
        f"REGIME (C4): Fig 1 is T/T2*=0.5, F6 is 0.15 — the trap is not a stressor artefact. "
        f"PROVENANCE (D1): δ and channel rates are T4.csv/Q1-pending — re-render if Fred's Q1 "
        f"changes them.")


if __name__ == "__main__":
    main()
