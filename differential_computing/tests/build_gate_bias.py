"""
build_gate_bias.py — SEC6_FOLLOWUP C3: the PSR kick-gate bias, measured.

The 0.028 finding of F6 (excluded there because F6 = dephasing + δ) is placed here as a
Sec-6.3 entry of the strategies' COMPLEMENTARY failure modes: raw PSR's kick is a digital
op and pays a gate-infidelity price; NSR (a waveform shift, no inserted op) is immune to
it and instead pays the certificate/bandwidth scale.

We measure, on the F6 TFIM-2q program (θ·Z0Z1 + g·ΣX, differentiate θ), the raw-PSR bias
  bias(ε_gate) = PSR_exact_branches(gate ON) − ∇C_noisy(gate OFF)
at gate rates {0.5, 1, 2}×(T4 2q rate = 1e-3), for TWO kick variants:
  - standard kick and short (symmetric ±π/4) kick.
FINDING (measured): both kick variants give an IDENTICAL bias, because in the T4 model the
kick gate error is a FIXED post-kick Z-channel on the kicked qubits (apply_gate_error),
applied the same way regardless of kick shaping — so a symmetric kick does NOT echo it
away.  The ~0.028 digital price is intrinsic to inserting the kick op; NSR (no inserted
op) pays none of it.  The bias scales like √ε_gate (coherent-dominated at these rates).
Error bar = shot-noise std of the finite-shot PSR estimate at the F-loop budget N=1e4.
NSR bias ≡ 0 (no inserted op) is drawn as the reference line.
δ, T2*, gate coherent-fraction = T4 best-guess (flagged). NOT the F-loop; a 3-point cell.
Run: conda run -n qec_pg python differential_computing/tests/build_gate_bias.py
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

G_FIELD, T = 1.0, 1.5
T2 = T / 0.15
GATE_2Q_T4 = 1.0e-3            # T4 best-guess 2q gate infidelity (Evered et al. 2026)
COH_FRAC = 0.5                 # T4 best-guess coherent fraction
RATES = [0.5 * GATE_2Q_T4, 1.0 * GATE_2Q_T4, 2.0 * GATE_2Q_T4]
N_BUDGET = 10000              # F-loop per-gradient budget → shot-noise error bar
N_POOL = 48                   # deterministic-τ pool size
SEEDS = 200                   # for the shot-noise std of the estimate
OBS = qp.tensor(qp.sigmaz(), qp.sigmaz())
PSI0 = qp.tensor(qp.basis(2, 0), qp.basis(2, 0))
FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
C_PSR, C_SHORT, C_NSR = "#0072B2", "#56B4E9", "#009E73"


def Htfim():
    th = sp.Symbol("th"); qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    return th * q[0].Z * q[1].Z + G_FIELD * (q[0].X + q[1].X), "th"


def psr_branches(H, var, th0, ex, short):
    """Deterministic-τ PSR branch values (f⁻, f⁺) at θ0 through the (noisy) runner ex,
    and the gradient prefactor u'·(T/nb).  ex already carries the gate-error channel."""
    orig = np.random.rand; np.random.rand = lambda k: (np.arange(k) + 0.5) / k
    try:
        progs = observable_program_generator(H, T, n_sample=N_POOL, n_repetition=1,
                                             diff_var=var, value=th0, short_kick=short)
    finally:
        np.random.rand = orig
    H_tot, ug, _ = progs[0]; nb = len(H_tot) // 2
    pm = np.array([ex(H_tot[2 * i]) for i in range(nb)])       # f⁻
    pp = np.array([ex(H_tot[2 * i + 1]) for i in range(nb)])   # f⁺
    return pm, pp, float(ug), nb


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    H, var = Htfim()

    # target ∇C_noisy: gate OFF (dephasing only), exact fine-FD, no shots
    ex0 = NoisyQuTiPRunner(2, noise=NoiseModel(n_qubits=2, T2=T2)).make_expectation_fn(PSI0, OBS)
    C0 = lambda th: ex0([[H.set_parameterizedHam({"th": float(th)}), T]])
    scan = np.linspace(0.2, 2.2, 60); h = 1e-3
    th0 = float(scan[np.argmax([abs((C0(t + h) - C0(t - h)) / (2 * h)) for t in scan])])
    grad_true = float((C0(th0 + h) - C0(th0 - h)) / (2 * h))
    print(f"TFIM θ0={th0:.3f}  ∇C_noisy(gate off)={grad_true:+.5f}")

    def psr_exact_and_std(rate, short):
        noise = NoiseModel(n_qubits=2, T2=T2, gate_error_2q=rate, gate_coherent_frac=COH_FRAC)
        ex = NoisyQuTiPRunner(2, noise=noise).make_expectation_fn(PSI0, OBS)
        pm, pp, ug, nb = psr_branches(H, var, th0, ex, short)
        g_exact = (T / nb) * ug * float(np.sum(pm - pp))       # exact-branch PSR (gate on)
        # shot-noise std at N_BUDGET: split N over 2·nb branches, binomial sampling
        nper = int(max(1, round(N_BUDGET / (2 * nb))))
        ests = []
        for s in range(SEEDS):
            rng = np.random.default_rng(7000 + s)
            fm = 2 * rng.binomial(nper, 0.5 * (1 + np.clip(pm, -1, 1))) / nper - 1
            fp = 2 * rng.binomial(nper, 0.5 * (1 + np.clip(pp, -1, 1))) / nper - 1
            ests.append((T / nb) * ug * float(np.sum(fm - fp)))
        return g_exact, g_exact - grad_true, float(np.std(ests))

    rows = {"standard": [], "short": []}
    print(f"{'rate/1e-3':>10} {'kick':>9} {'PSR_exact':>10} {'bias':>10} {'shot_std':>9}")
    for rate in RATES:
        for short, key in [(False, "standard"), (True, "short")]:
            gx, bias, sd = psr_exact_and_std(rate, short)
            rows[key].append(dict(rate=rate, psr_exact=gx, bias=bias, shot_std=sd))
            print(f"{rate/1e-3:10.2f} {key:>9} {gx:+10.5f} {bias:+10.5f} {sd:9.5f}")

    # ── figure: bias vs gate rate (raw PSR; short kick is identical → noted, not drawn) ──
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    rr = np.array([r["rate"] for r in rows["standard"]]) / 1e-3
    b = np.array([r["bias"] for r in rows["standard"]])
    e = np.array([r["shot_std"] for r in rows["standard"]])
    ax.errorbar(rr, b, yerr=e, fmt="o-", color=C_PSR, ms=5, lw=1.6, capsize=3,
                label="raw PSR (kick — digital price)")
    # √ε_gate guide through the T4-rate point (coherent-dominated)
    b_ref = next(r["bias"] for r in rows["standard"] if abs(r["rate"] - GATE_2Q_T4) < 1e-12)
    ax.plot(rr, b_ref * np.sqrt(rr), ":", color="#888", lw=1.1, label=r"$\propto\sqrt{\varepsilon_{\rm gate}}$ guide")
    ax.axhline(0, color=C_NSR, lw=2.0, ls="--", label="NSR (no inserted op) — immune")
    ax.axvline(1.0, color="#999", lw=0.8, ls=":")
    ax.text(1.03, b.min() * 0.5, "T4 rate\n(1e-3)", fontsize=6.5, color="#666")
    ax.set_xlabel(r"2q gate infidelity $\varepsilon_{\rm gate}$ (units of T4 rate $10^{-3}$)")
    ax.set_ylabel(r"raw-PSR bias  $\hat g-\nabla C_{\rm noisy}$")
    ax.set_title("C3 — PSR kick-gate bias vs NSR immunity  ($T/T_2^*$=0.15, coh-frac 0.5)", fontsize=8)
    ax.legend(fontsize=6.8, loc="lower left"); ax.grid(True, alpha=0.15)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"gate_bias.{e}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    out = dict(th0=th0, grad_true=grad_true, T_over_T2=0.15, gate_rate_T4=GATE_2Q_T4,
               coherent_frac=COH_FRAC, N_budget=N_BUDGET, rates=RATES, rows=rows)
    json.dump(out, open(os.path.join(FIGDIR, "gate_bias.json"), "w"), indent=2, default=float)
    b1 = next(r["bias"] for r in rows["standard"] if abs(r["rate"] - GATE_2Q_T4) < 1e-12)
    s1 = next(r["bias"] for r in rows["short"] if abs(r["rate"] - GATE_2Q_T4) < 1e-12)
    print("wrote gate_bias.pdf/.png/.json")
    print(f"DATA NOTE (C3): TFIM-2q θ0={th0:.3f}, ∇C_noisy={grad_true:+.4f}. Raw-PSR bias from the "
          f"kick gate error (Z-type, Evered et al.; coh-frac {COH_FRAC}); NSR immune (no inserted op). "
          f"At T4 rate 1e-3: raw-PSR bias={b1:+.4f}. Kick shaping does NOT remove it: standard "
          f"and short (symmetric) kicks give an identical bias ({b1:+.4f} vs {s1:+.4f}) because the "
          f"T4 gate error is a fixed post-kick Z-channel, not echoed by kick symmetry — the digital "
          f"price is intrinsic to inserting the op. Bias scales ~√ε_gate (coherent-dominated): "
          f"{rows['standard'][0]['bias']:+.4f}/{rows['standard'][1]['bias']:+.4f}/"
          f"{rows['standard'][2]['bias']:+.4f} at 0.5×/1×/2×. Error bars = shot std at N={N_BUDGET}. "
          f"T2*, gate rate, coh-frac = T4 best-guess (flagged).")


if __name__ == "__main__":
    main()
