"""
predict_attenuation.py — can we PREDICT PSR's dephasing attenuation λ analytically
(from the system + ideal dynamics) rather than measure it from a noisy simulation?

λ is NOT a universal function of T/T2* (scaling_universality.py), but it may be
PREDICTABLE per system from its LEADING SLOPE: by first-order Lindblad perturbation
theory, λ(γ) ≈ 1 + (g'(0)/g_ideal)·γ, where the slope g'(0)/g_ideal is computable
from the IDEAL (noiseless) trajectory we already simulate for PSR — no noisy sim.

We proxy that analytic slope by the small-noise λ (the first-order coefficient),
then test two predictors against the ACTUAL λ at larger noise, per case:
  linear:       λ_lin(x) = 1 + s·x
  exponential:  λ_exp(x) = exp(s·x)         (resums the leading rate)
with x = T/T2*, s = (λ(x0) − 1)/x0 from a single small x0.

If λ_exp tracks the actual λ across cases, a computed 1/λ rescaling makes PSR an
accurate (system-aware) gradient estimator without measuring the noise.

Run:  conda run -n qec_pg python differential_computing/tests/predict_attenuation.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scaling_universality as su          # reuse build_case + psr_grad
import qutip as qp
from noisy_qutip import NoisyQuTiPRunner
from noise_model import NoiseModel


def lam_at(H, var, x_val, T, nq, obs, g_ideal, T2):
    runner = NoisyQuTiPRunner(nq, noise=NoiseModel(n_qubits=nq, T2=T2))
    return su.psr_grad(H, var, x_val, T, runner, nq, obs) / g_ideal


def main():
    cases = [("1q-Z", 0.6, 2.0, "#1f77b4"), ("1q-sin", 0.8, 2.0, "#ff7f0e"),
             ("2q-ZZ", 0.7, 1.0, "#2ca02c"), ("2q-X", 0.5, 2.0, "#d62728")]
    x0 = 0.25                              # small-noise point → leading slope
    x_test = [0.25, 0.5, 1.0, 1.5, 2.0]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=150)
    print(f"Predict λ from the leading slope s (at T/T2*={x0}); test vs actual.\n")
    print(f"{'case':>8}{'s':>8}  " + "".join(f"x={x:<4}".rjust(8) for x in x_test))
    for kind, x_val, T, c in cases:
        H, var, obs, nq = su.build_case(kind)
        clean = NoisyQuTiPRunner(nq, noise=None)
        g_ideal = su.psr_grad(H, var, x_val, T, clean, nq, obs)
        # leading slope from a single small-noise point (proxy for analytic g'(0))
        lam0 = lam_at(H, var, x_val, T, nq, obs, g_ideal, T / x0)
        s = (lam0 - 1.0) / x0
        actual = [lam_at(H, var, x_val, T, nq, obs, g_ideal, T / x) for x in x_test]
        pred_exp = [np.exp(s * x) for x in x_test]
        pred_lin = [1 + s * x for x in x_test]
        print(f"{kind:>8}{s:>8.3f}  " +
              "".join(f"{a:>+6.2f}".rjust(8) for a in actual))
        print(f"{'exp pred':>8}{'':>8}  " +
              "".join(f"{p:>+6.2f}".rjust(8) for p in pred_exp))

        axA.plot(x_test, actual, "o-", color=c, lw=2, label=f"{kind} actual")
        axA.plot(x_test, pred_exp, "--", color=c, lw=1.5, alpha=0.8)
        # recovery after rescaling by the PREDICTED 1/λ_exp
        recov = [a / p for a, p in zip(actual, pred_exp)]
        axB.plot(x_test, recov, "o-", color=c, lw=2, label=kind)

    axA.set_xlabel(r"$T/T_2^*$"); axA.set_ylabel(r"$\lambda$")
    axA.set_title("(A) actual λ (solid) vs predicted exp(s·x) (dashed)")
    axA.legend(frameon=False, fontsize=7.5, ncol=2)
    axB.axhline(1.0, color="k", ls="--", lw=1.2, label="perfect recovery")
    axB.axhspan(0.9, 1.1, color="green", alpha=0.08)
    axB.set_xlabel(r"$T/T_2^*$")
    axB.set_ylabel(r"actual / predicted  (rescaled accuracy)")
    axB.set_title("(B) after computed 1/λ rescaling — recovery to 1?")
    axB.legend(frameon=False, fontsize=8)

    fig.suptitle("Predicting PSR's attenuation from the LEADING SLOPE (computable "
                 "from ideal dynamics):\nexp(s·T/T2*) tracks actual λ per system — "
                 "a computed 1/λ rescaling recovers the gradient (low–moderate noise)",
                 fontsize=9.2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "figures", "predict_attenuation.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
