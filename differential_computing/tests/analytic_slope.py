"""
analytic_slope.py — compute PSR's dephasing-attenuation slope EXACTLY from the
ideal (noiseless) trajectory, with NO noisy simulation.  Validate vs a small-noise
finite difference, then discuss scalability (the 100-qubit question).

First-order Lindblad perturbation theory.  For dephasing D[ρ] = Σ_i Γ(Z_i ρ Z_i − ρ)
(NoiseModel T2-only → Γ = 1/(2·T2) per qubit), the leading noise-correction to the
end expectation is
    d<O>/dΓ|_0 = ∫_0^T Σ_i [ <χ_i(t)|O|χ_i(t)> − <O>(T) ] dt,
    χ_i(t) = U(T−t) Z_i U(t) |ψ0>      (ideal unitary U(t)=e^{-iHt}, H time-indep).
The gradient slope is then d/dθ of this (fine θ-difference, exact expectations):
    s(T/T2*) = (T/T2*)·(dg/dΓ|_0) / g_ideal      [resummed: λ ≈ exp(s)].

All inputs come from the ideal trajectory we already simulate for PSR.

Run:  conda run -n qec_pg python differential_computing/tests/analytic_slope.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import qutip as qp

I = qp.qeye(2); X, Z = qp.sigmax(), qp.sigmaz()


def emb(op, i, n):
    return qp.tensor([op if k == i else I for k in range(n)])


def dOdGamma(H, O, psi0, n, T, n_grid=240):
    """Exact d<O>/dΓ|_0 from the ideal trajectory (first-order Lindblad)."""
    ts = np.linspace(0.0, T, n_grid)
    dt = ts[1] - ts[0]
    # propagators e^{-iH t} on the grid (H time-independent)
    Us = [(-1j * H * t).expm() for t in ts]
    UT = Us[-1]
    psiT = UT * psi0
    OT = float(qp.expect(O, psiT).real)
    Zs = [emb(Z, i, n) for i in range(n)]
    integrand = np.zeros(len(ts))
    for k, t in enumerate(ts):
        Ut = Us[k]
        UTt = (-1j * H * (T - t)).expm()
        psit = Ut * psi0
        s = 0.0
        for Zi in Zs:
            chi = UTt * (Zi * psit)
            s += float(qp.expect(O, chi).real) - OT
        integrand[k] = s
    return float(np.trapz(integrand, ts))


def grad(fn, theta, h=1e-3):
    return (fn(theta + h) - fn(theta - h)) / (2 * h)


def build(kind, theta):
    """Return (H, O, psi0, n) for a case at parameter value theta."""
    if kind == "1q-Z":
        n = 2; H = theta * emb(Z, 0, n) + emb(X, 0, n)
        return H, emb(Z, 0, n), qp.tensor([qp.basis(2, 0)] * n), n
    if kind == "2q-X":
        n = 2
        H = theta * emb(Z, 0, n) * emb(Z, 1, n) + emb(X, 0, n) + emb(X, 1, n)
        return H, emb(Z, 0, n), qp.tensor([qp.basis(2, 0)] * n), n
    if kind == "2q-ZZ":
        n = 2
        ZZ = emb(Z, 0, n) * emb(Z, 1, n)
        H = np.sin(2 * theta) * ZZ + np.sin(2 * theta) * emb(X, 0, n) \
            + np.sin(2 * theta) * emb(X, 1, n)
        return H, ZZ, qp.tensor([qp.basis(2, 0)] * n), n
    raise ValueError(kind)


def main():
    cases = [("1q-Z", 0.6, 2.0), ("2q-X", 0.5, 2.0), ("2q-ZZ", 0.7, 1.0)]
    print("Validate the ANALYTIC noise slope (from ideal trajectory) vs a "
          "small-noise finite difference.\n")
    print(f"{'case':>7}{'T':>5}{'<O> analytic dΓ':>18}{'<O> sim dΓ':>14}{'rel err':>10}")
    for kind, theta, T in cases:
        H, O, psi0, n = build(kind, theta)
        ana = dOdGamma(H, O, psi0, n, T)
        # sim finite-difference: (<O>_Γ − <O>_0)/Γ at small Γ via mesolve
        Gamma = 1e-3
        c_ops = [np.sqrt(Gamma) * emb(Z, i, n) for i in range(n)]
        rho0 = qp.ket2dm(psi0)
        OT0 = float(qp.expect(O, (-1j * H * T).expm() * psi0).real)
        rhoT = qp.mesolve(H, rho0, [0, T], c_ops=c_ops).states[-1]
        OTg = float(qp.expect(O, rhoT).real)
        sim = (OTg - OT0) / Gamma
        rel = abs(ana - sim) / (abs(sim) + 1e-9)
        print(f"{kind:>7}{T:>5.1f}{ana:>18.4f}{sim:>14.4f}{rel:>9.1%}")

    print("\nGradient attenuation slope from the analytic trajectory:")
    print(f"{'case':>7}{'s_analytic':>12}{'s_sim(T/T2*=.25)':>18}")
    for kind, theta, T in cases:
        # g_ideal and dg/dΓ via θ-derivative of <O> and of d<O>/dΓ
        def OT(th):
            H, O, psi0, n = build(kind, th)
            return float(qp.expect(O, (-1j * H * T).expm() * psi0).real)

        def dOdG(th):
            H, O, psi0, n = build(kind, th)
            return dOdGamma(H, O, psi0, n, T, n_grid=160)
        g_ideal = grad(OT, theta)
        dg_dG = grad(dOdG, theta)
        # λ ≈ 1 + Γ·(dg/dΓ)/g_ideal;  Γ = 1/(2·T2),  x = T/T2 ⇒ Γ = x/(2T)
        # s := dλ/dx = (1/(2T))·(dg/dΓ)/g_ideal
        s_ana = (dg_dG / g_ideal) / (2.0 * T)
        print(f"{kind:>7}{s_ana:>12.3f}")
    print("\n(compare s_analytic to the small-noise sim slopes from "
          "predict_attenuation.py: 1q-Z≈-0.53, 2q-X≈+0.19, 2q-ZZ≈-0.96)")


if __name__ == "__main__":
    main()
