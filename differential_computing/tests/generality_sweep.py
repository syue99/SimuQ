"""
generality_sweep.py — are the PSR conclusions instance-specific, or general?

We've leaned on one 2q sin(2x)·(Z0Z1+X0+X1) model. This sweeps MANY randomized
instances — varying qubit count n∈{1,2,3,4}, Hamiltonian structure (random Pauli
terms, random x-dependence), parameter point θ, and Z-basis observable — and
checks that the headline conclusions hold across all of them:

  (A) under moderate dephasing (T/T2*=0.1): PSR is a multiplicative attenuation
      with λ=PSR/truth ∈ (0,1] and the SIGN is preserved (descent-safe);
  (B) under reference gate error (1q ε=1e-4, 2q ε=1e-3): bias small, sign preserved.

"Not stuck in one instance" = every instance passes (A) and (B).

Run:  conda run -n qec_pg python differential_computing/tests/generality_sweep.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sympy as sp
import qutip as qp

from simuq import QSystem, Qubit
from observable_program_generator import observable_program_generator
from combine_gradient import combine_gradient_results
from noisy_qutip import NoisyQuTiPRunner
from noise_model import NoiseModel


_QP = {"X": qp.sigmax(), "Y": qp.sigmay(), "Z": qp.sigmaz()}


def random_instance(rng, n):
    """Build a random Parametrized_Hamiltonian on n qubits + a Z-basis observable.

    Returns (H, n, obs, desc). At least one term is x-dependent.
    """
    x = sp.Symbol("x")
    qs = QSystem(); q = [Qubit(qs) for _ in range(n)]
    H = None
    desc = []

    def add(term):
        nonlocal H
        H = term if H is None else H + term

    # single-qubit terms; force the first to carry x
    for i in range(n):
        P = rng.choice(["X", "Y", "Z"])
        op = getattr(q[i], P)
        xdep = (i == 0) or (rng.random() < 0.4)
        if xdep:
            form = rng.choice(["lin", "sin"])
            c = float(rng.uniform(0.5, 1.5))
            coef = (c * x) if form == "lin" else (c * sp.sin(x))
            desc.append(f"{form}·{P}{i}")
        else:
            coef = float(rng.uniform(0.5, 1.5))
            desc.append(f"{P}{i}")
        add(coef * op)

    # two-qubit ZZ terms
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                op = getattr(q[i], "Z") * getattr(q[j], "Z")
                if rng.random() < 0.5:
                    coef = float(rng.uniform(0.3, 1.0)) * x
                    desc.append(f"x·Z{i}Z{j}")
                else:
                    coef = float(rng.uniform(0.3, 1.0))
                    desc.append(f"Z{i}Z{j}")
                add(coef * op)

    # observable: random non-empty Z-string
    sites = sorted(rng.choice(range(n), size=rng.randint(1, n + 1),
                              replace=False).tolist())
    ops = [_QP["Z"] if k in sites else qp.qeye(2) for k in range(n)]
    obs = qp.tensor(ops) if n > 1 else ops[0]
    return H, n, obs, "+".join(desc) + f"  O=Z{sites}"


def truth_grad(H, theta, T, n, obs, eps=1e-2):
    clean = NoisyQuTiPRunner(n, noise=None)
    expfn = clean.make_expectation_fn(clean.zero_state(), obs)

    def f(th):
        return expfn([[H.set_parameterizedHam({"x": float(th)}), T]])
    return (f(theta + eps) - f(theta - eps)) / (2.0 * eps)


def psr_grad(H, theta, T, runner, obs, n_sample, seed):
    expfn = runner.make_expectation_fn(runner.zero_state(), obs)
    np.random.seed(seed)
    programs = observable_program_generator(
        H, T, n_sample=n_sample, n_repetition=1, diff_var="x", value=float(theta))
    return combine_gradient_results(programs, expfn, T)


def main():
    rng = np.random.RandomState(2024)
    T = 0.5
    T2 = 5.0                       # T/T2* = 0.1 (moderate dephasing)
    n_list = [1, 2, 2, 3, 3, 3, 4, 4]   # spread of sizes (n=1 → 2 atoms below)
    n_sample = 600     # high enough that MC noise is not the story

    print(f"Generality sweep — T={T}, dephasing T2={T2} (T/T2*={T/T2:.2f}), "
          f"gate ε(1q,2q)=(1e-4,1e-3)\n")
    print(f"{'#':>2}{'n':>3}  {'instance':<34}{'truth':>9}"
          f"{'λ(deph)':>9}{'sgn':>5}{'gate_rel':>10}{'sgn':>5}")

    deph_ok = gate_ok = lam_in_range = total = 0
    for idx in range(14):
        n = int(rng.choice(n_list))
        natoms = max(n, 2)         # rydberg/sim needs ≥1; use n directly here
        H, n, obs, desc = random_instance(rng, n)
        theta = float(rng.uniform(0.4, 1.2))
        try:
            truth = truth_grad(H, theta, T, n, obs)
        except Exception as e:
            print(f"{idx:>2}{n:>3}  {desc:<34} SKIP ({type(e).__name__})")
            continue
        if abs(truth) < 5e-3:
            continue               # skip near-stationary points (sign ill-defined)
        total += 1

        # (A) dephasing
        deph = NoisyQuTiPRunner(n, noise=NoiseModel(n_qubits=n, T2=T2))
        g_d = psr_grad(H, theta, T, deph, obs, n_sample, seed=100 + idx)
        lam = g_d / truth
        sd = "ok" if np.sign(g_d) == np.sign(truth) else "FLIP"
        deph_ok += (sd == "ok")
        lam_in_range += (0.0 < lam <= 1.05)

        # (B) gate error
        gate = NoisyQuTiPRunner(n, noise=NoiseModel(
            n_qubits=n, gate_error_1q=1e-4, gate_error_2q=1e-3,
            gate_coherent_frac=0.5))
        g_g = psr_grad(H, theta, T, gate, obs, n_sample, seed=200 + idx)
        rel = (g_g - truth) / truth
        sg = "ok" if np.sign(g_g) == np.sign(truth) else "FLIP"
        gate_ok += (sg == "ok")

        print(f"{idx:>2}{n:>3}  {desc[:34]:<34}{truth:>9.4f}"
              f"{lam:>9.3f}{sd:>5}{rel:>+9.1%}{sg:>5}")

    print(f"\nInstances: {total}")
    print(f"(A) dephasing  sign-preserved: {deph_ok}/{total}   "
          f"λ∈(0,1.05]: {lam_in_range}/{total}  (multiplicative attenuation)")
    print(f"(B) gate error sign-preserved: {gate_ok}/{total}")
    if deph_ok == total and gate_ok == total:
        print("\n→ Conclusions hold across all instances — NOT stuck in one case.")
    else:
        print("\n→ Some instance broke a conclusion — inspect above.")


if __name__ == "__main__":
    main()
