"""
nyquist_shift.py — Nyquist waveform-shift differentiation for TIHamiltonian
control programs (arXiv:2207.01587, operationalized).

Unlike the kick parameter-shift rule (observable_program_generator.py), this
strategy stays INSIDE the control family: it never inserts a separate generator
exp(-iαH_j).  It shifts the control waveform along its tangent and re-runs the
SAME Hamiltonian family.  No Pauli / involution / separate-synthesizability
assumption on H_j — only that the shifted Hamiltonian remains executable.

For a time-independent control over [0, T]:

    H(θ) = H_c + Σ_j u_j(θ) H_j          (nominal B = H(θ0))
    A    = Σ_j (∂u_j/∂θ)|_{θ0} H_j        (tangent, from classical AD)
    H_s  = B + s A                        (an ordinary TIHamiltonian evolved T)

Bandwidth bound (single TI segment, ħ = 1):

    K = (T / 2π) · diam(A),   diam(A) = λ_max(A) − λ_min(A).

Paired Nyquist shift rule, shifts  s_n = (n + 1/2)/(2K):

    dJ/dθ = (2K/π) Σ_{n≥0} (-1)^n/(n+1/2)^2 · [ J(+s_n) − J(-s_n) ].

Deterministic estimator: truncate at N pairs (2N evaluations).
Stochastic estimator: draw n with prob ∝ 1/(n+1/2)^2 and σ∈{±1} uniform; the
weight 2Kπ(-1)^n σ on J(σ s_n) is unbiased for dJ/dθ (one shifted execution =
one gradient sample, exactly as the kick rule's one branch = one sample).

Both estimators return programs whose H_lists are single-segment [[H_s, T]] —
so they run through the SAME QuTiP/noisy runner and (unlike the kick) compile
with NO zone/transport/gate, just a shifted waveform on existing channels.
"""
import numpy as np
import sympy as sp

from simuq.hamiltonian import productHamiltonian, TIHamiltonian


# ── tangent + bandwidth ──────────────────────────────────────────────────────

def tangent_hamiltonian(parametrized_H, diff_var, value):
    """Return (B, A): nominal B = H(θ0) and tangent A = Σ_j (∂u_j/∂θ) H_j,
    both TIHamiltonians evaluated at diff_var = value."""
    B = parametrized_H.set_parameterizedHam({diff_var: value})
    u_grad = parametrized_H.take_diff_coef(diff_var)
    A = None
    for Hj_tuple, ugrad_raw in u_grad.items():
        if isinstance(ugrad_raw, sp.Expr):
            vj = float(ugrad_raw.subs(diff_var, value))
        else:
            vj = float(ugrad_raw)
        if vj == 0.0:
            continue
        prod = productHamiltonian(from_list=Hj_tuple)
        term = TIHamiltonian(parametrized_H.sites_type,
                             parametrized_H.sites_name, [(prod, vj)])
        A = term if A is None else (A + term)
    if A is None:                                  # no θ-dependence anywhere
        A = 0 * TIHamiltonian.identity(parametrized_H.sites_type,
                                       parametrized_H.sites_name)
    return B, A


def spectral_diameter(H_ti):
    """λ_max − λ_min of a TIHamiltonian (via its dense QuTiP operator)."""
    evals = H_ti.to_qutip_qobj().eigenenergies()
    return float(evals[-1] - evals[0])


def bandwidth_K(A, T):
    """Nyquist bandwidth for a single time-independent segment (ħ = 1)."""
    return T / (2.0 * np.pi) * spectral_diameter(A)


# ── program generation ───────────────────────────────────────────────────────

def _shifted_hlist(B, A, s, T):
    return [[B + s * A, T]]


def nyquist_program_generator(parametrized_H, T, diff_var, value,
                              N=8, mode="deterministic", n_sample=64,
                              seed=None, max_n=64):
    """Generate shifted-waveform programs for ∂J/∂diff_var at diff_var=value.

    Returns (programs, info) where
      programs = list of {"H_list": [[H_s, T]], "weight": float}
      info     = {"K": K, "A": A, "B": B, "mode": mode, "shifts": [...]}
    The gradient is combine_nyquist_results(programs, expfn) = Σ w_i · expfn(H_list_i).

    max_n bounds the largest Nyquist index used (deterministic N is also capped
    at max_n), i.e. it bounds the shift magnitude |s| ≤ (max_n+1/2)/(2K).  This
    is the Theis-trilemma knob: a finite shift budget trades exact unbiasedness
    for a bounded (∼ 2K/π·1/max_n) truncation bias, and is also what real
    amplitude/clipping limits force.  Large shifts also make the shifted H stiff
    to integrate, so keep max_n modest unless the runner uses a large nsteps.
    """
    B, A = tangent_hamiltonian(parametrized_H, diff_var, value)
    K = bandwidth_K(A, T)
    programs, shifts = [], []
    if K <= 0:                                     # zero tangent → zero gradient
        return programs, {"K": 0.0, "A": A, "B": B, "mode": mode, "shifts": []}

    if mode == "deterministic":
        for n in range(min(N, max_n)):
            s_n = (n + 0.5) / (2.0 * K)
            w = (2.0 * K / np.pi) * ((-1) ** n) / (n + 0.5) ** 2
            programs.append({"H_list": _shifted_hlist(B, A, +s_n, T), "weight": +w})
            programs.append({"H_list": _shifted_hlist(B, A, -s_n, T), "weight": -w})
            shifts += [+s_n, -s_n]
    elif mode == "stochastic":
        rng = np.random.default_rng(seed)
        ns = np.arange(max_n)
        p = 1.0 / (ns + 0.5) ** 2
        p /= p.sum()                               # ∝ 1/(n+1/2)^2 (truncated at max_n)
        draws = rng.choice(ns, size=n_sample, p=p)
        signs = rng.choice([-1.0, 1.0], size=n_sample)
        for n, sigma in zip(draws, signs):
            s_n = (n + 0.5) / (2.0 * K)
            w = 2.0 * K * np.pi * ((-1) ** int(n)) * sigma / n_sample
            programs.append({"H_list": _shifted_hlist(B, A, sigma * s_n, T), "weight": w})
            shifts.append(sigma * s_n)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    return programs, {"K": K, "A": A, "B": B, "mode": mode, "shifts": shifts}


def combine_nyquist_results(programs, expfn):
    """∂J/∂θ = Σ_i weight_i · expfn(H_list_i).  expfn maps an H_list → ⟨O⟩."""
    return float(sum(p["weight"] * expfn(p["H_list"]) for p in programs))


def nyquist_gradient(parametrized_H, T, diff_var, value, expfn,
                     N=8, mode="deterministic", n_sample=64, seed=None, max_n=64):
    """Convenience: generate + combine in one call."""
    programs, info = nyquist_program_generator(
        parametrized_H, T, diff_var, value, N=N, mode=mode,
        n_sample=n_sample, seed=seed, max_n=max_n)
    return combine_nyquist_results(programs, expfn), info
