"""
td_psr
------

Parameter-shift-rule gradient estimator for time-dependent Hamiltonians.

This is the TD analogue of ``observable_program_generator.py``:

- The PSR is still indexed term-by-term (the envelope grouping in
  ``td_hamiltonian.py`` is for *compilation*, not for gradient accounting).
- Each term's coefficient is a sympy expression ``f_j(v, t)``. Its v-derivative
  ``∂f_j/∂v`` survives as a function of ``t`` after substituting ``v = value``.
- We sample ``τ_k ~ Uniform[0, T]`` and emit one PSR branch pair per
  ``(term_j, τ_k)`` with ``ugrad = (∂f_j/∂v)(v_0, τ_k)``.
- Each branch has three segments:

      [TD-segment(0, τ_k), TI-kick-segment(H_j, (1±3/4)π), TD-segment(τ_k, T)]

  The kick freezes the envelopes (matches Algorithm 1 of arXiv:2210.15812).

TD segments use a dict form to distinguish them from TI segments::

    {"kind": "td", "H": Parametrized_Hamiltonian, "t_sym": sympy.Symbol,
     "t_start": float, "t_end": float}

TI segments keep the existing shape ``[TIHamiltonian, duration]``.
"""

from __future__ import annotations

import numpy as np
import qutip as qp
import sympy as sp

from simuq.hamiltonian import (
    Parametrized_Hamiltonian,
    TIHamiltonian,
    productHamiltonian,
)


_KICK_COEFF = 3.0 / 4.0  # duration = (1 + sgn*3/4)·π


def _to_sym(x):
    if isinstance(x, sp.Symbol):
        return x
    return sp.Symbol(str(x))


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────

def observable_program_generator_td(
    H_td: Parametrized_Hamiltonian,
    t_sym,
    T: float,
    n_sample: int,
    n_repetition: int,
    diff_var: str,
    value: float,
    param_dict: dict | None = None,
    tau_list=None,
):
    """Generate PSR branches for ``∂⟨O⟩/∂(diff_var)`` on a TD Hamiltonian.

    Parameters
    ----------
    H_td, t_sym
        Time-dependent parametrized Hamiltonian and its time symbol.
    T, n_sample, n_repetition
        Total duration, stochastic τ samples, and shots/branch (shots are
        irrelevant for the simulator path).
    diff_var, value
        Name of the symbol to differentiate w.r.t., and the point at which to
        evaluate the gradient.
    param_dict
        Optional values for any other parameter symbols in ``H_td``. The
        ``diff_var`` itself is always substituted to ``value``.
    tau_list
        Optional pre-sampled τ values (useful for deterministic tests).

    Returns
    -------
    list[[branches, ugrad_per_tau, n_repetition]]
        ``branches`` is a flat list of length ``2 * n_sample``; pairs
        ``(2i, 2i+1)`` correspond to ``(sgn=-1, sgn=+1)`` for the i-th τ sample.
        ``ugrad_per_tau[i]`` is a float: ``(∂f_j/∂v)(value, τ_i)``.
    """
    t_sym = _to_sym(t_sym)
    diff_sym = sp.Symbol(str(diff_var))

    full_subs = {diff_sym: value}
    if param_dict:
        for k, v in param_dict.items():
            full_subs[sp.Symbol(str(k))] = v

    # H with diff_var and other params substituted; t_sym may remain.
    H_subst = H_td.set_parameterizedHam(
        {str(k): v for k, v in full_subs.items()}
    )

    if tau_list is None:
        tau_list = np.random.rand(n_sample) * T
    else:
        tau_list = np.asarray(tau_list, dtype=float)
        assert len(tau_list) == n_sample

    # Group the ORIGINAL H_td by h_tuple (summing coefs) — PSR iterates terms.
    # H_td may have multiple entries per h_tuple if the user wrote redundantly;
    # cleanHam in TIHamiltonian already merges them, so H_td.ham entries have
    # unique h_tuples. We still guard with a dict.
    term_coefs: dict[tuple, sp.Expr] = {}
    for h_prod, coef in H_td.ham:
        htup = h_prod.to_tuple()
        if htup in term_coefs:
            term_coefs[htup] = term_coefs[htup] + coef
        else:
            term_coefs[htup] = coef

    returnlist = []
    for htup, coef in term_coefs.items():
        # Symbolic derivative w.r.t. diff_var.
        if isinstance(coef, sp.Expr):
            dcoef = sp.diff(coef, diff_sym)
            if dcoef == 0:
                continue
            # Substitute all params (including diff_var itself if it remained).
            dcoef = dcoef.subs(full_subs)
        else:
            continue  # numeric coef → zero derivative

        # Evaluate dcoef(t) at each τ.
        ugrad_per_tau = []
        for tau in tau_list:
            if isinstance(dcoef, sp.Expr):
                val_t = dcoef.subs(t_sym, float(tau))
                try:
                    ugrad_per_tau.append(float(val_t))
                except (TypeError, ValueError):
                    raise ValueError(
                        f"∂coef/∂{diff_var} for term {htup} did not reduce "
                        f"to a float at τ={tau}; residual symbols: "
                        f"{getattr(val_t, 'free_symbols', '?')}"
                    )
            else:
                ugrad_per_tau.append(float(dcoef))

        if all(abs(u) < 1e-14 for u in ugrad_per_tau):
            continue

        # Build Hj for the kick segment.
        Hj = TIHamiltonian(
            H_td.sites_type,
            H_td.sites_name,
            [(productHamiltonian(from_list=htup), 1)],
        )

        branches = []
        for tau in tau_list:
            tau = float(tau)
            for sgn in (-1, 1):
                kick_dur = (1 + sgn * _KICK_COEFF) * np.pi
                branches.append([
                    {
                        "kind": "td",
                        "H": H_subst,
                        "t_sym": t_sym,
                        "t_start": 0.0,
                        "t_end": tau,
                    },
                    [Hj, kick_dur],
                    {
                        "kind": "td",
                        "H": H_subst,
                        "t_sym": t_sym,
                        "t_start": tau,
                        "t_end": float(T),
                    },
                ])

        returnlist.append([branches, ugrad_per_tau, n_repetition])

    return returnlist


# ─────────────────────────────────────────────────────────────────────────────
# QuTiP sequential runner with TD-segment support
# ─────────────────────────────────────────────────────────────────────────────

def _op_from_prod(prod, n_qubits):
    from qutip import qeye, sigmax, sigmay, sigmaz, tensor
    ops = []
    for i in range(n_qubits):
        s = prod[i]  # productHamiltonian returns "" for missing keys
        if s == "":
            ops.append(qeye(2))
        elif s == "X":
            ops.append(sigmax())
        elif s == "Y":
            ops.append(sigmay())
        else:
            ops.append(sigmaz())
    return tensor(ops)


def _parametrized_to_qutip_td(H_param, t_sym, n_qubits):
    """Convert a t-dependent Parametrized_Hamiltonian to QuTiP TD H list.

    Returns either:
      - a list ``[H0_const, [op1, f1], [op2, f2], ...]`` (QuTiP TD format), or
      - a single ``Qobj`` if all coefs are constant, or
      - ``None`` if the Hamiltonian is empty.
    """
    const_op = None
    td_entries = []

    for prod, coef in H_param.ham:
        op = _op_from_prod(prod, n_qubits)

        if isinstance(coef, sp.Expr) and t_sym in coef.free_symbols:
            f = sp.lambdify(t_sym, coef, modules="numpy")
            # QuTiP expects coef_fn(t, args) → scalar.
            td_entries.append([op, _wrap_time_fn(f)])
        else:
            if isinstance(coef, sp.Expr):
                c = float(coef)
            else:
                c = float(coef)
            term = c * op
            const_op = term if const_op is None else const_op + term

    if not td_entries and const_op is None:
        return None
    if not td_entries:
        return const_op
    if const_op is None:
        return td_entries
    return [const_op] + td_entries


def _wrap_time_fn(f):
    # Closure preserves f per-call; ensure float return.
    def coef(t, args=None):
        return float(f(t))
    return coef


def run_td_sequence(H_list, psi0, n_qubits):
    """Evolve ``psi0`` through a mixed TI/TD segment sequence.

    Segment forms:
      - TI:  ``[TIHamiltonian, duration]``  — evolves for ``duration`` under H
      - TD:  ``{"kind": "td", "H": ..., "t_sym": ..., "t_start": ..., "t_end": ...}``

    The kick (TI) between TD segments freezes absolute time, which is why the
    second TD segment starts from ``t_end_1 = t_start_2 = τ`` — identical.
    """
    state = psi0
    for seg in H_list:
        if isinstance(seg, dict) and seg.get("kind") == "td":
            t0, t1 = seg["t_start"], seg["t_end"]
            if t1 <= t0:
                continue
            H_td_qp = _parametrized_to_qutip_td(seg["H"], seg["t_sym"], n_qubits)
            if H_td_qp is None:
                continue  # empty H → identity evolution
            result = qp.sesolve(H_td_qp, state, [float(t0), float(t1)])
            state = result.states[-1]
        else:
            H, duration = seg
            if duration == 0:
                continue
            H_qobj = H.to_qutip_qobj()
            result = qp.sesolve(H_qobj, state, [0.0, float(duration)])
            state = result.states[-1]
    return state


def make_td_expectation_fn(psi0, observable, n_qubits):
    """Return ``expfn(H_list) -> float`` for mixed TI/TD segment sequences."""
    def expfn(H_list):
        final = run_td_sequence(H_list, psi0, n_qubits)
        return float(qp.expect(observable, final).real)
    return expfn


# ─────────────────────────────────────────────────────────────────────────────
# Combine step (per-τ ugrad)
# ─────────────────────────────────────────────────────────────────────────────

def combine_gradient_results_td(programs, expfn, T):
    """TD counterpart of ``combine_gradient_results``.

    Accepts per-entry ``ugrad`` as a **list of floats** (one per τ sample),
    since ``∂f_j/∂v`` now varies with τ.

    Formula (Algorithm 1, generalized to TD):
      ∂⟨O⟩/∂v ≈ (T / n_sample) · Σ_j Σ_k ugrad_j(τ_k) · (p̃⁻ − p̃⁺)
    """
    grad = 0.0
    for branches, ugrad_per_tau, _n_rep in programs:
        n_sample = len(branches) // 2
        if n_sample == 0:
            continue
        if len(ugrad_per_tau) != n_sample:
            raise ValueError(
                f"ugrad list length {len(ugrad_per_tau)} != n_sample {n_sample}"
            )
        diff_sum = 0.0
        for i in range(n_sample):
            f_minus = expfn(branches[2 * i])
            f_plus = expfn(branches[2 * i + 1])
            diff_sum += float(ugrad_per_tau[i]) * (f_minus - f_plus)
        grad += T / n_sample * diff_sum
    return grad
