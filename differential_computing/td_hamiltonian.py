"""
td_hamiltonian
--------------

Time-dependent Hamiltonian support for DiffSimuQ.

A TD Hamiltonian is a ``Parametrized_Hamiltonian`` whose coefficients are sympy
expressions in a time symbol ``t`` (and possibly tunable parameter symbols).
After substituting parameter values, each term's coefficient becomes a pure
function of ``t``.  We factor the whole Hamiltonian into

    H(t) = Σ_k f_k(t) · H_k

where each ``H_k`` is a *time-independent* ``TIHamiltonian`` and ``f_k(t)`` is a
scalar envelope. The envelope can be reproduced on the AWG as a waveform while
``H_k`` is compiled once through the usual SimuQ path.

This module provides:

- :func:`factor_td_hamiltonian` — split the TD Hamiltonian into envelope groups.
- :func:`check_dressing_collision` — detect groups whose ≥2-body terms share
  qubits (the case where independent-group compilation breaks down and we
  fall back to Trotterization).
- :func:`trotterize_td` — the Trotter fallback.
- :func:`build_channel_envelopes` — sample the per-channel waveform once each
  group has been compiled and mapped to hardware channels.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import sympy as sp

from simuq.hamiltonian import (
    Parametrized_Hamiltonian,
    TIHamiltonian,
    productHamiltonian,
)


_TOL = 1e-12


def _to_sym(symbol_or_name):
    """Accept either a sympy Symbol or a plain name and return a Symbol."""
    if isinstance(symbol_or_name, sp.Symbol):
        return symbol_or_name
    return sp.Symbol(str(symbol_or_name))


def _normalize_param_dict(param_dict):
    """Allow callers to pass either {Symbol: val} or {'name': val}."""
    if param_dict is None:
        return {}
    return {str(k): v for k, v in param_dict.items()}


def factor_td_hamiltonian(
    H_td: Parametrized_Hamiltonian,
    t_sym,
    param_dict: dict | None = None,
):
    """Group TD Hamiltonian terms by their time envelope.

    Parameters
    ----------
    H_td : Parametrized_Hamiltonian
        Coefficients may mix a time symbol ``t_sym`` with parameter symbols.
    t_sym : sympy.Symbol or str
        The symbol representing time.
    param_dict : dict, optional
        Values for non-time parameters, e.g. ``{x: 1.0}``. After substitution,
        every coefficient is a pure function of ``t_sym`` (or a constant).

    Returns
    -------
    list[(envelope, TIHamiltonian)]
        ``envelope`` is a sympy expression in ``t_sym`` — possibly the constant
        ``1`` for genuinely time-independent terms. Each ``TIHamiltonian``
        shares sites with the original ``H_td``.

    Notes
    -----
    Factoring is done term-by-term. A coefficient that is a *sum* of distinct
    time functions (e.g. ``2*sin(t) + 3*cos(t)``) is split into two groups.
    A coefficient that is a *product* of a constant and a time function
    (e.g. ``2*sin(t)``) becomes ``weight=2, envelope=sin(t)``.
    """
    t_sym = _to_sym(t_sym)
    param_dict = _normalize_param_dict(param_dict)

    # Substitute parameter values. set_parameterizedHam returns either a
    # TIHamiltonian (all params resolved to floats) or a Parametrized_Hamiltonian
    # (some symbols remain — either t_sym or un-substituted params).
    if param_dict:
        H_sub = H_td.set_parameterizedHam(param_dict)
    else:
        H_sub = H_td

    # envelope_key (str) → {h_tuple: accumulated_weight}
    groups: Dict[str, Dict[tuple, object]] = {}
    # envelope_key (str) → sympy expression (for reconstruction)
    envelope_exprs: Dict[str, sp.Expr] = {}

    for h_prod, coef in H_sub.ham:
        htup = h_prod.to_tuple()

        if isinstance(coef, sp.Expr):
            additive_terms = sp.Add.make_args(sp.expand(coef))
        else:
            additive_terms = (coef,)

        for term in additive_terms:
            if isinstance(term, sp.Expr):
                weight, envelope = term.as_independent(t_sym, as_Add=False)
                # Drop residual free parameter symbols only if truly none
                # depend on t; keep the weight symbolic otherwise.
            else:
                weight = term
                envelope = sp.Integer(1)

            # Skip zero-weight terms.
            if isinstance(weight, sp.Expr):
                if weight == 0:
                    continue
            else:
                if abs(weight) < _TOL:
                    continue

            env_key = sp.sympify(envelope)
            env_str = sp.srepr(env_key)

            if env_str not in groups:
                groups[env_str] = {}
                envelope_exprs[env_str] = env_key

            if htup in groups[env_str]:
                groups[env_str][htup] = groups[env_str][htup] + weight
            else:
                groups[env_str][htup] = weight

    results = []
    for env_str, terms_dict in groups.items():
        ham = []
        for htup, w in terms_dict.items():
            if isinstance(w, sp.Expr):
                # Try to collapse to a float if purely numeric.
                try:
                    w_f = float(w)
                    if abs(w_f) < _TOL:
                        continue
                    w = w_f
                except (TypeError, ValueError):
                    pass  # keep symbolic
            else:
                if abs(w) < _TOL:
                    continue
            ham.append((productHamiltonian(from_list=htup), w))

        if not ham:
            continue

        ti = TIHamiltonian(H_sub.sites_type, H_sub.sites_name, ham)
        results.append((envelope_exprs[env_str], ti))

    return results


def _active_qubits(ti: TIHamiltonian) -> set:
    """Qubits that appear in a ≥2-body term of ``ti``."""
    active = set()
    for h_prod, _c in ti.ham:
        qubits = [i for i in h_prod.keys() if h_prod[i] != ""]
        if len(qubits) >= 2:
            active.update(qubits)
    return active


def check_dressing_collision(groups: Sequence[Tuple[sp.Expr, TIHamiltonian]]) -> bool:
    """Return True if any pair of groups shares a qubit in their ≥2-body terms.

    Dressing and ZZ compilation work per-group: each group gets its own
    ``sol_gvars`` layout, and qubits not participating in that group are parked
    in the idle zone. If two groups both want to dress/ZZ the same qubit at
    overlapping times, their pulses can't be played independently and we have
    to Trotterize.
    """
    active_sets = [_active_qubits(ti) for _env, ti in groups]
    for i in range(len(active_sets)):
        for j in range(i + 1, len(active_sets)):
            if active_sets[i] & active_sets[j]:
                return True
    return False


def trotterize_td(
    H_td: Parametrized_Hamiltonian,
    t_sym,
    T: float,
    n_steps: int,
    param_dict: dict | None = None,
):
    """Discretize ``[0, T]`` into ``n_steps`` midpoint slabs.

    Each slab evaluates the envelopes at its midpoint ``τ_i = T * (i + 0.5)/n``
    and emits a :class:`TIHamiltonian` whose coefficients are the substituted
    numeric values. The returned list can feed the normal TI compilation path:

        [[H_slab_0, dt], [H_slab_1, dt], ..., [H_slab_{n-1}, dt]]

    where ``dt = T / n_steps``.
    """
    t_sym = _to_sym(t_sym)
    param_dict = _normalize_param_dict(param_dict)

    # Substitute params once; time substitution happens inside the loop.
    if param_dict:
        H_sub = H_td.set_parameterizedHam(param_dict)
    else:
        H_sub = H_td

    dt = float(T) / int(n_steps)
    segments = []
    for i in range(int(n_steps)):
        tau = (i + 0.5) * dt
        ham = []
        for h_prod, coef in H_sub.ham:
            if isinstance(coef, sp.Expr):
                val = coef.subs(t_sym, tau)
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Coefficient {coef} did not reduce to a float after "
                        f"substituting t={tau}; residual free symbols remain."
                    )
            else:
                val = float(coef)
            if abs(val) < _TOL:
                continue
            ham.append((productHamiltonian(from_list=h_prod.to_tuple()), val))
        ti = TIHamiltonian(H_sub.sites_type, H_sub.sites_name, ham)
        segments.append([ti, dt])
    return segments


def build_channel_envelopes(
    groups_with_channels: Sequence[Tuple[sp.Expr, Dict[int, float]]],
    t_sym,
    T: float,
    sample_rate: float,
) -> Dict[int, np.ndarray]:
    """Sample per-channel AWG waveforms from (envelope, channel-weights) pairs.

    Parameters
    ----------
    groups_with_channels : sequence of (envelope_expr, {channel_idx: weight})
        One entry per TD envelope group. ``weight`` is the solver's gain for
        that group on that channel (e.g., a Rabi amplitude, a detuning value).
        When two groups share a channel, their samples are summed.
    t_sym : sympy.Symbol or str
    T : float
        Total duration in the same units used by the envelope.
    sample_rate : float
        Samples per unit time. ``n_samples = int(T * sample_rate)``.

    Returns
    -------
    dict[int, np.ndarray]
        ``{channel_idx: waveform[n_samples]}``.
    """
    t_sym = _to_sym(t_sym)
    n_samples = int(round(float(T) * float(sample_rate)))
    ts = np.linspace(0.0, float(T), n_samples, endpoint=False)

    waveforms: Dict[int, np.ndarray] = {}
    for env_expr, channel_weights in groups_with_channels:
        if isinstance(env_expr, sp.Expr):
            f = sp.lambdify(t_sym, env_expr, modules="numpy")
            samples = np.asarray(f(ts), dtype=float)
            # lambdify returns a scalar when env_expr has no free symbols.
            if samples.ndim == 0:
                samples = np.full(n_samples, float(samples))
        else:
            samples = np.full(n_samples, float(env_expr))

        for ch, w in channel_weights.items():
            contribution = float(w) * samples
            if ch in waveforms:
                waveforms[ch] = waveforms[ch] + contribution
            else:
                waveforms[ch] = contribution.copy()

    return waveforms
