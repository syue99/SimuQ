"""
observable_program_generator

Generates the parameter-shift branches needed to differentiate
<O(T)> with respect to a symbolic variable in a Parametrized_Hamiltonian.

Implements Algorithm 1 from arXiv:2210.15812.

Output format:
  returnlist[j] = [H_tot_list, evaluated_ugrad, n_repetition]

  H_tot_list is a flat list of H_lists ordered in pairs per tau sample:
    index 2i   → sgn = -1  (kick duration π/4)
    index 2i+1 → sgn = +1  (kick duration 7π/4)

  Each H_list has three segments:
    [[evaluated_H, tau], [H_j, kick_duration], [evaluated_H, T - tau]]

  kick_duration = (1 + sgn * 3/4) * π   (Algorithm 1, paper eq.)
    sgn = -1  →  π/4   ≈ 0.785
    sgn = +1  →  7π/4  ≈ 5.497

Gradient formula (Algorithm 1):
  ∂L/∂v ≈ (T / n_sample) Σ_k Σ_j (∂u_j/∂v)(τ_k) · (p̃⁻_j(τ_k) − p̃⁺_j(τ_k))
"""
import numpy as np
import sympy as sp

from simuq.hamiltonian import productHamiltonian, TIHamiltonian


def _eval_ugrad(ugrad_raw, diff_var, value):
    """Return float value of gradient coefficient at diff_var=value."""
    if isinstance(ugrad_raw, sp.Expr):
        return float(ugrad_raw.subs(diff_var, value))
    return float(ugrad_raw)


def observable_program_generator(parametrized_H, T, n_sample, n_repetition,
                                  diff_var, value):
    """
    Generate parameter-shift branches for differentiating <O> w.r.t. diff_var.

    Implements Algorithm 1 of arXiv:2210.15812.

    Parameters
    ----------
    parametrized_H : Parametrized_Hamiltonian
    T              : float  — total evolution time
    n_sample       : int    — number of stochastic tau samples (b_int in paper)
    n_repetition   : int    — shot count per branch (b_obs in paper)
    diff_var       : str    — name of the sympy variable to differentiate
    value          : float  — point at which to evaluate the gradient

    Returns
    -------
    list of [H_tot_list, evaluated_ugrad, n_repetition]
        One entry per Hj term with non-zero gradient coefficient.
    """
    returnlist = []

    u_grad_dict = parametrized_H.take_diff_coef(diff_var)
    evaluated_H = parametrized_H.set_parameterizedHam({diff_var: value})
    tau_list = np.random.rand(n_sample) * T

    for Hj_tuple, ugrad_raw in u_grad_dict.items():
        evaluated_ugrad = _eval_ugrad(ugrad_raw, diff_var, value)

        # Skip terms whose coefficient doesn't depend on diff_var (zero gradient)
        if evaluated_ugrad == 0.0:
            continue

        Hj = TIHamiltonian(
            parametrized_H.sites_type,
            parametrized_H.sites_name,
            [(productHamiltonian(from_list=Hj_tuple), 1)],
        )

        H_tot_list = []
        for tau in tau_list:
            for sgn in [-1, 1]:
                # Algorithm 1: kick duration = (1 + sgn*3/4) * π
                # sgn=-1 → π/4 ≈ 0.785,  sgn=+1 → 7π/4 ≈ 5.497
                kick = (1 + sgn * 3 / 4) * np.pi
                H_tot_list.append([
                    [evaluated_H, tau],
                    [Hj, kick],
                    [evaluated_H, T - tau],
                ])

        returnlist.append([H_tot_list, evaluated_ugrad, n_repetition])

    return returnlist
