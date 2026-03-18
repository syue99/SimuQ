"""
combine_gradient_results: assembles the gradient estimate from the output of
observable_program_generator() and an expectation function.

Implements Algorithm 1 of arXiv:2210.15812:
  ∂̃L/∂v = (T / b_int) Σ_k Σ_j (∂u_j/∂v)(τ_k) · (p̃⁻_j(τ_k) − p̃⁺_j(τ_k))

  Note: the paper formula is (p̃⁻ − p̃⁺), i.e. f_minus - f_plus.
  In H_tot_list: even indices (2i) are sgn=-1 (p̃⁻), odd (2i+1) are sgn=+1 (p̃⁺).

programs[j] = [H_tot_list, evaluated_ugrad, n_repetition]
  H_tot_list: flat list of H_lists in pairs (sgn=-1, sgn=+1) per tau sample
  evaluated_ugrad: float — (d u_j / d theta_l) evaluated at the given parameter value
  n_repetition: int — unused for the QuTiP path (shots only matter on real device)

expfn: callable  H_list -> float  (e.g. from QuTiPSequentialRunner.make_expectation_fn)
T:     float — total evolution time
"""


def combine_gradient_results(programs, expfn, T):
    """
    Compute gradient of an observable expectation w.r.t. a parameter.

    Implements the estimator from Algorithm 1 of arXiv:2210.15812:
      ∂̃L/∂v = (T / b_int) Σ_k Σ_j (∂u_j/∂v)(τ_k) · (p̃⁻_j − p̃⁺_j)

    Returns a float: the gradient estimate.
    """
    grad = 0.0
    for H_tot_list, ugrad, _n_rep in programs:
        # H_tot_list pairs: index 2i = sgn=-1 (p̃⁻),  2i+1 = sgn=+1 (p̃⁺)
        n_sample = len(H_tot_list) // 2
        if n_sample == 0:
            continue
        diff_sum = 0.0
        for i in range(n_sample):
            H_list_minus = H_tot_list[2 * i]       # sgn = -1  →  p̃⁻
            H_list_plus  = H_tot_list[2 * i + 1]   # sgn = +1  →  p̃⁺
            f_minus = expfn(H_list_minus)
            f_plus  = expfn(H_list_plus)
            # Paper: (p̃⁻ − p̃⁺)
            diff_sum += float(ugrad) * (f_minus - f_plus)
        # Monte Carlo average over tau samples, scaled by T
        grad += T / n_sample * diff_sum
    return grad
