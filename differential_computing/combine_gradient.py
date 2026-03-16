"""
combine_gradient_results: assembles the gradient estimate from the output of
observable_program_generator() and an expectation function.

Formula (Section 2.4 of paper):
  d f / d theta_l  =  T * E_tau [ sum_j (d u_j / d theta_l)(tau) * (f_+ - f_-) ]

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

    Returns a float: the gradient estimate.
    """
    grad = 0.0
    for H_tot_list, ugrad, _n_rep in programs:
        # H_tot_list is ordered as pairs: (sgn=-1, sgn=+1) for each tau
        n_sample = len(H_tot_list) // 2
        if n_sample == 0:
            continue
        diff_sum = 0.0
        for i in range(n_sample):
            H_list_minus = H_tot_list[2 * i]       # sgn = -1
            H_list_plus  = H_tot_list[2 * i + 1]   # sgn = +1
            f_minus = expfn(H_list_minus)
            f_plus  = expfn(H_list_plus)
            diff_sum += float(ugrad) * (f_plus - f_minus)
        # Monte Carlo average over tau samples, scaled by T
        grad += T / n_sample * diff_sum
    return grad
