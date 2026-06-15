"""
shot_sampling.py — finite-measurement (shot) noise on top of exact expectations.

The noisy QuTiP runner gives the exact (decohered) expectation ⟨O⟩ = Tr(Oρ).
On real hardware you instead get a sample mean over a finite number of shots.
This layer injects that measurement noise so FD and PSR can be compared at a
fixed measurement budget — the SAME sampler wraps both, keeping it apples-to-
apples.

Assumption: the observable has eigenvalues ±1 (e.g. Z_i, Z_iZ_j).  Then ⟨O⟩ ∈
[−1, 1] with p(+1) = (1+⟨O⟩)/2, and N shots give the unbiased estimate
2k/N − 1, k ~ Binomial(N, p₊).  General-spectrum observables are out of scope
here (the benchmark uses ±1 observables).
"""

import numpy as np


def sample_pm1_expectation(exact, n_shots, rng):
    """Finite-shot estimate of a ±1-eigenvalue observable's expectation.

    exact   : float — exact ⟨O⟩ ∈ [−1, 1]
    n_shots : int   — number of measurement shots
    rng     : np.random.Generator
    """
    e = min(1.0, max(-1.0, float(exact)))
    p_plus = 0.5 * (1.0 + e)
    k = rng.binomial(int(n_shots), p_plus)
    return 2.0 * k / n_shots - 1.0


def make_shot_expfn(exact_expfn, n_shots, rng):
    """Wrap an exact expfn(H_list)->float into a shot-noisy one.

    Each call draws an independent finite-shot estimate — i.e. each PSR branch
    (or FD evaluation) is measured with its own n_shots, as on hardware.  Plugs
    straight into combine_gradient_results in place of the exact expfn.
    """
    def expfn(H_list):
        return sample_pm1_expectation(exact_expfn(H_list), n_shots, rng)
    return expfn
