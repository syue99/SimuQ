"""
test_short_kick.py — short_kick mode of observable_program_generator.

short_kick=True realizes the f₊ branch's −π/4 shift via the negated generator over
a π/4 duration ([−Hj, π/4]) instead of the long forward duration [Hj, 7π/4].
Exact for Pauli generators, so:
  - kick durations are both π/4 (vs π/4 and 7π/4),
  - the +1 branch generator is the negation of the −1 branch generator,
  - the NOISELESS gradient is identical to the standard generator,
  - under dephasing the short-kick gradient is closer to the noiseless truth.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sympy as sp
import qutip as qp
import pytest

from simuq import QSystem, Qubit
from observable_program_generator import observable_program_generator
from combine_gradient import combine_gradient_results
from qutip_sequential import QuTiPSequentialRunner
from noisy_qutip import NoisyQuTiPRunner
from noise_model import NoiseModel


def _build_1q():
    x = sp.Symbol("x")
    qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    H = x * q[0].Z + q[0].X
    return H, 2, qp.tensor(qp.sigmaz(), qp.qeye(2)), "x"


def _build_2q():
    x = sp.Symbol("x")
    qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    H = sp.sin(2 * x) * q[0].Z * q[1].Z + sp.sin(2 * x) * q[0].X \
        + sp.sin(2 * x) * q[1].X
    return H, 2, qp.tensor(qp.sigmaz(), qp.sigmaz()), "x"


def test_short_kick_durations_are_both_quarter_pi():
    H, n, _, var = _build_1q()
    np.random.seed(0)
    progs = observable_program_generator(H, 0.5, n_sample=1, n_repetition=1,
                                         diff_var=var, value=0.7, short_kick=True)
    for H_tot, _, _ in progs:
        for H_list in H_tot:
            kick_dur = H_list[1][1]
            assert abs(kick_dur - np.pi / 4) < 1e-12


def test_short_kick_plus_branch_generator_is_negated():
    H, n, _, var = _build_1q()
    np.random.seed(0)
    progs = observable_program_generator(H, 0.5, n_sample=1, n_repetition=1,
                                         diff_var=var, value=0.7, short_kick=True)
    for H_tot, _, _ in progs:
        # pair: index 0 = f₋ (sgn -1), index 1 = f₊ (sgn +1)
        Hj_minus = H_tot[0][1][0]   # kick Hamiltonian of f₋
        Hj_plus = H_tot[1][1][0]    # kick Hamiltonian of f₊
        diff = (Hj_minus.to_qutip_qobj() + Hj_plus.to_qutip_qobj())
        assert diff.norm() < 1e-12   # f₊ generator == −(f₋ generator)


@pytest.mark.parametrize("build", [_build_1q, _build_2q])
def test_short_kick_gradient_matches_standard_noiseless(build):
    H, n, obs, var = build()
    T, x_val = 0.5, 0.7
    runner = QuTiPSequentialRunner(n)
    expfn = runner.make_expectation_fn(runner.zero_state(), obs)

    np.random.seed(5)
    progs_std = observable_program_generator(H, T, n_sample=200, n_repetition=1,
                                             diff_var=var, value=x_val)
    np.random.seed(5)
    progs_short = observable_program_generator(H, T, n_sample=200, n_repetition=1,
                                               diff_var=var, value=x_val,
                                               short_kick=True)
    g_std = combine_gradient_results(progs_std, expfn, T)
    g_short = combine_gradient_results(progs_short, expfn, T)
    # same τ samples → identical noiseless gradient (residual ~1e-6 is sesolve's
    # integration tolerance; the long 7π/4 path accrues slightly more numerical
    # error than the short π/4 — itself consistent with the short kick being cleaner).
    assert abs(g_std - g_short) < 1e-5


def test_short_kick_helps_only_when_kick_dephases():
    # The short kick reduces error ONLY if the kick segment itself dephases
    # (kick_dephases=True, the legacy conservative model). Under the physically
    # faithful default (kick is a gate → no dressing-T2* during it), the kick
    # does not dephase, so short ≈ standard.
    H, n, obs, var = _build_2q()
    T, x_val = 0.5, 0.7

    clean = QuTiPSequentialRunner(n)
    np.random.seed(1)
    truth = combine_gradient_results(
        observable_program_generator(H, T, 400, 1, var, x_val),
        clean.make_expectation_fn(clean.zero_state(), obs), T)

    def grad(short, kick_dephases):
        noisy = NoisyQuTiPRunner(n, noise=NoiseModel(n_qubits=n, T2=2.0),
                                 kick_dephases=kick_dephases)
        nexp = noisy.make_expectation_fn(noisy.zero_state(), obs)
        np.random.seed(1)
        return combine_gradient_results(
            observable_program_generator(H, T, 400, 1, var, x_val,
                                         short_kick=short), nexp, T)

    # legacy model (kick dephases): short kick is substantially more accurate
    err_std_T = abs(grad(False, True) - truth)
    err_short_T = abs(grad(True, True) - truth)
    assert err_short_T < 0.5 * err_std_T

    # faithful model (kick = gate, no dephasing): short ≈ standard
    err_std_F = abs(grad(False, False) - truth)
    err_short_F = abs(grad(True, False) - truth)
    assert abs(err_short_F - err_std_F) < 0.2 * max(err_std_F, 1e-6)


def test_default_is_standard_kick():
    # default (short_kick omitted) keeps the Algorithm-1 7π/4 long kick
    H, n, _, var = _build_1q()
    np.random.seed(0)
    progs = observable_program_generator(H, 0.5, 1, 1, var, 0.7)
    kicks = sorted(H_list[1][1] for H_tot, _, _ in progs for H_list in H_tot)
    assert abs(kicks[0] - np.pi / 4) < 1e-9
    assert abs(kicks[-1] - 7 * np.pi / 4) < 1e-9


if __name__ == "__main__":
    import pytest as _pt
    _pt.main([__file__, "-v"])
