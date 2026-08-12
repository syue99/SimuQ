"""
test_nyquist_shift.py — unit tests for the Nyquist waveform-shift differentiator.

Validates the tangent construction, the bandwidth bound, and that the estimator
recovers the true gradient: deterministic Nyquist converges to a fine finite
difference, the stochastic estimator is unbiased, and on a Pauli system all
three routes (Nyquist / kick-PSR / FD) agree.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sympy as sp
import qutip as qp

from simuq import QSystem, Qubit
from qutip_sequential import QuTiPSequentialRunner
from observable_program_generator import observable_program_generator
from combine_gradient import combine_gradient_results
from nyquist_shift import (
    tangent_hamiltonian, spectral_diameter, bandwidth_K,
    nyquist_program_generator, nyquist_gradient,
)

T, X0 = 1.5, 0.7
NSTEPS = 300000


def _1q():
    x = sp.Symbol("x"); qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    return x * q[0].Z + q[0].X            # tangent = Z0


def _coupled():
    x = sp.Symbol("x"); qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    J = sp.sin(2 * x)
    return J * q[0].Z * q[1].Z + J * q[0].X + J * q[1].X


def _setup(n_qubits=2):
    runner = QuTiPSequentialRunner(n_qubits, nsteps=NSTEPS)
    psi0 = runner.zero_state()
    obs = qp.tensor(qp.sigmaz(), qp.qeye(2))
    return runner, runner.make_expectation_fn(psi0, obs)


def _fd_ref(H, expfn, x0=X0, eps=1e-4):
    def f(xv):
        return expfn([[H.set_parameterizedHam({"x": xv}), T]])
    return (f(x0 + eps) - f(x0 - eps)) / (2 * eps)


# ── tangent + bandwidth ──────────────────────────────────────────────────────

def test_tangent_is_Z0():
    B, A = tangent_hamiltonian(_1q(), "x", X0)
    Z0 = qp.tensor(qp.sigmaz(), qp.qeye(2))
    assert (A.to_qutip_qobj() - Z0).norm() < 1e-9


def test_spectral_diameter_of_Z0():
    _, A = tangent_hamiltonian(_1q(), "x", X0)
    assert abs(spectral_diameter(A) - 2.0) < 1e-9


def test_bandwidth_matches_formula():
    _, A = tangent_hamiltonian(_1q(), "x", X0)
    assert abs(bandwidth_K(A, T) - T / np.pi) < 1e-9      # (T/2π)·diam=2


# ── estimator correctness ────────────────────────────────────────────────────

def test_deterministic_converges_to_fd():
    _, expfn = _setup()
    H = _1q(); fd = _fd_ref(H, expfn)
    errs = [abs(nyquist_gradient(H, T, "x", X0, expfn, N=N)[0] - fd)
            for N in (2, 8, 32)]
    assert errs[0] > errs[1] > errs[2]                   # monotone convergence
    assert errs[2] < 1e-4


def test_deterministic_matches_kick_and_fd():
    runner, expfn = _setup()
    H = _1q(); fd = _fd_ref(H, expfn)
    g_ny = nyquist_gradient(H, T, "x", X0, expfn, N=32)[0]
    np.random.seed(0)
    progs = observable_program_generator(H, T, n_sample=256, n_repetition=1,
                                         diff_var="x", value=X0)
    g_kick = combine_gradient_results(progs, expfn, T)
    assert abs(g_ny - fd) < 1e-3                          # Nyquist: tight
    assert abs(g_kick - fd) < 1e-1                        # kick: stochastic-τ MC noise
    assert abs(g_ny - g_kick) < 1e-1                      # all three agree


def test_stochastic_unbiased():
    _, expfn = _setup()
    H = _1q(); fd = _fd_ref(H, expfn)
    ests = [nyquist_gradient(H, T, "x", X0, expfn, mode="stochastic",
                             n_sample=4000, seed=s, max_n=32)[0] for s in range(5)]
    assert abs(np.mean(ests) - fd) < 0.02                # unbiased within MC noise


def test_coupled_system_matches_fd():
    _, expfn = _setup()
    H = _coupled(); fd = _fd_ref(H, expfn)
    g_ny = nyquist_gradient(H, T, "x", X0, expfn, N=32)[0]
    assert abs(g_ny - fd) < 1e-3


def test_zero_gradient_when_no_dependence():
    x = sp.Symbol("x"); qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    H = 1.0 * q[0].Z + q[0].X                            # no x-dependence
    _, expfn = _setup()
    progs, info = nyquist_program_generator(H, T, "x", X0)
    assert info["K"] == 0.0 and progs == []


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
