"""
test_noise_infra.py — formal tests for the QuTiP noise infrastructure.

Codifies the noise_infra_check.py validation:
  - NoiseModel collapse operators (T1/T2 + Pauli rates) are well-formed,
  - the mesolve NoisyQuTiPRunner reproduces the coherent sesolve PSR gradient,
  - PSR converges to finite difference in the noiseless limit,
  - T1/T2 and Pauli channels each shift the gradient,
  - the ±1 shot sampler is unbiased with 1/sqrt(N) variance scaling.

Expensive mesolve paths use few evaluations; the shot-sampler statistics test
hits the pure sampler (no evolution) so it stays fast.
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
from shot_sampling import sample_pm1_expectation, make_shot_expfn


# ── model builders ────────────────────────────────────────────────────────────

def _build_1q():
    x = sp.Symbol("x")
    qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    H = x * q[0].Z + q[0].X
    obs = qp.tensor(qp.sigmaz(), qp.qeye(2))
    return H, 2, obs, "x"


def _build_2q():
    x = sp.Symbol("x")
    qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    H = sp.sin(2 * x) * q[0].Z * q[1].Z + sp.sin(2 * x) * q[0].X \
        + sp.sin(2 * x) * q[1].X
    obs = qp.tensor(qp.sigmaz(), qp.sigmaz())
    return H, 2, obs, "x"


def _programs(H, var, x_val, T, n_sample=1, seed=7):
    np.random.seed(seed)
    return observable_program_generator(
        H, T, n_sample=n_sample, n_repetition=1, diff_var=var, value=x_val)


def _fd(H, var, theta, T, runner, obs, eps=1e-4):
    expfn = runner.make_expectation_fn(runner.zero_state(), obs)

    def f(th):
        return expfn([[H.set_parameterizedHam({var: th}), T]])
    return (f(theta + eps) - f(theta - eps)) / (2 * eps)


# ── NoiseModel collapse operators ─────────────────────────────────────────────

def test_noiseless_model_has_no_collapse_ops():
    nm = NoiseModel(n_qubits=2)
    assert not nm.has_noise()
    assert nm.collapse_ops() == []


def test_collapse_op_counts():
    n = 2
    assert len(NoiseModel(n, T1=10.0).collapse_ops()) == n          # σ⁻ per qubit
    assert len(NoiseModel(n, T2=10.0).collapse_ops()) == n          # Z per qubit
    assert len(NoiseModel(n, T1=10.0, T2=5.0).collapse_ops()) == 2 * n
    assert len(NoiseModel(n, pauli_rates={"X": 0.1, "Z": 0.2}).collapse_ops()) == 2 * n


def test_t2_greater_than_2t1_raises():
    with pytest.raises(ValueError):
        NoiseModel(n_qubits=1, T1=1.0, T2=3.0).collapse_ops()


def test_pauli_rate_collapse_operator_value():
    nm = NoiseModel(n_qubits=1, pauli_rates={"X": 0.25})
    c = nm.collapse_ops()
    assert len(c) == 1
    # c = sqrt(0.25) * X = 0.5 * sigmax
    assert (c[0] - 0.5 * qp.sigmax()).norm() < 1e-12


def test_bad_pauli_key_raises():
    with pytest.raises(ValueError):
        NoiseModel(n_qubits=1, pauli_rates={"W": 0.1})


# ── mesolve runner correctness ────────────────────────────────────────────────

@pytest.mark.parametrize("build", [_build_1q, _build_2q])
def test_mesolve_reproduces_sesolve(build):
    H, n, obs, var = build()
    T, x_val = 0.5, 0.7
    programs = _programs(H, var, x_val, T, n_sample=1)

    ses = QuTiPSequentialRunner(n)
    mes = NoisyQuTiPRunner(n, noise=None)
    g_ses = combine_gradient_results(
        programs, ses.make_expectation_fn(ses.zero_state(), obs), T)
    g_mes = combine_gradient_results(
        programs, mes.make_expectation_fn(mes.zero_state(), obs), T)
    assert abs(g_mes - g_ses) < 1e-5


def test_psr_converges_to_fd_1q():
    H, n, obs, var = _build_1q()
    T, x_val = 0.5, 0.7
    mes = NoisyQuTiPRunner(n, noise=None)
    expfn = mes.make_expectation_fn(mes.zero_state(), obs)
    g_fd = _fd(H, var, x_val, T, mes, obs)
    g_psr = combine_gradient_results(_programs(H, var, x_val, T, n_sample=300), expfn, T)
    assert abs(g_psr - g_fd) < 1e-2


def test_psr_equals_fd_2q_single_sample():
    # 2q has a common sin(2x) prefactor → τ-independent integrand → exact at n=1.
    H, n, obs, var = _build_2q()
    T, x_val = 0.5, 0.7
    mes = NoisyQuTiPRunner(n, noise=None)
    expfn = mes.make_expectation_fn(mes.zero_state(), obs)
    g_fd = _fd(H, var, x_val, T, mes, obs)
    g_psr = combine_gradient_results(_programs(H, var, x_val, T, n_sample=1), expfn, T)
    assert abs(g_psr - g_fd) < 1e-3


@pytest.mark.parametrize("noise", [
    NoiseModel(n_qubits=2, T1=2.0, T2=1.0),
    NoiseModel(n_qubits=2, pauli_rates={"X": 0.3, "Z": 0.3}),
])
def test_noise_shifts_gradient(noise):
    H, n, obs, var = _build_2q()
    T, x_val = 0.5, 0.7
    programs = _programs(H, var, x_val, T, n_sample=1)
    clean = NoisyQuTiPRunner(n, noise=None)
    noisy = NoisyQuTiPRunner(n, noise=noise)
    g_clean = combine_gradient_results(
        programs, clean.make_expectation_fn(clean.zero_state(), obs), T)
    g_noisy = combine_gradient_results(
        programs, noisy.make_expectation_fn(noisy.zero_state(), obs), T)
    assert abs(g_noisy - g_clean) > 1e-4


# ── shot sampling ─────────────────────────────────────────────────────────────

def test_shot_sampler_unbiased_and_scales():
    exact = 0.3
    rng = np.random.default_rng(0)
    prev_std = None
    for N in (64, 256, 1024, 4096):
        draws = [sample_pm1_expectation(exact, N, rng) for _ in range(4000)]
        mean, std = np.mean(draws), np.std(draws)
        # unbiased: mean within a few standard errors of exact
        assert abs(mean - exact) < 5 * std / np.sqrt(4000)
        # variance matches (1 - e^2)/N
        assert abs(std - np.sqrt((1 - exact ** 2) / N)) < 0.1 * std + 1e-3
        if prev_std is not None:
            assert std < prev_std
        prev_std = std


def test_shot_sampler_clamps_and_bounds():
    rng = np.random.default_rng(1)
    # out-of-range exact value is clamped; output always in [-1, 1]
    for _ in range(50):
        v = sample_pm1_expectation(1.5, 16, rng)
        assert -1.0 <= v <= 1.0
    # exact = +1 → always +1
    assert sample_pm1_expectation(1.0, 100, rng) == 1.0


def test_make_shot_expfn_composes():
    calls = {"n": 0}

    def fake_exact(H_list):
        calls["n"] += 1
        return 0.5
    rng = np.random.default_rng(2)
    shot = make_shot_expfn(fake_exact, 256, rng)
    vals = [shot([["H", 1.0]]) for _ in range(2000)]
    assert calls["n"] == 2000
    assert abs(np.mean(vals) - 0.5) < 0.02
    assert all(-1.0 <= v <= 1.0 for v in vals)


if __name__ == "__main__":
    import pytest as _pt
    _pt.main([__file__, "-v"])
