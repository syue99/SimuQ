"""
Unit tests for differential_computing.td_psr.

Validates the TD parameter-shift gradient against finite differences on a
time-dependent Hamiltonian example from the plan:

    H(v, t) = v·sin(t)·X_0 + cos(t)·Z_0

and a harder "parameter inside the envelope" case:

    H(v, t) = sin(v·t)·X_0 + cos(t)·Z_0

The PSR formula (TD, Algorithm 1 of arXiv:2210.15812, generalized):

    ∂⟨O⟩/∂v ≈ (T / n_sample) · Σ_j Σ_k (∂f_j/∂v)(v_0, τ_k) · (p̃⁻_j − p̃⁺_j)

Tests use a seeded RNG and large n_sample to tame Monte Carlo variance.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import qutip as qp
import sympy as sp

from simuq import QSystem, Qubit
from simuq.hamiltonian import TIHamiltonian

from td_psr import (
    _parametrized_to_qutip_td,
    combine_gradient_results_td,
    make_td_expectation_fn,
    observable_program_generator_td,
    run_td_sequence,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fd_gradient(H_td, t_sym, v_sym, v0, T, psi0, observable, n_qubits, eps=1e-4):
    def obs_at(v_val):
        H_subst = H_td.set_parameterizedHam({str(v_sym): v_val})
        seg = [{
            "kind": "td",
            "H": H_subst,
            "t_sym": t_sym,
            "t_start": 0.0,
            "t_end": T,
        }]
        state = run_td_sequence(seg, psi0, n_qubits)
        return float(qp.expect(observable, state).real)
    return (obs_at(v0 + eps) - obs_at(v0 - eps)) / (2.0 * eps)


def _psr_gradient(H_td, t_sym, v0, T, psi0, observable, n_qubits,
                  n_sample, tau_list=None, param_dict=None):
    programs = observable_program_generator_td(
        H_td, t_sym, T,
        n_sample=n_sample,
        n_repetition=1,
        diff_var="v",
        value=v0,
        param_dict=param_dict,
        tau_list=tau_list,
    )
    expfn = make_td_expectation_fn(psi0, observable, n_qubits=n_qubits)
    return combine_gradient_results_td(programs, expfn, T), programs


# ═════════════════════════════════════════════════════════════════════════════
# Runner sanity
# ═════════════════════════════════════════════════════════════════════════════

class TestRunTDSequence:

    def test_constant_H_matches_TI_evolution(self):
        """A TD segment whose coef has no t-dependence must match TI sesolve."""
        t = sp.Symbol("t")
        qs = QSystem(); q = [Qubit(qs)]
        H = 1.0 * q[0].X  # constant
        H_subst = H.set_parameterizedHam({})  # no-op → TIHamiltonian
        psi0 = qp.basis(2, 0)

        # TD-segment path
        seg = [{"kind": "td", "H": H_subst, "t_sym": t, "t_start": 0.0, "t_end": np.pi / 2}]
        final_td = run_td_sequence(seg, psi0, n_qubits=1)

        # TI-segment path
        final_ti = run_td_sequence([[H_subst, np.pi / 2]], psi0, n_qubits=1)

        overlap = final_td.overlap(final_ti)
        assert abs(abs(overlap) - 1.0) < 1e-6

    def test_kick_freezes_envelope(self):
        """Kick is a TI segment with duration (1+sgn*3/4)π on H_j = X_0.

        With sin(t) envelope frozen during the kick, the evolution over the
        kick segment is the pure rotation e^{-i·α·X}; verify against analytic.
        """
        t = sp.Symbol("t")
        qs = QSystem(); q = [Qubit(qs)]
        Hj = 1.0 * q[0].X
        Hj_ti = Hj.set_parameterizedHam({})  # TIHamiltonian

        psi0 = qp.basis(2, 0)
        alpha = 0.3
        state = run_td_sequence([[Hj_ti, alpha]], psi0, n_qubits=1)

        expected = (-1j * alpha * qp.sigmax()).expm() * psi0
        assert abs(abs(state.overlap(expected)) - 1.0) < 1e-6

    def test_parametrized_to_qutip_td_empty(self):
        """Empty Hamiltonian → None."""
        t = sp.Symbol("t")
        from simuq.hamiltonian import Parametrized_Hamiltonian
        H_empty = Parametrized_Hamiltonian(["qubit"], ["q0"], [])
        assert _parametrized_to_qutip_td(H_empty, t, 1) is None


# ═════════════════════════════════════════════════════════════════════════════
# PSR vs finite differences
# ═════════════════════════════════════════════════════════════════════════════

class TestPSRvsFiniteDiff:

    def test_example1_linear_in_v(self):
        """H = v·sin(t)·X_0 + cos(t)·Z_0 — ∂f/∂v = sin(t), plain TD ugrad."""
        np.random.seed(0)
        v, t = sp.symbols("v t")
        qs = QSystem(); q = [Qubit(qs)]
        H_td = v * sp.sin(t) * q[0].X + sp.cos(t) * q[0].Z

        v0 = 1.0
        T = np.pi / 2
        psi0 = qp.basis(2, 0)
        Z0 = qp.sigmaz()

        grad_fd = _fd_gradient(H_td, t, v, v0, T, psi0, Z0, n_qubits=1)
        assert abs(grad_fd) > 1e-3  # non-trivial gradient

        n_sample = 600
        tau_list = np.random.rand(n_sample) * T
        grad_psr, programs = _psr_gradient(
            H_td, t, v0, T, psi0, Z0, n_qubits=1,
            n_sample=n_sample, tau_list=tau_list,
        )

        # One term contributes (coef of X_0 has v in it; Z_0 term does not).
        assert len(programs) == 1

        rel_err = abs(grad_psr - grad_fd) / max(abs(grad_fd), 1e-8)
        assert rel_err < 0.02, (
            f"PSR-TD grad {grad_psr:.6f} vs FD {grad_fd:.6f}  rel err {rel_err:.3%}"
        )

    def test_example2_parameter_inside_envelope(self):
        """H = sin(v·t)·X_0 + cos(t)·Z_0 — ugrad(τ) = τ·cos(v·τ)."""
        np.random.seed(42)
        v, t = sp.symbols("v t")
        qs = QSystem(); q = [Qubit(qs)]
        H_td = sp.sin(v * t) * q[0].X + sp.cos(t) * q[0].Z

        v0 = 0.8
        T = np.pi / 2
        psi0 = qp.basis(2, 0)
        Z0 = qp.sigmaz()

        grad_fd = _fd_gradient(H_td, t, v, v0, T, psi0, Z0, n_qubits=1)
        assert abs(grad_fd) > 1e-3

        n_sample = 800
        tau_list = np.random.rand(n_sample) * T
        grad_psr, _ = _psr_gradient(
            H_td, t, v0, T, psi0, Z0, n_qubits=1,
            n_sample=n_sample, tau_list=tau_list,
        )

        rel_err = abs(grad_psr - grad_fd) / max(abs(grad_fd), 1e-8)
        assert rel_err < 0.03, (
            f"PSR-TD grad {grad_psr:.6f} vs FD {grad_fd:.6f}  rel err {rel_err:.3%}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# Generator structure
# ═════════════════════════════════════════════════════════════════════════════

class TestGeneratorStructure:

    def test_skips_terms_with_zero_ugrad(self):
        """Terms whose coef doesn't depend on diff_var are dropped."""
        v, t = sp.symbols("v t")
        qs = QSystem(); q = [Qubit(qs)]
        H_td = v * sp.sin(t) * q[0].X + 3.0 * q[0].Z  # only X term has v

        programs = observable_program_generator_td(
            H_td, t, T=np.pi,
            n_sample=4, n_repetition=1,
            diff_var="v", value=1.0,
            tau_list=[0.1, 0.2, 0.3, 0.4],
        )
        assert len(programs) == 1
        branches, ugrad, _ = programs[0]
        assert len(branches) == 2 * 4
        assert len(ugrad) == 4

    def test_branch_segment_shape(self):
        """Each branch: [TD-seg, TI-kick, TD-seg] with correct τ split."""
        v, t = sp.symbols("v t")
        qs = QSystem(); q = [Qubit(qs)]
        H_td = v * sp.sin(t) * q[0].X
        T = 2.0
        tau_list = [0.7]

        programs = observable_program_generator_td(
            H_td, t, T,
            n_sample=1, n_repetition=1,
            diff_var="v", value=1.0,
            tau_list=tau_list,
        )
        branches, ugrad, _ = programs[0]
        assert len(branches) == 2
        for branch, sgn in zip(branches, (-1, 1)):
            assert len(branch) == 3
            seg0, kick, seg2 = branch
            assert seg0["kind"] == "td" and seg0["t_start"] == 0.0 and seg0["t_end"] == 0.7
            assert seg2["kind"] == "td" and seg2["t_start"] == 0.7 and seg2["t_end"] == T
            _, kick_dur = kick
            expected_kick = (1 + sgn * 0.75) * np.pi
            assert abs(kick_dur - expected_kick) < 1e-12

    def test_ugrad_evaluation_matches_sympy(self):
        """∂f/∂v evaluated at each τ matches symbolic derivative."""
        v, t = sp.symbols("v t")
        qs = QSystem(); q = [Qubit(qs)]
        H_td = sp.sin(v * t) * q[0].X  # ∂/∂v = t·cos(v·t)

        v0 = 0.5
        tau_list = [0.3, 0.6, 1.2]
        programs = observable_program_generator_td(
            H_td, t, T=2.0,
            n_sample=len(tau_list), n_repetition=1,
            diff_var="v", value=v0,
            tau_list=tau_list,
        )
        _, ugrad, _ = programs[0]
        for i, tau in enumerate(tau_list):
            expected = tau * np.cos(v0 * tau)
            assert abs(ugrad[i] - expected) < 1e-10


# ═════════════════════════════════════════════════════════════════════════════
# combine_gradient_results_td
# ═════════════════════════════════════════════════════════════════════════════

class TestCombineTD:

    def test_length_mismatch_raises(self):
        """ugrad length must equal n_sample."""
        v, t = sp.symbols("v t")
        qs = QSystem(); q = [Qubit(qs)]
        H_td = v * sp.sin(t) * q[0].X

        programs = observable_program_generator_td(
            H_td, t, T=np.pi,
            n_sample=2, n_repetition=1,
            diff_var="v", value=1.0,
            tau_list=[0.1, 0.2],
        )
        # Corrupt one ugrad list
        programs[0][1] = programs[0][1] + [0.0]  # length 3 != 2

        psi0 = qp.basis(2, 0)
        Z0 = qp.sigmaz()
        expfn = make_td_expectation_fn(psi0, Z0, n_qubits=1)

        with pytest.raises(ValueError, match="ugrad list length"):
            combine_gradient_results_td(programs, expfn, np.pi)
