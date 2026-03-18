"""
Unit tests for the DiffSimuQ gradient pipeline.

Tests cover:
  - QuTiPSequentialRunner (state helpers, segment chaining)
  - observable_program_generator (output structure, kick durations, ugrad values)
  - combine_gradient_results (trivial cases, sign, scaling)
  - End-to-end: parameter-shift gradient vs finite differences
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sympy as sp
import pytest
import qutip as qp

from simuq import QSystem, Qubit
from simuq.hamiltonian import productHamiltonian, TIHamiltonian

from qutip_sequential import QuTiPSequentialRunner
from combine_gradient import combine_gradient_results
from observable_program_generator import observable_program_generator


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_1q_H(coef, op="Z"):
    qs = QSystem(); q = [Qubit(qs)]
    return coef * (q[0].Z if op == "Z" else q[0].X)


def make_2q_zz(coef=1.0):
    qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    return coef * q[0].Z * q[1].Z


def build_param_H(theta_val=np.pi / 4 - 0.1):
    """2-qubit Parametrized_Hamiltonian used in the notebook demo."""
    x, theta = sp.symbols("x theta")
    J0 = sp.sin(2 * x + theta)
    qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
    H = J0*q[0].Z*q[1].Z + J0*q[0].X + J0*q[1].X + J0*q[0].Z*q[1].Z
    return H.set_parameterizedHam({"theta": theta_val})


# ═════════════════════════════════════════════════════════════════════════════
# QuTiPSequentialRunner
# ═════════════════════════════════════════════════════════════════════════════

class TestQuTiPSequentialRunner:

    def test_zero_state_1qubit(self):
        runner = QuTiPSequentialRunner(n_qubits=1)
        psi = runner.zero_state()
        assert psi.shape == (2, 1)
        arr = psi.full()
        assert abs(arr[0, 0] - 1.0) < 1e-12
        assert abs(arr[1, 0])       < 1e-12

    def test_zero_state_2qubit(self):
        runner = QuTiPSequentialRunner(n_qubits=2)
        psi = runner.zero_state()
        assert psi.shape == (4, 1)
        arr = psi.full()
        assert abs(arr[0, 0] - 1.0) < 1e-12
        for i in range(1, 4):
            assert abs(arr[i, 0]) < 1e-12

    def test_zz_observable_diagonal(self):
        """Z0⊗Z1 must be diagonal with eigenvalues (+1,-1,-1,+1)."""
        runner = QuTiPSequentialRunner(n_qubits=2)
        obs = runner.zz_observable(0, 1)
        diag = np.diag(obs.full()).real
        expected = [1.0, -1.0, -1.0, 1.0]
        for d, e in zip(diag, expected):
            assert abs(d - e) < 1e-12

    def test_run_sequence_zero_duration_skipped(self):
        """Segments with duration 0 must be skipped (state unchanged)."""
        runner = QuTiPSequentialRunner(n_qubits=1)
        psi0 = runner.zero_state()
        H = make_1q_H(1.0, "Z")
        final = runner.run_sequence([[H, 0.0]], psi0)
        assert qp.fidelity(psi0, final) > 1 - 1e-10

    def test_run_sequence_single_segment_matches_sesolve(self):
        """Single segment must reproduce direct sesolve."""
        runner = QuTiPSequentialRunner(n_qubits=1)
        psi0 = runner.zero_state()
        H = make_1q_H(1.0, "X")
        t = 0.5
        psi_runner = runner.run_sequence([[H, t]], psi0)
        ref = qp.sesolve(H.to_qutip_qobj(), psi0, [0, t]).states[-1]
        assert qp.fidelity(psi_runner, ref) > 1 - 1e-8

    def test_run_sequence_two_segments_chained(self):
        """Two segments must chain: final state of seg1 feeds into seg2."""
        runner = QuTiPSequentialRunner(n_qubits=1)
        psi0 = runner.zero_state()
        H1 = make_1q_H(1.0, "X")
        H2 = make_1q_H(1.0, "Z")
        t1, t2 = 0.3, 0.7
        psi_runner = runner.run_sequence([[H1, t1], [H2, t2]], psi0)
        mid = qp.sesolve(H1.to_qutip_qobj(), psi0,  [0, t1]).states[-1]
        ref = qp.sesolve(H2.to_qutip_qobj(), mid,   [0, t2]).states[-1]
        assert qp.fidelity(psi_runner, ref) > 1 - 1e-8

    def test_expectation_fn_returns_real_float(self):
        runner = QuTiPSequentialRunner(n_qubits=2)
        psi0 = runner.zero_state()
        obs = runner.zz_observable(0, 1)
        expfn = runner.make_expectation_fn(psi0, obs)
        H = make_2q_zz(0.5)
        val = expfn([[H, 1.0]])
        assert isinstance(val, float)

    def test_expectation_zz_on_zero_state_is_one(self):
        """<00|Z⊗Z|00> = +1 (both qubits in |0>, eigenvalue +1)."""
        runner = QuTiPSequentialRunner(n_qubits=2)
        psi0 = runner.zero_state()
        obs = runner.zz_observable(0, 1)
        val = float(qp.expect(obs, psi0).real)
        assert abs(val - 1.0) < 1e-12

    def test_expectation_zz_eigenstate_unchanged(self):
        """|00> is an eigenstate of Z⊗Z; <ZZ> stays 1 for any evolution time."""
        runner = QuTiPSequentialRunner(n_qubits=2)
        psi0 = runner.zero_state()
        obs = runner.zz_observable(0, 1)
        expfn = runner.make_expectation_fn(psi0, obs)
        H = make_2q_zz(1.0)
        for t in [0.1, np.pi / 4, np.pi / 2]:
            val = expfn([[H, t]])
            assert abs(val - 1.0) < 1e-6, f"<ZZ> should be 1 at t={t}, got {val}"


# ═════════════════════════════════════════════════════════════════════════════
# observable_program_generator
# ═════════════════════════════════════════════════════════════════════════════

class TestObservableProgramGenerator:

    def setup_method(self):
        np.random.seed(0)
        self.H_param = build_param_H()

    def test_number_of_terms(self):
        """One entry per distinct Hj operator with non-zero gradient."""
        programs = observable_program_generator(
            self.H_param, T=1.0, n_sample=3, n_repetition=1, diff_var="x", value=1.0
        )
        u_grad_dict = self.H_param.take_diff_coef("x")
        n_nonzero = sum(1 for v in u_grad_dict.values()
                        if (float(v.subs("x", 1.0)) if isinstance(v, sp.Expr) else float(v)) != 0.0)
        assert len(programs) == n_nonzero

    def test_number_of_hlists_per_term(self):
        """Each term must have 2*n_sample H_lists (sgn=±1 per tau)."""
        n_sample = 5
        programs = observable_program_generator(
            self.H_param, T=1.0, n_sample=n_sample, n_repetition=1, diff_var="x", value=1.0
        )
        for H_tot_list, _, _ in programs:
            assert len(H_tot_list) == 2 * n_sample

    def test_hlist_has_three_segments(self):
        programs = observable_program_generator(
            self.H_param, T=1.0, n_sample=4, n_repetition=1, diff_var="x", value=1.0
        )
        for H_tot_list, _, _ in programs:
            for H_list in H_tot_list:
                assert len(H_list) == 3

    def test_kick_durations_correct(self):
        """Kick durations must be π/4 (sgn=-1) and 7π/4 (sgn=+1) per Algorithm 1."""
        kick_minus = np.pi / 4          # (1 - 3/4) * π
        kick_plus  = 7 * np.pi / 4     # (1 + 3/4) * π
        programs = observable_program_generator(
            self.H_param, T=1.0, n_sample=3, n_repetition=1, diff_var="x", value=1.0
        )
        for H_tot_list, _, _ in programs:
            for i in range(0, len(H_tot_list), 2):
                assert abs(H_tot_list[i][1][1]   - kick_minus) < 1e-10, "sgn=-1 kick wrong"
                assert abs(H_tot_list[i+1][1][1] - kick_plus)  < 1e-10, "sgn=+1 kick wrong"

    def test_kick_durations_both_positive(self):
        """Both kick durations must be positive (no backward evolution)."""
        programs = observable_program_generator(
            self.H_param, T=1.0, n_sample=3, n_repetition=1, diff_var="x", value=1.0
        )
        for H_tot_list, _, _ in programs:
            for H_list in H_tot_list:
                assert H_list[1][1] > 0, f"Negative kick duration: {H_list[1][1]}"

    def test_tau_segments_sum_to_T(self):
        T = 2.5
        programs = observable_program_generator(
            self.H_param, T=T, n_sample=4, n_repetition=1, diff_var="x", value=1.0
        )
        for H_tot_list, _, _ in programs:
            for H_list in H_tot_list:
                assert abs(H_list[0][1] + H_list[2][1] - T) < 1e-10

    def test_ugrad_is_correct_sympy_derivative(self):
        value = 1.0
        programs = observable_program_generator(
            self.H_param, T=1.0, n_sample=2, n_repetition=1, diff_var="x", value=value
        )
        u_grad_dict = self.H_param.take_diff_coef("x")
        refs = []
        for raw in u_grad_dict.values():
            ev = float(raw.subs("x", value)) if isinstance(raw, sp.Expr) else float(raw)
            if ev != 0.0:
                refs.append(ev)
        assert len(programs) == len(refs)
        for (_, ugrad, _), expected in zip(programs, refs):
            assert abs(float(ugrad) - expected) < 1e-10

    def test_sgn_ordering_minus_first(self):
        """Even indices → sgn=-1 (π/4 kick); odd → sgn=+1 (7π/4 kick)."""
        programs = observable_program_generator(
            self.H_param, T=1.0, n_sample=3, n_repetition=1, diff_var="x", value=1.0
        )
        for H_tot_list, _, _ in programs:
            for i in range(0, len(H_tot_list), 2):
                assert H_tot_list[i][1][1]   < H_tot_list[i+1][1][1], \
                    "Even index (sgn=-1) must have shorter kick than odd index (sgn=+1)"

    def test_zero_gradient_terms_skipped(self):
        """Terms with zero gradient coefficient must not appear in output."""
        x_sym = sp.Symbol("x")
        qs = QSystem(); q = [Qubit(qs)]
        # H = x*Z + 1*X: only Z term depends on x
        H = x_sym * q[0].Z + 1 * q[0].X
        programs = observable_program_generator(H, T=1.0, n_sample=3, n_repetition=1,
                                                 diff_var="x", value=1.0)
        # Only the Z term contributes; X has constant coefficient 1 → d(1)/dx = 0
        assert len(programs) == 1


# ═════════════════════════════════════════════════════════════════════════════
# combine_gradient_results
# ═════════════════════════════════════════════════════════════════════════════

class TestCombineGradientResults:

    def test_zero_gradient_when_f_minus_equals_f_plus(self):
        """If f_- == f_+ for all branches, gradient must be zero."""
        constant_expfn = lambda H_list: 0.5
        H_tot = [[[None, 0], [None, 1], [None, 1]]] * 4
        programs = [(H_tot, 2.0, 1)]
        assert abs(combine_gradient_results(programs, constant_expfn, T=1.0)) < 1e-12

    def test_gradient_sign_positive(self):
        """ugrad>0 and f_- > f_+ → gradient > 0 (paper: p̃⁻ − p̃⁺)."""
        def expfn(H_list):
            # shorter kick (π/4) → larger value; longer kick (7π/4) → smaller
            return 1.0 if H_list[1][1] < 2.0 else 0.0

        H_tot = [
            [[None, 0.3], [None, np.pi/4],     [None, 0.7]],  # sgn=-1, p̃⁻=1
            [[None, 0.3], [None, 7*np.pi/4],   [None, 0.7]],  # sgn=+1, p̃⁺=0
        ]
        programs = [(H_tot, 1.0, 1)]
        assert combine_gradient_results(programs, expfn, T=1.0) > 0

    def test_gradient_scales_linearly_with_T(self):
        def expfn(H_list):
            return 0.7 if H_list[1][1] < 2.0 else 0.3  # f_minus > f_plus

        H_tot = [
            [[None, 0.5], [None, np.pi/4],   [None, 0.5]],
            [[None, 0.5], [None, 7*np.pi/4], [None, 0.5]],
        ]
        programs = [(H_tot, 1.0, 1)]
        g1 = combine_gradient_results(programs, expfn, T=1.0)
        g2 = combine_gradient_results(programs, expfn, T=2.0)
        assert abs(g2 - 2 * g1) < 1e-10

    def test_gradient_scales_linearly_with_ugrad(self):
        def expfn(H_list):
            return 1.0 if H_list[1][1] < 2.0 else 0.0

        H_tot = [
            [[None, 0.5], [None, np.pi/4],   [None, 0.5]],
            [[None, 0.5], [None, 7*np.pi/4], [None, 0.5]],
        ]
        g1 = combine_gradient_results([(H_tot, 1.0, 1)], expfn, T=1.0)
        g2 = combine_gradient_results([(H_tot, 3.0, 1)], expfn, T=1.0)
        assert abs(g2 - 3 * g1) < 1e-10

    def test_empty_programs(self):
        assert combine_gradient_results([], lambda _: 0.5, T=1.0) == 0.0

    def test_averages_over_tau_samples(self):
        """
        Two tau samples: (f_minus - f_plus)=1.0 for tau0, 0.0 for tau1.
        Expected: T * mean([1, 0]) * ugrad = 1.0 * 0.5 * 1.0 = 0.5
        """
        H_tot = [
            [[None, 0.1], [None, np.pi/4],     [None, 0.9]],  # tau0, sgn=-1 → 1
            [[None, 0.1], [None, 7*np.pi/4],   [None, 0.9]],  # tau0, sgn=+1 → 0
            [[None, 0.6], [None, np.pi/4],     [None, 0.4]],  # tau1, sgn=-1 → 1
            [[None, 0.6], [None, 7*np.pi/4],   [None, 0.4]],  # tau1, sgn=+1 → 1
        ]
        def expfn(H_list):
            tau, kick = H_list[0][1], H_list[1][1]
            if tau < 0.5:
                return 1.0 if kick < 2.0 else 0.0  # f_minus=1, f_plus=0 → diff=1
            return 1.0   # both same → diff=0
        programs = [(H_tot, 1.0, 1)]
        assert abs(combine_gradient_results(programs, expfn, T=1.0) - 0.5) < 1e-10


# ═════════════════════════════════════════════════════════════════════════════
# End-to-end: parameter-shift vs finite differences
# ═════════════════════════════════════════════════════════════════════════════

class TestEndToEndGradient:
    """
    Validates the full pipeline converges to the FD gradient.
    Uses T=1 to keep dynamics well-behaved.
    With the correct formula (arXiv:2210.15812 Algorithm 1), n_sample=2000
    gives <1% error (essentially exact up to MC variance at these seeds).
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.T = 1.0
        runner = QuTiPSequentialRunner(n_qubits=2)
        self.psi0 = runner.zero_state()
        self.obs  = runner.zz_observable(0, 1)
        self.expfn = runner.make_expectation_fn(self.psi0, self.obs)

    def _fd_2q(self, x_val, eps=1e-4):
        """Central FD reference for d<ZZ>/dx on the 2-qubit notebook Hamiltonian."""
        def f(xv):
            J = sp.sin(2*sp.Symbol("x") + sp.Symbol("theta"))
            qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
            H = J*q[0].Z*q[1].Z + J*q[0].X + J*q[1].X + J*q[0].Z*q[1].Z
            He = H.set_parameterizedHam({"theta": np.pi/4 - 0.1, "x": xv})
            r = qp.sesolve(He.to_qutip_qobj(),
                           qp.tensor([qp.basis(2, 0)] * 2), [0, self.T])
            return float(qp.expect(self.obs, r.states[-1]).real)
        return (f(x_val + eps) - f(x_val - eps)) / (2 * eps)

    def test_gradient_sign_matches_fd(self):
        np.random.seed(1)
        programs = observable_program_generator(
            build_param_H(), self.T, n_sample=100, n_repetition=1,
            diff_var="x", value=1.0
        )
        grad = combine_gradient_results(programs, self.expfn, T=self.T)
        fd   = self._fd_2q(1.0)
        assert np.sign(grad) == np.sign(fd), \
            f"Sign mismatch: PSR={grad:.4f}, FD={fd:.4f}"

    def test_gradient_converges_to_fd_at_x1(self):
        """With correct formula, 2000 samples gives <2% error at x=1."""
        np.random.seed(7)
        programs = observable_program_generator(
            build_param_H(), self.T, n_sample=2000, n_repetition=1,
            diff_var="x", value=1.0
        )
        grad = combine_gradient_results(programs, self.expfn, T=self.T)
        fd   = self._fd_2q(1.0)
        rel  = abs(grad - fd) / (abs(fd) + 1e-12)
        assert rel < 0.02, f"Rel error {rel:.3%} (PSR={grad:.4f}, FD={fd:.4f})"

    def test_gradient_converges_to_fd_at_x3(self):
        """With correct formula, 2000 samples gives <2% error at x=3."""
        np.random.seed(7)
        programs = observable_program_generator(
            build_param_H(), self.T, n_sample=2000, n_repetition=1,
            diff_var="x", value=3.0
        )
        grad = combine_gradient_results(programs, self.expfn, T=self.T)
        fd   = self._fd_2q(3.0)
        rel  = abs(grad - fd) / (abs(fd) + 1e-12)
        assert rel < 0.02, f"Rel error {rel:.3%} at x=3 (PSR={grad:.4f}, FD={fd:.4f})"

    def test_single_qubit_gradient_noncommuting(self):
        """
        1-qubit H(x) = x*Z + X (non-commuting).
        With 2000 samples the correct formula gives <2% error.
        """
        runner1 = QuTiPSequentialRunner(n_qubits=1)
        psi0_1  = runner1.zero_state()
        obs_z   = qp.sigmaz()
        expfn1  = runner1.make_expectation_fn(psi0_1, obs_z)

        x_sym = sp.Symbol("x")
        qs1 = QSystem(); q1 = [Qubit(qs1)]
        H1q = x_sym * q1[0].Z + 1 * q1[0].X

        np.random.seed(7)
        programs = observable_program_generator(
            H1q, T=1.0, n_sample=2000, n_repetition=1, diff_var="x", value=1.0
        )
        grad = combine_gradient_results(programs, expfn1, T=1.0)

        def f1q(xv):
            qs_ = QSystem(); q_ = [Qubit(qs_)]
            He  = (x_sym * q_[0].Z + 1 * q_[0].X).set_parameterizedHam({"x": xv})
            r   = qp.sesolve(He.to_qutip_qobj(), qp.basis(2, 0), [0, 1.0])
            return float(qp.expect(obs_z, r.states[-1]).real)

        eps = 1e-4
        fd  = (f1q(1.0 + eps) - f1q(1.0 - eps)) / (2 * eps)
        rel = abs(grad - fd) / (abs(fd) + 1e-12)
        assert rel < 0.02, f"1-qubit rel error {rel:.3%} (PSR={grad:.4f}, FD={fd:.4f})"
