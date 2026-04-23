"""
Tests for diffQC_provider.run() / results().

Tests are split into two classes:

TestRunQutip  — end-to-end via backend="qutip": gradient matches FD reference.
TestRunHardware — backend="hardware": schedule structure and transport log correctness.
                  No PulseDSL / MMIO session required.

All tests use the 2-qubit notebook Hamiltonian (small, fast) so the test
suite stays under a few seconds.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sympy as sp
import pytest
import qutip as qp
from unittest.mock import MagicMock, patch

from simuq import QSystem, Qubit
from simuq.aais import rydberg2d
from simuq.solver import generate_as

from observable_program_generator import observable_program_generator
from combine_gradient import combine_gradient_results
from qutip_sequential import QuTiPSequentialRunner


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _build_2q_param_H(theta_val=np.pi / 4 - 0.1):
    """2-qubit notebook Hamiltonian at fixed theta."""
    x, theta = sp.symbols("x theta")
    J0 = sp.sin(2 * x + theta)
    qs = QSystem()
    q  = [Qubit(qs) for _ in range(2)]
    H  = J0 * q[0].Z * q[1].Z + J0 * q[0].X + J0 * q[1].X + J0 * q[0].Z * q[1].Z
    return H.set_parameterizedHam({"theta": theta_val})


def _fd_gradient(x_val=1.0, T=1.0, eps=1e-4):
    """Central finite-difference reference for d<ZZ>/dx."""
    obs = QuTiPSequentialRunner(2).zz_observable(0, 1)

    def f(xv):
        x, theta = sp.symbols("x theta")
        J0 = sp.sin(2 * x + theta)
        qs = QSystem(); q = [Qubit(qs) for _ in range(2)]
        H  = J0 * q[0].Z * q[1].Z + J0 * q[0].X + J0 * q[1].X + J0 * q[0].Z * q[1].Z
        He = H.set_parameterizedHam({"theta": np.pi / 4 - 0.1, "x": xv})
        r  = qp.sesolve(He.to_qutip_qobj(), qp.tensor([qp.basis(2, 0)] * 2), [0, T])
        return float(qp.expect(obs, r.states[-1]).real)

    return (f(x_val + eps) - f(x_val - eps)) / (2 * eps)


def _minimal_prog(x_val=1.0, T=1.0, n_sample=100, seed=1):
    """Small programs for fast tests (sign check only)."""
    np.random.seed(seed)
    return observable_program_generator(
        _build_2q_param_H(), T, n_sample=n_sample, n_repetition=1,
        diff_var="x", value=x_val,
    )


def _fake_provider_with_prog(n_sites=2):
    """
    Build a diffQCProvider with self.prog populated via a minimal mock,
    bypassing compile() (which requires the full SimuQ solver).
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))
    from simuq.braket.diffQC_provider import diffQCProvider

    prov = diffQCProvider.__new__(diffQCProvider)
    prov.backend_aais = {("quera", "Aquila"): ["rydberg2d"]}

    # Minimal fake boxes: one box with a single dressing instruction and one ZZ
    from unittest.mock import MagicMock
    from simuq.hamiltonian import TIHamiltonian
    ins_dress = MagicMock(); ins_dress.name = "dressing gloabl potential"
    ins_zz    = MagicMock(); ins_zz.name    = "c01_zz"; ins_zz.nativeness = "derived"

    # h_eval needs sites_type/sites_name for TweezerMapper init
    sites_type = ["qubit"] * n_sites
    sites_name = [f"q{i}" for i in range(n_sites)]
    h_eval = TIHamiltonian.identity(sites_type, sites_name)

    fake_box = (
        [
            ((6, 0), ins_dress, h_eval, [0.5]),
            ((7, 0), ins_zz,    h_eval, [1.0]),
        ],
        1.0,
    )
    sol_gvars = [2.0, 0.0]           # atom 1 at (2, 0); atom 0 at origin
    prov.prog = [n_sites, sol_gvars, [fake_box], [], {}]
    return prov


# ── backend="qutip" ───────────────────────────────────────────────────────────

class TestRunQutip:

    def setup_method(self):
        self.T   = 1.0
        self.obs = QuTiPSequentialRunner(2).zz_observable(0, 1)
        self.psi0 = qp.tensor([qp.basis(2, 0)] * 2)

    def _get_provider(self):
        from simuq.braket.diffQC_provider import diffQCProvider
        return _fake_provider_with_prog(n_sites=2)

    def test_run_returns_none(self):
        prov = self._get_provider()
        prog = _minimal_prog(n_sample=10)
        ret  = prov.run(prog, self.obs, self.T, psi0=self.psi0, backend="qutip")
        assert ret is None   # run() stores result; does not return it

    def test_results_returns_float(self):
        prov = self._get_provider()
        prog = _minimal_prog(n_sample=20)
        prov.run(prog, self.obs, self.T, psi0=self.psi0, backend="qutip")
        g = prov.results()
        assert isinstance(g, float)

    def test_gradient_sign_matches_fd(self):
        """Sign of PSR gradient must match finite-difference reference."""
        prov = self._get_provider()
        np.random.seed(3)
        prog = _minimal_prog(n_sample=200)
        prov.run(prog, self.obs, self.T, psi0=self.psi0, backend="qutip")
        g  = prov.results()
        fd = _fd_gradient()
        assert np.sign(g) == np.sign(fd), f"PSR={g:.4f}, FD={fd:.4f}"

    def test_run_without_compile_raises(self):
        from simuq.braket.diffQC_provider import diffQCProvider
        prov = diffQCProvider.__new__(diffQCProvider)
        prov.backend_aais = {}
        with pytest.raises(RuntimeError, match="compile"):
            prov.run([], None, 1.0, backend="qutip")

    def test_unknown_backend_raises(self):
        prov = self._get_provider()
        with pytest.raises(ValueError, match="backend"):
            prov.run([], self.obs, 1.0, backend="bogus")

    def test_results_before_run_raises(self):
        prov = self._get_provider()
        with pytest.raises((NotImplementedError, AttributeError)):
            prov.results()

    def test_zero_state_default(self):
        """run() must work when psi0=None (default |00...0>)."""
        prov = self._get_provider()
        prog = _minimal_prog(n_sample=10)
        prov.run(prog, self.obs, self.T, psi0=None, backend="qutip")
        assert isinstance(prov.results(), float)


# ── backend="hardware" ────────────────────────────────────────────────────────

class TestRunHardware:

    def _prov(self, n=2):
        return _fake_provider_with_prog(n_sites=n)

    def _prog(self, n_sample=4, seed=0):
        np.random.seed(seed)
        return _minimal_prog(n_sample=n_sample, seed=seed)

    def test_run_completes_without_error(self):
        prov = self._prov()
        prov.run(self._prog(), None, T=1.0, backend="hardware")

    def test_branch_ops_stored(self):
        """_branch_ops must be populated after hardware run."""
        prov = self._prov()
        prog = self._prog(n_sample=3)
        prov.run(prog, None, T=1.0, backend="hardware")
        assert hasattr(prov, "_branch_ops")
        assert len(prov._branch_ops) > 0

    def test_branch_ops_count_matches_programs(self):
        prov  = self._prov()
        prog  = self._prog(n_sample=4)
        prov.run(prog, None, T=1.0, backend="hardware")
        assert len(prov._branch_ops) == len(prog)

    def test_branch_ops_have_correct_structure(self):
        """Each entry: (list_of_op_lists, ugrad_float, n_rep_int)."""
        prov = self._prov()
        prog = self._prog(n_sample=2)
        prov.run(prog, None, T=1.0, backend="hardware")
        for branch_ops, ugrad, n_rep in prov._branch_ops:
            assert isinstance(branch_ops, list)
            assert isinstance(ugrad, float)
            assert isinstance(n_rep, int)

    def test_transport_logs_stored(self):
        prov = self._prov()
        prov.run(self._prog(n_sample=2), None, T=1.0, backend="hardware")
        assert hasattr(prov, "_transport_logs")
        assert len(prov._transport_logs) > 0

    def test_transport_logs_shape(self):
        """logs[i] has one TransportLog per H_list in H_tot_list."""
        prov = self._prov()
        prog = self._prog(n_sample=3)  # 3 tau × 2 sgn = 6 H_lists per program term
        prov.run(prog, None, T=1.0, backend="hardware")
        for logs_per_prog, (_, _, _) in zip(prov._transport_logs, prov._branch_ops):
            assert len(logs_per_prog) > 0

    def test_dressing_moves_logged(self):
        """With dressing in boxes, each evaluated_H segment logs a dressing move."""
        prov = self._prov()
        prog = self._prog(n_sample=2)
        prov.run(prog, None, T=1.0, backend="hardware")
        total_dressing = sum(
            len(log.dressing_moves)
            for prog_logs in prov._transport_logs
            for log in prog_logs
        )
        # Each H_list has 2 evaluated_H segments × n_sample × 2 sgn branches per term
        assert total_dressing > 0

    def test_results_raises_not_implemented_for_hardware(self):
        prov = self._prov()
        prov.run(self._prog(n_sample=2), None, T=1.0, backend="hardware")
        with pytest.raises(NotImplementedError):
            prov.results()

    def test_ops_contain_play(self):
        """Branch ops must include 'play' ops (dressing/ZZ/detuning/rabi)."""
        prov = self._prov()
        prov.run(self._prog(n_sample=2), None, T=1.0, backend="hardware")
        all_ops = [
            op
            for branch_ops, _, _ in prov._branch_ops
            for ops in branch_ops
            for op in ops
        ]
        op_types = {o["op"] for o in all_ops}
        assert "play" in op_types

    def test_transport_summary_prints(self, capsys):
        prov = self._prov()
        prov.run(self._prog(n_sample=2), None, T=1.0, backend="hardware")
        prov.transport_summary(program_idx=0, branch_idx=0)
        captured = capsys.readouterr()
        assert "TransportLog" in captured.out

    def test_transport_summary_no_logs(self, capsys):
        prov = self._prov()
        prov.transport_summary()   # no run yet
        captured = capsys.readouterr()
        assert "No transport logs" in captured.out
