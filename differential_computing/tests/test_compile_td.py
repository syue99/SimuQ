"""
End-to-end tests for diffQCProvider.compile_td / run_td.

The simulator path here is the same math as test_td_psr.py, but routed through
the provider API so that downstream TD users call::

    prov.compile_td(H_td, t, T)
    programs = observable_program_generator_td(...)
    prov.run_td(programs, observable)
    grad = prov.results()
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
from simuq.braket.diffQC_provider import diffQCProvider

from td_psr import observable_program_generator_td, run_td_sequence


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fd(H_td, t_sym, v_sym, v0, T, psi0, obs, n_q, eps=1e-4):
    def at(v_val):
        H_sub = H_td.set_parameterizedHam({str(v_sym): v_val})
        seg = [{"kind": "td", "H": H_sub, "t_sym": t_sym,
                "t_start": 0.0, "t_end": T}]
        state = run_td_sequence(seg, psi0, n_q)
        return float(qp.expect(obs, state).real)
    return (at(v0 + eps) - at(v0 - eps)) / (2 * eps)


# ═════════════════════════════════════════════════════════════════════════════
# compile_td structure
# ═════════════════════════════════════════════════════════════════════════════

class TestCompileTDStructure:

    def test_envelope_strategy_for_disjoint(self):
        """Disjoint 1-body H → 2 groups, strategy='envelope'."""
        v, t = sp.symbols("v t")
        qs = QSystem(); q = [Qubit(qs)]
        H_td = v * sp.sin(t) * q[0].X + sp.cos(t) * q[0].Z

        prov = diffQCProvider()
        tdc = prov.compile_td(H_td, t, T=np.pi / 2)

        assert tdc["strategy"] == "envelope"
        assert tdc["collision"] is False
        assert len(tdc["groups"]) == 2
        assert tdc["n_sites"] == 1
        assert tdc["T"] == pytest.approx(np.pi / 2)

    def test_trotter_strategy_for_collision(self):
        """Shared atom in ≥2-body groups → strategy='trotter'."""
        t = sp.Symbol("t")
        qs = QSystem(); q = [Qubit(qs) for _ in range(3)]
        H_td = sp.sin(t) * q[0].Z * q[1].Z + sp.cos(t) * q[1].Z * q[2].Z

        prov = diffQCProvider()
        tdc = prov.compile_td(H_td, t, T=1.0)

        assert tdc["strategy"] == "trotter"
        assert tdc["collision"] is True

    def test_param_dict_passed_through(self):
        """compile_td stores the param_dict verbatim for downstream use."""
        v, x, t = sp.symbols("v x t")
        qs = QSystem(); q = [Qubit(qs)]
        H_td = v * sp.sin(x * t) * q[0].X

        prov = diffQCProvider()
        tdc = prov.compile_td(H_td, t, T=1.0, param_dict={x: 2.0})
        assert tdc["param_dict"] == {x: 2.0}


# ═════════════════════════════════════════════════════════════════════════════
# run_td: simulator path vs finite differences
# ═════════════════════════════════════════════════════════════════════════════

class TestRunTDSimulator:

    def test_gradient_matches_finite_diff(self):
        np.random.seed(7)
        v, t = sp.symbols("v t")
        qs = QSystem(); q = [Qubit(qs)]
        H_td = v * sp.sin(t) * q[0].X + sp.cos(t) * q[0].Z

        v0 = 1.0
        T = np.pi / 2
        psi0 = qp.basis(2, 0)
        Z0 = qp.sigmaz()

        # Reference
        grad_fd = _fd(H_td, t, v, v0, T, psi0, Z0, n_q=1)

        # Provider path
        prov = diffQCProvider()
        prov.compile_td(H_td, t, T)

        n_sample = 600
        tau_list = np.random.rand(n_sample) * T
        programs = observable_program_generator_td(
            H_td, t, T,
            n_sample=n_sample, n_repetition=1,
            diff_var="v", value=v0,
            tau_list=tau_list,
        )
        grad_prov = prov.run_td(programs, Z0, psi0=psi0, backend="qutip")

        rel = abs(grad_prov - grad_fd) / max(abs(grad_fd), 1e-8)
        assert rel < 0.02, (
            f"prov grad {grad_prov:.6f} vs FD {grad_fd:.6f}  rel err {rel:.3%}"
        )

    def test_hardware_backend_raises(self):
        """Hardware path is intentionally not wired yet."""
        v, t = sp.symbols("v t")
        qs = QSystem(); q = [Qubit(qs)]
        H_td = v * sp.sin(t) * q[0].X

        prov = diffQCProvider()
        prov.compile_td(H_td, t, T=1.0)

        programs = observable_program_generator_td(
            H_td, t, 1.0,
            n_sample=2, n_repetition=1,
            diff_var="v", value=1.0,
            tau_list=[0.3, 0.7],
        )

        with pytest.raises(NotImplementedError, match="play_wf"):
            prov.run_td(programs, qp.sigmaz(), backend="hardware")

    def test_run_td_before_compile_raises(self):
        prov = diffQCProvider()
        with pytest.raises(RuntimeError, match="compile_td"):
            prov.run_td([], qp.sigmaz())
