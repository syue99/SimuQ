"""
noisy_qutip.py — open-system (density-matrix) sequential runner.

Mirrors QuTiPSequentialRunner but evolves a density matrix with mesolve so that
T1/T2 collapse operators and a discrete Pauli channel (see noise_model.py) can
act on the multi-segment PSR Hamiltonian sequences.

Drop-in for the gradient assembler: make_expectation_fn(psi0, O) returns the same
expfn(H_list) -> float seam that combine_gradient_results consumes, so the PSR
gradient pipeline runs unchanged — only the expectation becomes noisy.

With noise=None (or an all-off NoiseModel) the runner does unitary density-matrix
evolution, which reproduces the sesolve expectations of QuTiPSequentialRunner —
that equivalence is the first validation check.
"""

import sys
sys.path.insert(0, "/Users/syue99/research/SimuQ/src/")

import qutip as qp


class NoisyQuTiPRunner:
    def __init__(self, n_qubits, noise=None):
        """
        n_qubits : int
        noise    : NoiseModel | None — None means coherent (no noise).
        """
        self.n_qubits = n_qubits
        self.noise = noise

    # ── States / observables (same API as QuTiPSequentialRunner) ──────────────
    def zero_state(self):
        """|00...0> as a QuTiP ket."""
        return qp.tensor([qp.basis(2, 0)] * self.n_qubits)

    def zz_observable(self, i, j):
        ops = [qp.qeye(2)] * self.n_qubits
        ops[i] = qp.sigmaz()
        ops[j] = qp.sigmaz()
        return qp.tensor(ops)

    # ── Evolution ─────────────────────────────────────────────────────────────
    def run_sequence(self, H_list, psi0):
        """Evolve psi0 through each segment as a density matrix.

        psi0 : QuTiP ket or density matrix
        Returns the final density matrix.

        All noise (T1/T2 + Pauli rates) enters as mesolve collapse operators, so
        it is integrated over each segment's real duration — automatically
        duration-scaled and fair between FD (1 segment) and PSR (3 segments).
        """
        rho = qp.ket2dm(psi0) if psi0.isket else psi0

        c_ops = (self.noise.collapse_ops()
                 if (self.noise is not None and self.noise.has_noise()) else [])

        for H, duration in H_list:
            if duration == 0:
                continue
            H_qobj = H.to_qutip_qobj()
            result = qp.mesolve(H_qobj, rho, [0.0, float(duration)], c_ops=c_ops)
            rho = result.states[-1]
        return rho

    def make_expectation_fn(self, psi0, observable):
        """Return expfn(H_list) -> float = Tr(O · ρ_final) (exact noisy ⟨O⟩)."""
        def expfn(H_list):
            rho = self.run_sequence(H_list, psi0)
            return float(qp.expect(observable, rho).real)
        return expfn
