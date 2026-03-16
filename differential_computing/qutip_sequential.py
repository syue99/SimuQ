"""
QuTiPSequentialRunner: chains multiple sesolve calls to simulate
multi-segment Hamiltonian sequences produced by observable_program_generator().

Each segment in H_list is (TIHamiltonian, duration). The final state of
segment i becomes the initial state of segment i+1.
"""
import sys
sys.path.insert(0, "/Users/syue99/research/SimuQ/src/")

import qutip as qp


class QuTiPSequentialRunner:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits

    def zero_state(self):
        """Return |00...0> as a QuTiP ket."""
        return qp.tensor([qp.basis(2, 0)] * self.n_qubits)

    def zz_observable(self, i, j):
        """Return Z_i ⊗ Z_j observable."""
        ops = [qp.qeye(2)] * self.n_qubits
        ops[i] = qp.sigmaz()
        ops[j] = qp.sigmaz()
        return qp.tensor(ops)

    def run_sequence(self, H_list, psi0):
        """
        Evolve psi0 through each segment in H_list sequentially.

        H_list: list of [TIHamiltonian, duration] pairs
        psi0:   QuTiP ket (initial state)

        Returns the final QuTiP ket after all segments.
        """
        state = psi0
        for H, duration in H_list:
            if duration == 0:
                continue
            H_qobj = H.to_qutip_qobj()
            result = qp.sesolve(H_qobj, state, [0, float(duration)])
            state = result.states[-1]
        return state

    def make_expectation_fn(self, psi0, observable):
        """
        Return a function  expfn(H_list) -> float  that:
          1. Runs the full H_list sequence starting from psi0
          2. Returns <observable> on the final state
        """
        def expfn(H_list):
            final_state = self.run_sequence(H_list, psi0)
            return float(qp.expect(observable, final_state).real)
        return expfn
