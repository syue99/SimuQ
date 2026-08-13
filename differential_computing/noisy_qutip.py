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
    def __init__(self, n_qubits, noise=None, kick_dephases=False, nsteps=None):
        """
        n_qubits : int
        noise    : NoiseModel | None — None means coherent (no noise).
        kick_dephases : bool — whether the dressing-level dephasing/T1/Pauli
            collapse operators act during the PSR kick segment.  Physically the
            kick is compiled to a single/two-qubit GATE (clock-state rotation /
            Rydberg gate), NOT a dressed analog evolution, so the dressing T2*
            does not apply there — only the (separately modeled) gate error.
            DEFAULT False (physically faithful: collapse ops act only on the
            dressed evolution segments; the kick carries only its gate error).
            True keeps the conservative legacy "dephasing everywhere" behavior.
        """
        self.kick_dephases = kick_dephases
        self.n_qubits = n_qubits
        self.noise = noise
        self.nsteps = nsteps        # ODE step cap; larger for stiff (large-‖H‖·T) segments

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
    def _dressed_mask(self, H_list):
        """Which segments are dressed (leakage applies).

        The PSR branch is [evolve, kick, evolve]; the kick is a gate (its leakage
        is folded into the separate gate-error model), so only segments 0 and 2
        leak.  A 1-segment H_list (finite difference) is all dressed.  Any other
        length defaults to all-dressed.
        """
        if len(H_list) == 3:
            return [True, False, True]
        return [True] * len(H_list)

    def _kick_mask(self, H_list):
        """Which segments are the PSR kick (gate error applies after them)."""
        if len(H_list) == 3:
            return [False, True, False]
        return [False] * len(H_list)

    @staticmethod
    def _kick_support(H):
        """Qubit indices the kick Hamiltonian Hj acts on (its body count → ε)."""
        support = set()
        for prod, _coeff in H.ham:
            for site, op in prod.d.items():
                if op != "":
                    support.add(site)
        return sorted(support)

    def run_sequence(self, H_list, psi0):
        """Evolve psi0 through each segment as a density matrix.

        Trace-preserving noise (T1 σ⁻ / T2 dephasing / Pauli rates) enters as
        mesolve collapse operators.  Post-selected leakage (loss out of the
        subspace) enters on DRESSED segments as a CONDITIONAL (no-jump)
        evolution: H_eff = H − (i/2)·Σ Γ|1><1|_i, built into a custom Liouvillian
        alongside the trace-preserving dissipators.  The trace of ρ then decays =
        survival probability; make_expectation_fn renormalizes by it.

        All channels are integrated over each segment's real duration → duration-
        scaled and fair between FD (1 segment) and PSR (3 segments).
        """
        rho = qp.ket2dm(psi0) if psi0.isket else psi0
        opt = {"nsteps": self.nsteps} if self.nsteps else {}
        if self.noise is None or not self.noise.has_noise():
            for H, duration in H_list:
                if duration == 0:
                    continue
                rho = qp.mesolve(H.to_qutip_qobj(), rho,
                                 [0.0, float(duration)], c_ops=[], options=opt).states[-1]
            return rho

        c_ops = self.noise.collapse_ops() if self.noise.has_collapse() else []
        leak = self.noise.leak_generators() if self.noise.has_leakage() else []
        gate_err = self.noise.has_gate_error()
        mask = self._dressed_mask(H_list)
        kicks = self._kick_mask(H_list)

        for (H, duration), dressed, is_kick in zip(H_list, mask, kicks):
            if duration == 0:
                continue
            H_qobj = H.to_qutip_qobj()
            # The kick is a gate, not a dressed evolution: unless kick_dephases,
            # the dressing-level collapse operators do not act during it (only the
            # separately-applied gate error does).
            seg_c_ops = c_ops if (dressed or self.kick_dephases) else []
            if leak and dressed:
                # conditional no-jump generator: H_eff = H − (i/2) Σ Γ|1><1|.
                # normalize_output=False keeps Tr(ρ)<1 = survival probability
                # (post-selection); the default would hide the leakage by
                # renormalizing the state.
                H_eff = H_qobj - 0.5j * sum(leak)
                L = -1j * (qp.spre(H_eff) - qp.spost(H_eff.dag()))
                for c in c_ops:
                    L += qp.lindblad_dissipator(c)
                rho = qp.mesolve(L, rho, [0.0, float(duration)],
                                 options={**opt, "normalize_output": False}).states[-1]
            else:
                rho = qp.mesolve(H_qobj, rho, [0.0, float(duration)],
                                 c_ops=seg_c_ops, options=opt).states[-1]
            # kick gate error: a discrete Z-type channel on the kicked qubits
            if is_kick and gate_err:
                rho = self.noise.apply_gate_error(rho, self._kick_support(H))
        return rho

    def make_probs_fn(self, psi0):
        """Return probsfn(H_list) -> np.ndarray of computational-basis probabilities
        p_k = <k|ρ|k>/Tr(ρ) (length 2^n).  This is the Z-basis readout distribution:
        one hardware shot draws a bitstring ~ p, from which every DIAGONAL observable
        (all Z_iZ_j parities simultaneously) is read off.  Correct finite-shot model for
        summed diagonal costs like Σ_i Z_iZ_{i+1}, which a single-[-1,1] binomial cannot
        represent (the sum ranges over [-P, P]).  Post-selection: divide by Tr(ρ).
        """
        import numpy as _np

        def probsfn(H_list):
            rho = self.run_sequence(H_list, psi0)
            d = _np.real(rho.full().diagonal())
            tr = float(rho.tr().real)
            if abs(tr) > 1e-15:
                d = d / tr
            d = _np.clip(d, 0.0, None)
            s = d.sum()
            return d / s if s > 0 else d
        return probsfn

    def make_expectation_fn(self, psi0, observable):
        """Return expfn(H_list) -> float = post-selected ⟨O⟩ = Tr(O·ρ)/Tr(ρ).

        Under post-selected leakage Tr(ρ) < 1 (survival probability); dividing by
        it conditions on the atom being found in {|0>, |1>}, exactly as hardware
        post-selection does.  Without leakage Tr(ρ)=1 and this is the plain ⟨O⟩.
        """
        def expfn(H_list):
            rho = self.run_sequence(H_list, psi0)
            tr = float(rho.tr().real)
            val = float(qp.expect(observable, rho).real)
            return val / tr if abs(tr) > 1e-15 else val
        return expfn
