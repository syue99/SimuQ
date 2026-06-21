"""
noise_model.py — error models for the QuTiP gradient benchmark.

All noise is realized as Lindblad collapse operators integrated by mesolve over
each segment's real duration.  This is the right tool for small-n, fully-non-
Clifford analog evolution (Stim / stabilizer-rank simulators don't apply — see
project notes), and it makes every channel automatically DURATION-SCALED, so the
FD vs PSR comparison stays fair (a PSR branch's short kick segment accrues
proportionally tiny error, never a full discrete channel).

Two channel families, freely combined:

1. T1 / T2 — relaxation + dephasing (the standard coherence-time parameterization)
   - relaxation:  c = sqrt(1/T1) · σ⁻_i        (σ⁻ = |0><1|, decay |1>→|0>)
   - dephasing:   c = sqrt(γφ/2) · Z_i,  γφ = 1/T2 - 1/(2·T1)   (needs T2 ≤ 2·T1)
     gives coherence decay 1/T2 = 1/(2T1) + γφ.

2. Pauli error rates — a coarse "x% X, (1-x)% Z" style budget as Poisson rates:
   - Pauli P at rate Λ_P (per μs):  c = sqrt(Λ_P) · P_i
   - e.g. "x:(1-x) X:Z at total rate Λ"  ->  pauli_rates={"X": x*Λ, "Z": (1-x)*Λ}.
   Note a Z error rate Λ_Z induces coherence decay 2·Λ_Z (each Z fully dephases),
   a different convention from the T2 parameterization above — pick one.

Units: rates are per μs (the time unit of segment durations throughout DiffSimuQ).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import sys
sys.path.insert(0, "/Users/syue99/research/SimuQ/src/")

import qutip as qp


def _embed(op, i, n):
    """Place single-qubit `op` on qubit i in an n-qubit tensor space."""
    return qp.tensor([op if k == i else qp.qeye(2) for k in range(n)])


_PAULI = {"X": qp.sigmax, "Y": qp.sigmay, "Z": qp.sigmaz}


@dataclass
class NoiseModel:
    """Noise specification for NoisyQuTiPRunner (all channels via mesolve c_ops).

    n_qubits    : int
    T1          : float | None — amplitude-damping time (μs); None = no relaxation
    T2          : float | None — total coherence time (μs);  None = no dephasing
    pauli_rates : dict | None — per-qubit Pauli error rates (per μs), e.g.
                  {"X": 0.02, "Z": 0.05}.  None / missing Paulis = rate 0.

    With everything None the model is noiseless and collapse_ops() returns [],
    so the runner does unitary evolution (reproducing the sesolve path).
    """
    n_qubits: int
    T1: Optional[float] = None
    T2: Optional[float] = None
    pauli_rates: Optional[Dict[str, float]] = None
    leakage_rate: Optional[float] = None

    def __post_init__(self):
        if self.pauli_rates is not None:
            for P in self.pauli_rates:
                if P not in _PAULI:
                    raise ValueError(f"pauli_rates key {P!r} must be one of "
                                     f"{list(_PAULI)}")
            if any(r < 0 for r in self.pauli_rates.values()):
                raise ValueError("pauli_rates must be non-negative")
        if self.leakage_rate is not None and self.leakage_rate < 0:
            raise ValueError("leakage_rate must be non-negative")

    # ── trace-preserving channels (T1 σ⁻ / T2 dephasing / Pauli) ──────────────
    def has_collapse(self) -> bool:
        return (self.T1 is not None or self.T2 is not None
                or bool(self.pauli_rates))

    # ── post-selected leakage (loss OUT of the qubit subspace) ────────────────
    def has_leakage(self) -> bool:
        return bool(self.leakage_rate)

    def has_noise(self) -> bool:
        return self.has_collapse() or self.has_leakage()

    def leak_generators(self) -> List["qp.Qobj"]:
        """Per-qubit non-Hermitian leakage generators Γ·|1><1|_i.

        Models loss from the dressed |1> state out of the computational subspace
        (atom decays to a dark ground sublevel → discarded at readout).  Used as
        the anti-Hermitian part of a CONDITIONAL (no-jump) evolution:
            H_eff = H − (i/2) Σ_i Γ·|1><1|_i,
        which post-selects on "no leakage" and renormalizes — exactly hardware
        post-selection on the atom being found in {|0>, |1>}.  Only |1> leaks
        because only |1> is dressed.
        """
        if not self.leakage_rate:
            return []
        n = self.n_qubits
        n_proj = qp.basis(2, 1) * qp.basis(2, 1).dag()      # |1><1| = (I−Z)/2
        return [float(self.leakage_rate) * _embed(n_proj, i, n)
                for i in range(n)]

    def collapse_ops(self) -> List["qp.Qobj"]:
        """All Lindblad collapse operators for this model (T1/T2 + Pauli rates)."""
        c_ops = []
        n = self.n_qubits

        # T1 relaxation
        gamma1 = (1.0 / self.T1) if self.T1 else 0.0
        if gamma1 > 0:
            for i in range(n):
                c_ops.append((gamma1 ** 0.5) * _embed(qp.sigmam(), i, n))

        # T2 pure dephasing
        if self.T2:
            inv_T2 = 1.0 / self.T2
            inv_2T1 = (1.0 / (2.0 * self.T1)) if self.T1 else 0.0
            gamma_phi = inv_T2 - inv_2T1
            if gamma_phi < -1e-12:
                raise ValueError(
                    f"Unphysical T2 > 2·T1 (T1={self.T1}, T2={self.T2}); "
                    f"pure-dephasing rate would be negative.")
            gamma_phi = max(gamma_phi, 0.0)
            if gamma_phi > 0:
                for i in range(n):
                    c_ops.append(((gamma_phi / 2.0) ** 0.5)
                                 * _embed(qp.sigmaz(), i, n))

        # Pauli error rates  c = sqrt(Λ_P) · P
        if self.pauli_rates:
            for P, rate in self.pauli_rates.items():
                if rate > 0:
                    for i in range(n):
                        c_ops.append((float(rate) ** 0.5)
                                     * _embed(_PAULI[P](), i, n))

        return c_ops
