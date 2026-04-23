"""
verify_compilation.py — Round-trip verification of the tweezer compilation.

Reconstructs Hamiltonians from PulseLedger meta-parameters and compares
the resulting gradient against the direct QuTiP simulation (ground truth).

For dressing segments the stored TIHamiltonian is used directly.
For detuning, Rabi, and ZZ segments the Hamiltonian is rebuilt from
meta-parameters (amplitude, phase, target qubits, positions).

Usage:
    grad_recon = verify_compilation(
        programs, pulse_ledgers, n_qubits, psi0, observable, T
    )
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simuq.hamiltonian import TIHamiltonian, productHamiltonian
from qutip_sequential import QuTiPSequentialRunner
from combine_gradient import combine_gradient_results


def _reconstruct_H_from_entry(entry, sites_type, sites_name):
    """
    Reconstruct a TIHamiltonian from a single PulseLedger play entry.

    Parameters
    ----------
    entry      : LedgerEntry with op_type="play"
    sites_type : list of str — qubit type per site (from a QSystem)
    sites_name : list of str — qubit name per site

    Returns
    -------
    TIHamiltonian
    """
    kind = entry.channel_kind

    if kind in ("dressing", "kick"):
        # Use the stored Hamiltonian directly
        if entry.hamiltonian is None:
            raise ValueError(
                f"{kind} entry has no stored Hamiltonian — cannot reconstruct."
            )
        return entry.hamiltonian

    if kind == "detuning":
        # rydberg2d convention: H = -d * (I-Z)/2 = -d/2 + d/2 * Z
        # amplitude stores d (raw pulse amplitude)
        site = entry.target_qubits[0]
        I_H = TIHamiltonian.identity(sites_type, sites_name)
        Z_H = TIHamiltonian.op(sites_type, sites_name, site, "Z")
        return (entry.amplitude / 2.0) * ((-1.0) * I_H + Z_H)

    if kind == "rabi":
        # rydberg2d convention: H = Ω/2 * (cos(φ)*X - sin(φ)*Y)
        # amplitude stores Ω (raw pulse amplitude)
        site = entry.target_qubits[0]
        phase = entry.phase if entry.phase is not None else 0.0
        Hx = TIHamiltonian.op(sites_type, sites_name, site, "X")
        Hy = TIHamiltonian.op(sites_type, sites_name, site, "Y")
        return (entry.amplitude / 2.0) * (np.cos(phase) * Hx + (-np.sin(phase)) * Hy)

    if kind == "zz":
        # H = J * Z_q0 Z_q1  where J = amplitude (stored by _cz_ops)
        q0, q1 = entry.target_qubits[0], entry.target_qubits[1]
        prod = productHamiltonian()
        prod[q0] = "Z"
        prod[q1] = "Z"
        return TIHamiltonian(sites_type, sites_name, [(prod, entry.amplitude)])

    raise ValueError(f"Unknown channel_kind {kind!r} in ledger entry.")


def _ledger_to_H_list(ledger, sites_type, sites_name):
    """
    Convert a PulseLedger into a list of (TIHamiltonian, duration) pairs
    suitable for QuTiPSequentialRunner.

    Only "play" entries produce Hamiltonian segments. AOD/delay entries are
    skipped (they represent transport, not physics evolution).

    When consecutive play entries share the same duration (i.e., they come
    from the same box/segment), their Hamiltonians are summed into a single
    segment — matching how map_evaluated_H emits multiple instructions for
    one time segment.

    Parameters
    ----------
    ledger     : PulseLedger
    sites_type : list of str
    sites_name : list of str

    Returns
    -------
    list of [TIHamiltonian, duration]
    """
    H_list = []
    play_entries = ledger.play_entries()

    # Group consecutive play entries by duration (same time window = same segment).
    # Sum ALL entries in a group: each contributes its own channel Hamiltonian.
    # - Dressing/ZZ entries have stored TIHamiltonian (solver's decomposition)
    # - Detuning/rabi entries are reconstructed from meta-params
    # This honestly tests whether the solver's channel decomposition reproduces
    # the target Hamiltonian.
    i = 0
    while i < len(play_entries):
        dur = play_entries[i].duration
        # Collect all entries with the same duration (same segment)
        group = [play_entries[i]]
        j = i + 1
        while j < len(play_entries) and abs(play_entries[j].duration - dur) < 1e-12:
            group.append(play_entries[j])
            j += 1

        # Check if this group has a "kick" entry — if so, use its stored H
        # exclusively (the other entries are hardware ops, not physics).
        kick_entry = None
        for entry in group:
            if entry.channel_kind == "kick":
                kick_entry = entry
                break

        if kick_entry is not None:
            H_list.append([kick_entry.hamiltonian, dur])
        else:
            # Sum all entries in the group
            H = _reconstruct_H_from_entry(group[0], sites_type, sites_name)
            for entry in group[1:]:
                H = H + _reconstruct_H_from_entry(entry, sites_type, sites_name)
            H_list.append([H, dur])

        i = j

    return H_list


def _infer_sites(pulse_ledgers, n_qubits):
    """Infer sites_type/sites_name from ledger entries or use defaults."""
    for prog_ledgers in pulse_ledgers:
        for ledger in prog_ledgers:
            for entry in ledger.play_entries():
                if entry.hamiltonian is not None:
                    return entry.hamiltonian.sites_type, entry.hamiltonian.sites_name
    return ["qubit"] * n_qubits, [f"q{i}" for i in range(n_qubits)]


def _remove_identity(H_qobj):
    """Remove the identity component from a Hamiltonian Qobj.

    The identity part is a global phase shift and doesn't affect dynamics.
    Removing it gives a physically meaningful norm comparison.
    """
    import qutip as qp
    dim = H_qobj.shape[0]
    I = qp.qeye(H_qobj.dims[0])
    id_coeff = (H_qobj * I).tr() / dim  # <H, I> / dim
    return H_qobj - id_coeff * I


def norm_check(programs, pulse_ledgers, n_qubits):
    """
    Per-segment operator norm comparison: ||H_compiled - H_target||.

    Identity components (global phase) are subtracted before comparison,
    since they don't affect dynamics.

    Returns a list of dicts, one per (program, branch, segment):
        {"prog": int, "branch": int, "seg": int,
         "norm_diff": float, "dur_orig": float, "dur_recon": float}
    """
    sites_type, sites_name = _infer_sites(pulse_ledgers, n_qubits)
    results = []

    for prog_idx, (H_tot_list, ugrad, n_rep) in enumerate(programs):
        for branch_idx in range(len(H_tot_list)):
            orig_H_list = H_tot_list[branch_idx]
            ledger = pulse_ledgers[prog_idx][branch_idx]
            recon_H_list = _ledger_to_H_list(ledger, sites_type, sites_name)

            n_segs = min(len(orig_H_list), len(recon_H_list))
            for seg_idx in range(n_segs):
                H_o = _remove_identity(orig_H_list[seg_idx][0].to_qutip_qobj())
                H_r = _remove_identity(recon_H_list[seg_idx][0].to_qutip_qobj())
                results.append({
                    "prog": prog_idx,
                    "branch": branch_idx,
                    "seg": seg_idx,
                    "norm_diff": float((H_o - H_r).norm()),
                    "dur_orig": float(orig_H_list[seg_idx][1]),
                    "dur_recon": float(recon_H_list[seg_idx][1]),
                })
    return results


def verify_compilation(programs, pulse_ledgers, n_qubits, psi0, observable, T,
                       t_solver=None):
    """
    Compute gradient from PulseLedger-reconstructed Hamiltonians and
    per-segment operator norm diagnostics.

    Parameters
    ----------
    programs       : list — output of observable_program_generator()
    pulse_ledgers  : list of list of PulseLedger — from diffQCProvider._pulse_ledgers
    n_qubits       : int
    psi0           : QuTiP ket
    observable     : QuTiP Qobj
    T              : float — total evolution time used in PSR
    t_solver       : float | None — solver's box duration (informational)

    Returns
    -------
    dict with keys:
        "gradient"   : float — reconstructed gradient from compiled H
        "norm_diffs" : list of dicts — per-segment ||H_compiled - H_target||
    """
    sites_type, sites_name = _infer_sites(pulse_ledgers, n_qubits)

    runner = QuTiPSequentialRunner(n_qubits=n_qubits)

    # Rebuild programs with reconstructed H_lists from ledgers
    reconstructed_programs = []
    for prog_idx, (H_tot_list, ugrad, n_rep) in enumerate(programs):
        recon_H_tot_list = []
        for branch_idx in range(len(H_tot_list)):
            ledger = pulse_ledgers[prog_idx][branch_idx]
            H_list = _ledger_to_H_list(ledger, sites_type, sites_name)
            recon_H_tot_list.append(H_list)
        reconstructed_programs.append([recon_H_tot_list, ugrad, n_rep])

    expfn = runner.make_expectation_fn(psi0, observable)
    grad_recon = combine_gradient_results(reconstructed_programs, expfn, T)

    # Per-segment norm diagnostics
    norms = norm_check(programs, pulse_ledgers, n_qubits)

    return {
        "gradient": float(grad_recon),
        "norm_diffs": norms,
    }
