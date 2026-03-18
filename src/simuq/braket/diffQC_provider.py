"""
diffQC_provider.py — DiffSimuQ provider for Rydberg tweezer hardware.

Workflow
--------
1. compile(qs, ...)           — solve machine (generate_as), store boxes / sol_gvars
2. run(programs, obs, T, ...) — execute gradient branches (qutip or hardware backend)
3. results()                  — return gradient estimate

Backend "qutip"
    Uses QuTiPSequentialRunner directly on the H_list segments.  No transport
    layer involved — pure Hamiltonian simulation for validation.

Backend "hardware"
    Uses TweezerMapper to translate H_list branches into schedule op dicts and
    TransportLogs, then calls to_pulsedsl() to emit PulseDSL Play/Delay calls.
    Actual hardware execution and readout are not yet implemented (stub).
"""

import os
import sys

import numpy as np

from simuq.aais import heisenberg, two_pauli, rydberg1d_global, rydberg2d_global, rydberg2d
from simuq.provider import BaseProvider
from simuq.solver import generate_as

# braket is an optional dependency — only needed for the legacy AHS transpiler path
try:
    from simuq.braket.braket_ionq_transpiler import BraketIonQTranspiler, BraketIonQTranspiler_2Pauli
    from simuq.braket.braket_rydberg_transpiler import BraketRydbergTranspiler
    from braket.circuits import Circuit
    _BRAKET_AVAILABLE = True
except ImportError:
    _BRAKET_AVAILABLE = False

# DiffSimuQ differential_computing path
_DIFF_COMPUTING_PATH = os.path.join(os.path.dirname(__file__), "../../../differential_computing")


def to_pulsedsl(ops, channels, aod_ch):
    """
    Translate schedule op dicts (from TweezerMapper.map_hlist) to
    PulseDSL Play / Delay calls.

    Must be called within an active PulseDSL Schedule + Channels context.
    Import is deferred so that importing this module does not open the MMIO
    pipe prematurely.

    Parameters
    ----------
    ops      : list of op dicts — from TweezerMapper.map_hlist()
    channels : PulseDSL ChannelEnv (ch) — native laser/MW channels
    aod_ch   : PulseDSL Channel object — the single AOD transport channel

    Op dict keys
    ------------
    {"op": "aod",   "amplitude": float, "duration_ns": int, "positions": list, ...}
    {"op": "play",  "channel": int, "amplitude": float, "phase": float, "duration": float}
    {"op": "delay", "duration": float}
    """
    pulsedsl_path = "/Users/syue99/research/RISC-Q/PulseDSL/src/DSL/"
    if pulsedsl_path not in sys.path:
        sys.path.insert(0, pulsedsl_path)

    from PulseDSL_py.core import Pulse, Shape
    from PulseDSL_py.ops import Play, Delay as DSLDelay

    for op in ops:
        kind = op["op"]

        # All time fields in op dicts are in μs; PulseDSL Pulse.duration is in ns.
        # Conversion: duration_ns = int(duration_μs * 1000)

        if kind == "aod":
            # AOD transport: sine wave at fixed frequency (amplitude = position proxy)
            pulse = Pulse(
                shape=Shape.Sine,
                amplitude=float(op["amplitude"]),
                duration=int(op["duration"] * 1000),  # μs → ns
            )
            Play(pulse, aod_ch)

        elif kind == "play":
            # Native laser/MW pulse on a specific channel
            pulse = Pulse(
                shape=Shape.Constant,
                amplitude=float(op["amplitude"]),
                phase=float(op["phase"]),
                duration=int(op["duration"] * 1000),  # μs → ns
            )
            Play(pulse, channels[op["channel"]])

        elif kind == "delay":
            # Time advance: emit a zero-amplitude hold on the AOD channel
            DSLDelay(int(op["duration"] * 1000), aod_ch)  # μs → ns


# ── Provider ──────────────────────────────────────────────────────────────────

class diffQCProvider(BaseProvider):
    def __init__(self):
        self.backend_aais = dict()
        self.backend_aais[("quera", "Aquila")] = [
            "rydberg1d_global",
            "rydberg2d_global",
            "rydberg2d",
        ]
        super().__init__()

    def supported_backends(self):
        for comp, dev in self.backend_aais.keys():
            print(
                f"Hardware provider: {comp}  -  Device name: {dev}  -  "
                f"AAIS supports: {self.backend_aais[(comp, dev)]}"
            )

    def compile(
        self,
        qs,
        provider,
        device,
        aais,
        tol=0.01,
        trotter_num=6,
        trotter_mode=1,
        state_prep=None,
        meas_prep=None,
        no_main_body=False,
        verbose=0,
    ):
        if (provider, device) not in self.backend_aais.keys():
            raise Exception("Not supported hardware provider or device.")
        if aais not in self.backend_aais[(provider, device)]:
            raise Exception("Not supported AAIS on this device.")

        self.qs_names = qs.print_sites()
        self.provider = provider
        self.device   = device

        if self.provider == "quera":
            nsite = qs.num_sites

            if aais == "rydberg1d_global":
                if _BRAKET_AVAILABLE:
                    transpiler = BraketRydbergTranspiler(1)
                mach = rydberg1d_global.generate_qmachine(nsite)
            elif aais == "rydberg2d_global":
                if _BRAKET_AVAILABLE:
                    transpiler = BraketRydbergTranspiler(2)
                mach = rydberg2d_global.generate_qmachine(nsite)
            elif aais == "rydberg2d":
                if _BRAKET_AVAILABLE:
                    transpiler = BraketRydbergTranspiler(2)
                mach = rydberg2d.generate_qmachine(nsite)
            else:
                raise NotImplementedError

            if state_prep is None:
                state_prep = {"times": [], "omega": [], "delta": [], "phi": []}

            if meas_prep is not None:
                raise Exception(
                    "Currently SimuQ does not support measurement preparation "
                    "pulses for QuEra devices."
                )

            # First pass: solve with tight time penalty (used for layout / init)
            layout, sol_gvars, boxes, edges = generate_as(
                qs,
                mach,
                trotter_num=1,
                solver="least_squares",
                solver_args={"tol": tol, "time_penalty": 1},
                override_layout=list(range(nsite)),
                verbose=verbose,
            )
            self.sol = [layout, sol_gvars, boxes, edges]

            # Second pass: full solve with Trotter settings
            if trotter_mode == "random":
                trotter_args = {
                    "num": trotter_num, "order": 1,
                    "sequential": False, "randomized": True,
                }
            else:
                trotter_args = {
                    "num": trotter_num, "order": trotter_mode,
                    "sequential": True, "randomized": False,
                }

            layout, sol_gvars, boxes, edges = generate_as(
                qs,
                mach,
                trotter_args=trotter_args,
                solver="least_squares",
                solver_args={"tol": tol},
                override_layout=list(range(qs.num_sites)),
                verbose=verbose,
            )
            # prog stores everything the transport layer needs
            self.prog = [qs.num_sites, sol_gvars, boxes, edges, trotter_args]

    # ── run / results ─────────────────────────────────────────────────────────

    def run(
        self,
        programs,
        observable,
        T,
        psi0=None,
        backend="qutip",
        shots=1,
        verbose=0,
    ):
        """
        Execute DiffSimuQ gradient branches and store results.

        Parameters
        ----------
        programs   : list — output of observable_program_generator()
        observable : QuTiP Qobj (qutip backend) or hardware descriptor
        T          : float — total evolution time used in the gradient formula
        psi0       : QuTiP ket initial state, or None for |00...0>
        backend    : "qutip" (simulation, immediate) |
                     "hardware" (PulseDSL schedule generation; execution stub)
        shots      : int — repetitions per branch (hardware only; ignored for qutip)
        verbose    : int
        """
        if not hasattr(self, "prog"):
            raise RuntimeError("Call compile() before run().")

        n_sites, sol_gvars, boxes, edges, _ = self.prog
        self._T        = T
        self._programs = programs
        self._gradient = None

        if backend == "qutip":
            self._run_qutip(programs, observable, T, psi0, n_sites, verbose)

        elif backend == "hardware":
            self._run_hardware(programs, n_sites, sol_gvars, boxes, shots, verbose)

        else:
            raise ValueError(f"Unknown backend {backend!r}. Use 'qutip' or 'hardware'.")

    def _run_qutip(self, programs, observable, T, psi0, n_sites, verbose):
        """Simulation path: QuTiPSequentialRunner, no transport layer."""
        if _DIFF_COMPUTING_PATH not in sys.path:
            sys.path.insert(0, _DIFF_COMPUTING_PATH)
        from qutip_sequential import QuTiPSequentialRunner
        from combine_gradient import combine_gradient_results

        runner = QuTiPSequentialRunner(n_qubits=n_sites)
        if psi0 is None:
            psi0 = runner.zero_state()

        expfn = runner.make_expectation_fn(psi0, observable)
        self._gradient = combine_gradient_results(programs, expfn, T)
        self._transport_logs = None

        if verbose > 0:
            print(f"[diffQCProvider/qutip] gradient = {self._gradient:.6f}")

    def _run_hardware(self, programs, n_sites, sol_gvars, boxes, shots, verbose):
        """
        Hardware path: build TweezerMapper schedule + TransportLogs.

        For each branch H_list:
          1. map_hlist(H_list) → (ops, log)
          2. Store ops in self._branch_ops for to_pulsedsl() when hardware is ready.
          3. Store log in self._transport_logs for calibration / inspection.

        PulseDSL execution and measurement readout are not yet implemented.
        """
        if _DIFF_COMPUTING_PATH not in sys.path:
            sys.path.insert(0, _DIFF_COMPUTING_PATH)
        from tweezer_mapper import TweezerMapper

        mapper = TweezerMapper(
            n_qubits=n_sites,
            sol_gvars=sol_gvars,
            boxes=boxes,
            ramp_time=0.01,
        )

        self._branch_ops     = []   # (branch_op_list, ugrad, n_rep)
        self._transport_logs = []   # per-program list of TransportLogs

        for H_tot_list, ugrad, n_rep in programs:
            branch_ops   = []
            branch_logs  = []
            for H_list in H_tot_list:
                ops, log = mapper.map_hlist(H_list)
                branch_ops.append(ops)
                branch_logs.append(log)
            self._branch_ops.append((branch_ops, float(ugrad), int(n_rep)))
            self._transport_logs.append(branch_logs)

        n_branches = sum(len(b) for b, _, _ in self._branch_ops)
        if verbose > 0:
            print(
                f"[diffQCProvider/hardware] {n_branches} schedule branches generated."
            )
            dressing = sum(
                len(log.dressing_moves)
                for prog_logs in self._transport_logs
                for log in prog_logs
            )
            cz = sum(
                len(log.cz_moves)
                for prog_logs in self._transport_logs
                for log in prog_logs
            )
            print(
                f"[diffQCProvider/hardware] Transport: "
                f"{dressing} dressing moves, {cz} CZ moves across all branches."
            )
            print(
                "[diffQCProvider/hardware] PulseDSL execution: not yet implemented. "
                "Call to_pulsedsl(ops, channels, aod_ch) per branch to emit pulses."
            )

        # gradient will be populated once hardware readout is available
        self._gradient = None

    def results(self, verbose=0):
        """
        Return the gradient estimate from the last run() call.

        For backend='qutip'    : returns the gradient immediately.
        For backend='hardware' : raises NotImplementedError until hardware
                                 execution and readout are wired up.

        Returns
        -------
        float — gradient estimate ∂<O>/∂θ
        """
        if self._gradient is not None:
            return float(self._gradient)
        raise NotImplementedError(
            "Hardware readout not yet implemented. "
            "Use backend='qutip' for simulation, or wire up hardware "
            "execution before calling results()."
        )

    def transport_summary(self, program_idx=0, branch_idx=0):
        """
        Print the TransportLog for a specific branch (useful for inspection).

        Parameters
        ----------
        program_idx : int — index into programs (gradient term j)
        branch_idx  : int — index into H_tot_list (tau sample × sgn)
        """
        if not hasattr(self, "_transport_logs") or self._transport_logs is None:
            print("No transport logs available. Run with backend='hardware'.")
            return
        try:
            log = self._transport_logs[program_idx][branch_idx]
            print(log.summary())
        except IndexError:
            print(f"No log at program_idx={program_idx}, branch_idx={branch_idx}.")
