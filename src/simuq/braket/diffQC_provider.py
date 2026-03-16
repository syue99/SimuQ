import matplotlib.pyplot as plt
import numpy as np

from simuq import _version
from simuq.aais import heisenberg, two_pauli, rydberg1d_global, rydberg2d_global, rydberg2d
from simuq.braket.braket_ionq_transpiler import BraketIonQTranspiler, BraketIonQTranspiler_2Pauli
from simuq.braket.braket_rydberg_transpiler import BraketRydbergTranspiler
from simuq.provider import BaseProvider
from simuq.solver import generate_as

from braket.circuits import Circuit
#YS: I will learn from this code with comments

class diffQCProvider(BaseProvider):
    def __init__(self):
        self.backend_aais = dict()
        self.backend_aais[("quera", "Aquila")] = [
            "rydberg1d_global",
            "rydberg2d_global",
            "rydberg2d"
        ]

        super().__init__()

    def supported_backends(self):
        for comp, dev in self.backend_aais.keys():
            print(
                f"Hardware provider: {comp}  -  Device name: {dev}  -  AAIS supports: {self.backend_aais[(comp, dev)]}"
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
            print(provider)
            print(device)
            raise Exception("Not supported hardware provider or device.")
        if aais not in self.backend_aais[(provider, device)]:
            raise Exception("Not supported AAIS on this device.")

        self.qs_names = qs.print_sites()

        self.provider = provider
        self.device = device

        if self.provider == "quera":
            nsite = qs.num_sites
#will generate a machine with all avaliable parameterzed instruction sets; also initiate a transpiler for pulses, input parameter is dimension
#transpiler used to generate pulse sequences when corresponding parameters are calculauted (e.g. phase, amp of the global laser)
            if aais == "rydberg1d_global":
                transpiler = BraketRydbergTranspiler(1)
                mach = rydberg1d_global.generate_qmachine(nsite)
            elif aais == "rydberg2d_global":
                transpiler = BraketRydbergTranspiler(2)
                mach = rydberg2d_global.generate_qmachine(nsite)
            elif aais == "rydberg2d":
                transpiler = BraketRydbergTranspiler(2)
                mach = rydberg2d.generate_qmachine(nsite)
            else:
                raise NotImplementedError

            if state_prep is None:
                state_prep = {"times": [], "omega": [], "delta": [], "phi": []}

            if meas_prep is not None:
                raise Exception(
                    "Currently SimuQ does not support measurement preparation pulses for QuEra devices."
                )

            layout, sol_gvars, boxes, edges = generate_as(
                qs,
                mach,
                trotter_num=1,
                solver="least_squares",
                solver_args={"tol": tol, "time_penalty": 1},
                override_layout=[i for i in range(nsite)],
                verbose=verbose,
            )
            self.sol = [layout, sol_gvars, boxes, edges]

            # # Only use this when debugging state_preparation pulses
            # if no_main_body:
            #     for j, (b, t) in enumerate(boxes):
            #         boxes[j] = (b, 0.01)
            # self.ahs_prog = transpiler.transpile(
            #     sol_gvars, boxes, edges, state_prep=state_prep, verbose=verbose
            # )
            # self.prog = self.ahs_prog
            # self.layout = layout


            
            if trotter_mode == "random":
                trotter_args = {"num": trotter_num, "order": 1, "sequential": False, "randomized": True}
            else:
                trotter_args = {"num": trotter_num, "order": trotter_mode, "sequential": True, "randomized": False}

            layout, sol_gvars, boxes, edges = generate_as(
                qs,
                mach,
                trotter_args=trotter_args,
                solver="least_squares",
                solver_args={"tol": tol},
                override_layout=[i for i in range(qs.num_sites)],
                verbose=verbose,
            )
            self.prog = [qs.num_sites, sol_gvars, boxes, edges, trotter_args]
            
            #TI Hamiltonian, just two qubit 

            #self.prog = transpiler.transpile(qs.num_sites, sol_gvars, boxes, edges)#, trotter_args=trotter_args)

            #if state_prep is not None:
            #    self.prog = state_prep.copy().add(self.prog)

            #if meas_prep is not None:
            #    self.prog.add(meas_prep)

            #self.prog = Circuit().add_verbatim_box(self.prog.braket_circuit)

            #self.layout = layout
            #self.qs_names = qs.print_sites()







    def run(self, shots, on_simulator=False, verbose=0):
        pass

    def results(self, task_arn=None, verbose=0):
        pass


