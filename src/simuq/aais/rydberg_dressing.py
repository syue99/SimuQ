import numpy as np

from simuq.environment import Qubit
from simuq.expression import Expression
from simuq.hamiltonian import hlist_sum
from simuq.qmachine import QMachine


C_6 = 862690 * 2.0 * np.pi


#We will use raman pulses, MW pulses and dressing pulese for the ins. Hamiltonian
#Hsys will be 0
def generate_qmachine(n=3, inits=None):
    rydberg = QMachine()

    q = [Qubit(rydberg) for i in range(n)]

    l = (C_6 / 4) ** (1.0 / 6) / (2 - 2 * np.cos(2 * np.pi / n)) ** 0.5

    if inits is None:
        x = [(0.0, 0.0)] + [
            (
                rydberg.add_global_variable(init_value=l * (np.cos(i * 2 * np.pi / n) - 1)),
                rydberg.add_global_variable(init_value=l * np.sin(i * 2 * np.pi / n)),
            )
            for i in range(1, n)
        ]
    else:
        x = [(0.0, 0.0)] + [
            (
                rydberg.add_global_variable(init_value=inits[i][0]),
                rydberg.add_global_variable(init_value=inits[i][1]),
            )
            for i in range(1, n)
        ]
    noper = [(q[i].I - q[i].Z) / 2 for i in range(n)]

#We will turn this hlist to the dressing Hamiltonian
#missing def of hbar
    hlist = []
    for i in range(n):
        L = rydberg.add_signal_line()
        ins = L.add_instruction("native", f"dressing of site {i}")
        o = ins.add_local_variable()
        d = ins.add_local_variable()
        for j in range(i):
            dsqr = (x[i][0] - x[j][0]) ** 2 + (x[i][1] - x[j][1]) ** 2
            j_0 = -hbar*o**4/(8*d**3)
            r_c_6 = C_6/(2*hbar*d)
            hlist.append((j_0 / 1+(dsqr**6/r_c_6)) * noper[i] * noper[j])
    dressing_h = hlist_sum(hlist)
    ins.set_ham(dressing_h)

#detuning for th microwave pulse
    for i in range(n):
        L = rydberg.add_signal_line()
        ins = L.add_instruction("native", f"Detuning of site {i}")
        d = ins.add_local_variable()
        ins.set_ham(-d * noper[i])
#microwave pulse driving
    for i in range(n):
        L = rydberg.add_signal_line()
        ins = L.add_instruction("native", f"Rabi of site {i}")
        o = ins.add_local_variable()
        p = ins.add_local_variable()
        ins.set_ham(o / 2 * (Expression.cos(p) * q[i].X - Expression.sin(p) * q[i].Y))
#raman pulse 
    for i in range(n):
        L = rydberg.add_signal_line()
        ins = L.add_instruction("native", f"Raman of site {i}")
        

    return rydberg
