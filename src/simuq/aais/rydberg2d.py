import numpy as np

from simuq.environment import Qubit
from simuq.expression import Expression
from simuq.hamiltonian import hlist_sum
from simuq.qmachine import QMachine


C_6 = 862690 * 2.0 * np.pi
hbar = 1

def generate_qmachine(n=2, inits=None):
    rydberg = QMachine()
    link = [(i, j) for i in range(n) for j in range(i + 1, n)]
    q = [Qubit(rydberg) for i in range(n)]

    l = (C_6 / 4) ** (1.0 / 6) / (2 - 2 * np.cos(2 * np.pi / n)) ** 0.5

    #add distance as a global variable. In this config all rydberg atoms are fixed at the beginning of an experiment
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

    #in dressing pic or any gg qubit encoding sys_ham is none
    hlist = []
    # for i in range(n):
    #     for j in range(i):
    #         dsqr = (x[i][0] - x[j][0]) ** 2 + (x[i][1] - x[j][1]) ** 2
    #         #hlist.append((C_6 / (dsqr**3)) * noper[i] * noper[j])
    #         # hlist.append((C_6 / (dsqr**3)) * Z[i]Z[j])
    sys_h = hlist_sum(hlist)

    rydberg.set_sys_ham(sys_h)

    for i in range(n):
        L = rydberg.add_signal_line()
        ins = L.add_instruction("native", f"Detuning of site {i}")
        d = ins.add_local_variable()
        ins.set_ham(-d * noper[i])

    for i in range(n):
        L = rydberg.add_signal_line()
        ins = L.add_instruction("native", f"Rabi of site {i}")
        o = ins.add_local_variable()
        p = ins.add_local_variable()
        ins.set_ham(o / 2 * (Expression.cos(p) * q[i].X - Expression.sin(p) * q[i].Y))





#we add the dressing potential
#Jij (depend on C6 and r between atom i and j)
#we have
#Sum(i<j) Jij (Szi+1/2)(Szj+1/2) = Jij SziSzj + Jij*1/4(ommited) + 1/2JijSzi + 1/2JijSzj
#=Sum(i<j) 1/4*Jij*SziSzj + Sum(i)Sum(i neq j) 1/2*Jij*Szi 
#or we can just express in terms of noper
    L = rydberg.add_signal_line()
    ins = L.add_instruction("native", f"dressing gloabl potential")
    #d = ins.add_local_variable()
    o = ins.add_local_variable()
    hlist = []
    for i in range(n):
        for j in range(i):
            #rc = (C_6/(2*hbar*d))**(1/6)
            #J0 = -hbar*o**4/(8*d**3)
            dsqr = (x[i][0] - x[j][0]) ** 2 + (x[i][1] - x[j][1]) ** 2
            #Jij = J0/(1+(dsqr/rc)**6)
            #hlist.append(Jij * noper[i] * noper[j]) 
            hlist.append(o* C_6 / (dsqr**6) * noper[i] * noper[j]) 
    dressing_h = hlist_sum(hlist)
    ins.set_ham(dressing_h)

    for q0, q1 in link:
        L = rydberg.add_signal_line()

        ins = L.add_instruction("derived", "c{}{}_zz".format(q0, q1))
        o = ins.add_local_variable()
        ins.set_ham(
            o * q[q0].Z * q[q1].Z
        )

    return rydberg
