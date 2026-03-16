#we use this generator to generate the ensembles of evolution hamiltonians for differentiation
#we will need a classical sampler for timing, and symbolic evaluation of parameterized Hamiltonians
#input: A (list of) Hamiltonian with parameterized, number of bins(samples), number of repetitions(giving to machine) 
#output: A list of Hamitonians, corresponding Hj
from collections.abc import MutableMapping
from copy import copy, deepcopy

from simuq.config import Config
from simuq.expression import Expression

import random
#not sure what class tos use tho
#we write Gate into Hamiltonian
#Gate-> assume Qsim 


#dressing: |0> |1~>=|1>+beta|r>

#gate: |0>, |1>, + |r>

#lets just assume there is one parameterized H first
def observable_program_generator(parametrized_H, T, n_sample, n_repetition, diff_var, value, pitime_dict=None):
    
    returnlist = []
    
    u_grad_dict = parametrized_H.take_diff_coef(diff_var)
    Hj_list = list(u_grad_dict.keys())

    evaluated_H = parametrized_H.set_parameterizedHam({diff_var:value})
    tau_list = random.randrange(0, 1, n_sample)*T
    for Hj in Hj_list:
        try:
            pitime = pitime_dict[Hj]
        except:
            pitime = 1
        H_list = []
        u_grad = u_grad_dict[Hj]
        evaluated_ugrad = u_grad.subs(diff_var, value)
        for tau in tau_list:
            for sgn in [-1,1]:
                H_list.append(evaluated_H,tau)
                H_list.append(Hj, 1+sgn*3/4*pitime)
                H_list.append(evaluated_H,T-tau)
        info_list = [H_list,evaluated_ugrad,n_repetition]
        returnlist.append(info_list)
    
    # a list of lists for which every list contains a H_list, evaluated_ugrad and n_rep
    return returnlist

