#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 19:38:54 2018

@author: brian
"""

import numpy as np
from mps import MPS
from mps import MPO_from_local_matrix
from mps import MPS_from_product_states



###check product state construction

def test_product_state():
    L=10
    sps = 2
    D=1
    
    psiloc = np.array([0.0, 1.0])
    psi = MPS_from_product_states([psiloc]*L)
    print(psi.norm())
    
    O = np.array([[1, 0], [0, -1]])
    mpo = MPO_from_local_matrix(O, 4)
    
    psi.gauge(4)
    print(mpo._expt_value_local(psi))

### check imaginary-time trotter
    
from trotter import get_pauli_exp, sigmax
U = get_pauli_exp(1.0j, 'X')

L=10
D=5
sps=2
psi = MPS(L, sps=sps)
psi.init_random(D)
i=2
psi.normalize(i)
from trotter import _act_1qubit_gate
nstep = 100
sigma_x = MPO_from_local_matrix(sigmax(), i)
for _ in range(nstep):
    _act_1qubit_gate(U, psi, i)
    psi._normalize_site(i)
    print(sigma_x._expt_value_local(psi))
    
    
    