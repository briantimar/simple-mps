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

from trotter import ExpPauliHamiltonian

import sys
#sys.path.append("/home/brian/Documents/ryd-theory-code/ryd-theory/python_code/")
sys.path.append("/Users/btimar/Documents/ryd-theory-code/python_code/")
from ryd_base import make_1d_TFI_static

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

def test_imag_onesite():
    U = get_pauli_exp(-1.0j, 'X')
    
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
        

from trotter import TrotterLayers, sigmaz, identity

L=10
D=5
sps=2
J=0.1000
psi = MPS(L,sps,Dmax=D)
dtype=np.complex128
psi.init_random(dtype=dtype)
psi.left_normalize_full()
Omega =1.0
static = make_1d_TFI_static(J, Omega, L, bc='open')
#static =[ ['X', [[-Omega, i] for i in range(L)]]]

T=-20.0j/1.0
nlayer=50
expH = ExpPauliHamiltonian(static)
expH.set_layer(['X'])
expH.set_layer(['ZZ'])
#expH.set_layer(['X'])
tl = TrotterLayers(expH)
tl.set_evolve_time(T)
tl.set_num_layers(nlayer)
tl.run_evolution(psi, renormalize=True)
###
psi.left_normalize_full()

for i in range(L):
    psi.gauge(i)
    z = MPO_from_local_matrix(sigmaz(), i)
    x = MPO_from_local_matrix(sigmax(), i)
    I = MPO_from_local_matrix(identity(),i)
    print(z._expt_value_local(psi), x._expt_value_local(psi), I._expt_value_local(psi))
#    
    