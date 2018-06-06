#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 13:25:46 2018

@author: brian

tools for implementing trotterized time evolution.

For sps=2 (spin-1/2) everything is written the the z-basis, all ops are in the Pauli basis.

"""

import numpy as np

#for a particular local dimension, the recognized operators
_LOC_OPS = {2:['X', 'Y', 'Z', 'N', 'I']}

###pauli matrices
def sigmaz():
    return np.array([[1,0], [0, -1]])

def sigmax():
    return np.array([[0, 1], [1, 0]])
    
def sigmay():
    return np.array([[0, -1j], [1j, 0]])

def n():
    return 0.5 * (np.identity(2) + sigmaz())

def identity():
    return np.identity(2)

def get_pauli_mat(opstr):
    """ Returns 2x2 pauli matrix"""
    _pauli_gen = {'X': sigmax, 'Y': sigmay, 'Z': sigmaz, 'N':n, 'I': identity}
    return _pauli_gen[opstr.upper()]()

def get_pauli_exp(t, opstr):
    """ returns matrix exponential 
             exp(i t sigma)
         where sigma is specified by opstr as one of the Paulis.
         
         Note the i.
         """
    return np.cos(t) * np.identity(2) + 1j * np.sin(t) * get_pauli_mat(opstr)

def get_tensor_prod(ops):
    op = ops[0]
    for i in range(1, len(ops)):
        op = np.tensordot(op, ops[i])
    return op

def get_pauli_prod(opstr_list):
    """ Returns tensor product of Paulis of the type and order specified by opstr_list (e.g. ['X', 'X', 'Z']) returns X tensor X tensor Z.
         Note that they are 'adjacent' in the tensor product."""
    ops = [get_pauli_mat(o) for o in opstr_list]
    return get_tensor_prod(ops)

def get_exp_pauli_prod(t, opstr_list):
    """ Returns exp(i t Sigma1...SigmaN), where Sigmai are the Pauli ops specified in opstr_list"""
    exp_ops = [get_pauli_exp(t, o) for o in opstr_list]
    return get_tensor_prod(exp_ops)


def _act_1qubit_gate(U, psi, i):
    """Apply a single-qubit gate to MPS psi at site i.
       U = (sps) x (sps) array. For now assumed to be unitary.
       psi = (pure ) MPS state.
       
       Updates the state in-place.
       """
       
    A = psi.get_site(i)
    psi.set_site(i, np.tensordot(U, A, axes=([1], [0])))
    
  
    
def _act_2qubit_local_gate(U, psi, i):
    """ Apply two-qubit gate to sites i, i+1 of MPS psi.
        U = 4-index gate, each axis of dimension sps.
        psi = MPS
        
        Updates the state in-place.
        Note that after applying the gate, left-normalization if any is generally not preserved."""
    
    A1, A2 = psi.get_site(i), psi.get_site(i+1)
    sps = A1.shape[0]
    D1, D2 = A1.shape[1], A2.shape[2]
    AA= np.tensordot(A1, A2, axes=([2], [1])) ##sps, D1, sps, D2
    blob = np.tensordot(U, AA, axes =([2, 3], [0,2]))  # sps, sps, D1,D2
    blob = np.swapaxes(blob, 1, 2).reshape((sps*D1, sps*D2)) 
    u,s,v = np.linalg.svd(blob, full_matrices=False)
    k = s.shape[0]
    A1_tilde = u.reshape((sps, D1, k))
    A2_tilde = np.dot(np.diag(s), v).reshape((sps, k, D2))
    psi.set_sites([i, i+1], [A1_tilde, A2_tilde])
        
    

class LocalHamiltonian(object):
    """ A hamiltonian for a 1d system which is never fully stored in memory.
        Assumed to be a quasi-local; the local terms can be obtained in matrix representation on-demand."""
        
    def __init__(self, L, sps):
        """ L = chain length
            sps = states per site.
            """
        self.L = L
        self.sps = sps
        self._terms_by_k = dict()
        
    def get_klocal_terms(self, k):
        return self._terms_by_k[k]
        
    


    
    
    
    