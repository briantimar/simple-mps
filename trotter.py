#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 13:25:46 2018

@author: brian

tools for implementing trotterized time evolution.

For sps=2 (spin-1/2) everything is written the the z-basis, all ops are in the Pauli basis.

"""

import numpy as np
from scipy.linalg import expm

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
             exp(-i t sigma)
         where sigma is specified by opstr as a product of adjacent pauli's.
         
         Note the i.
         """
    L=len(opstr)
    kron = get_pauli_kron(opstr)
    U =  expm(-1j*t*kron)
    return U

def get_kron_prod(ops):
    op = ops[0]
    for i in range(1, len(ops)):
        op = np.kron(op, ops[i])
    return op

def get_tensor_prod(ops):
    op = ops[0]
    for i in range(1, len(ops)):
        op = np.tensordot(op, ops[i],axes=0)
    return op



def get_pauli_prod(opstr_list):
    """ Returns tensor product of Paulis of the type and order specified by opstr_list (e.g. ['X', 'X', 'Z']) returns X tensor X tensor Z.
         Note that they are 'adjacent' in the tensor product."""
    ops = [get_pauli_mat(o) for o in opstr_list]
    return get_tensor_prod(ops)

def get_pauli_kron(opstr_list):
    return get_kron_prod([get_pauli_mat(o) for o in opstr_list])


def _act_1qubit_gate(U, psi, i):
    """Apply a single-qubit gate to MPS psi at site i.
       U = (sps) x (sps) array. For now assumed to be unitary.
       psi = (pure ) MPS state.
       
       Updates the state in-place.
       """
       
    A = psi.get_site(i)
    psi.set_site(i,np.tensordot(U, A, axes=([1], [0])))
    
  
    
def _act_2qubit_local_gate(U, psi, i, Dmax=None):
    """ Apply two-qubit gate to sites i, i+1 of MPS psi.
        
        where the first two indices are for the first qubit, the second two for the second, and so on.
        psi = MPS
        
        Updates the state in-place.
        Note that after applying the gate, left-normalization if any is generally not preserved."""
    print(U)
    A1, A2 = psi.get_site(i), psi.get_site(i+1)
    sps = A1.shape[0]
    D1, D2 = A1.shape[1], A2.shape[2]
    AA= np.tensordot(A1, A2, axes=([2], [1])) ##sps, D1, sps, D2
    blob = np.tensordot(U, AA, axes =([1, 3], [0,2]))  # sps, sps, D1,D2
    blob = np.swapaxes(blob, 1, 2).reshape((sps*D1, sps*D2)) 
    u,s,v = np.linalg.svd(blob, full_matrices=False)
    k = s.shape[0]
    trunc_error = None
    if Dmax is not None:
        k = min(s.shape[0], Dmax)
        trunc_error = None if k==s.shape[0] else s[k]
        u=u[:, :k]
        s=s[:k]
        v=v[:k,:]
    A1_tilde = u.reshape((sps, D1, k))
    A2_tilde = np.dot(np.diag(s), v).reshape((sps, k, D2))
    psi.set_sites([i, i+1], [A1_tilde, A2_tilde])
    return trunc_error    
    
    
def _is_commuting(op1, op2):
    if op1.upper() in ['X'] and op2.upper() in ['Z', 'N']:
        return False
    return True

class LocalHamiltonian(object):
    """ A hamiltonian for a 1d system which is never fully stored in memory.
        Assumed to be a quasi-local; the local terms can be obtained in matrix representation on-demand."""
        
    def __init__(self, sps, static_list):
        """ L = chain length
            sps = states per site.
            static_list = a quspin-format static opstr list which fully specifies the hamiltonian
            """

        self.sps = sps
        self._static_list = static_list
        self._layers = []
        self._by_opstr = dict()
        for s in self._static_list:
            s[0] = s[0].upper()
            opstr, couplings = s[0], s[1]
            if opstr not in self._by_opstr.keys():
                self._by_opstr[opstr] = couplings
            else:
                self._by_opstr[opstr] += couplings
    
    def set_layer(self, opstrs):
        """ define a particular layer which is to be exponentiated all at once"""
        self._layers.append([o.upper() for o in opstrs])
            
    def get_all_terms(self, opstr_list):
        """ returns static list involving only the ops specified."""
        terms = []
        for o in opstr_list:
            terms.append( [o.upper(), self._by_opstr[o.upper()]] )
        return terms
    
class ExpPauliHamiltonian(LocalHamiltonian):
    """A trotterized exponential of hamiltonian in the pauli basis."""
    
    def __init__(self, static_list):
        LocalHamiltonian.__init__(self, 2, static_list)
    
    def _get_term_iterator(self, opstr, t):
        return ExpTermIterator( self.get_all_terms([opstr]), get_pauli_exp_gen(t))
    
    def _get_layer_iterator(self, layer, t):
        for opstr in layer:
            for (U,sites) in self._get_term_iterator(opstr, t):
                yield (U,sites)
    
    def layers(self, t):
        """Iterator over all gate layers."""
        for layer in self._layers:
            yield self._get_layer_iterator(layer, t)

    
    
        
def ExpTermIterator(static_list, exp_gen):
    """Generator which produces actual exponentials.
        exp_gen: a function which, given an opstr and a coupling strength, returns an exponential of the corresponding hamiltonian term.
        
        yields: tuples (U, sites) of unitaries and the corresponding sites they act on.
             """
    static_list = static_list.copy()
    np.random.shuffle(static_list)
    for opstr, couplings in static_list:
        for c in couplings:
            J, sites = c[0], c[1:]
            yield (exp_gen(opstr, J), sites)
            
            
def get_pauli_exp_gen(t):
    """ Returns a function which produces exponentials of Pauli matrices corresponding to a fixed Trotter time"""
    def pauli_exp_gen(opstr, J):
        """ opstr = quspin-style opstr, like 'Z' or 'nn'.
        J = a coupling strength
        
        Returns: exponential of the corresponding hamiltonian as np array
        """
        return get_pauli_exp(t*J, opstr)
    return pauli_exp_gen


    
class TrotterLayers(object):
    """ Stores trotter layers for a particular hamiltonian"""
    
    def __init__(self, expH):
        self.expH = expH
        self.evolve_time = None
        self.num_layers = None
        
    def set_evolve_time(self, T):
        self.evolve_time= T
    def set_num_layers(self, n):
        self.num_layers = n
    def _get_dt(self):
        return self.evolve_time / self.num_layers
    @property
    def layers(self):
        return self.expH.layers(self._get_dt())
    
    def apply(self, layer, psi):
        """Layer: a generator of local unitaries.
            psi: an MPS
            
            Applies each unitary in-place to psi.
            """
        for (U, sites) in layer:
            if len(sites)==1:
                
                _act_1qubit_gate(U, psi, sites[0])
            elif len(sites)==2 and sites[1]== (sites[0]+1):
                trunc_err=_act_2qubit_local_gate(U, psi, sites[0], Dmax=psi.Dmax)
                print("trunc error", trunc_err)
            else:
                raise NotImplementedError
        
    def run_evolution(self, psi, renormalize=False):
        """ Run trotter evolution on state psi (updates in-place).
            trot_layers: provides two (for now) noncommuting trotter layers.
            psi = MPS pure state.
            T = total evolution time.
            If renormalize=True: renormalize the state after layer application. """
            
        for _ in range(self.num_layers):
            for layer in self.layers:
                self.apply(layer, psi)
                if renormalize:
                    psi.normalize(np.random.randint(0, psi.L))
                    
                    
                    
        
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    