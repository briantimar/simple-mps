#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 13:20:31 2018

@author: btimar

Simple matrix product states
"""

import numpy as np


def gen_basis_strings(sps, L):
    if L ==1:
        return np.arange(sps, dtype=int).reshape((1,sps))
    else:
        sub_basis_str = gen_basis_strings(sps, L-1)
        Nsub = sub_basis_str.shape[1]
        basis_str = np.empty((L, sps**L),dtype=int)
        for d in range(sps):
            darr = d * np.ones((1,Nsub))
            basis_str[:, d*Nsub:(d+1)*Nsub] = np.concatenate((sub_basis_str, darr),axis=0)
        return basis_str


class DynamicArray(object):
    """Stores a sequence of arrays where the individual arrays can have any bond dimensions, but neighboring arrays have to agree with each other.
       Physical index is the zeroth axis of a particular array (there are L arrays, one per lattice site). The next two are bond (matrix) indices. 
       If num_phys_indices=2 is specified, another physical index is added at the end."""
    
    def __init__(self, sps, L, num_phys_indices=1):
        """num_phys_indices -- how many physical indices are attached to a particular node."""
        self.num_phys_indices=num_phys_indices
        self.sps = sps
        self.L = L
        self._arrs = []
        self.ndim = 4 if (num_phys_indices == 2) else 3
        
    def get_site(self,i):
        return self._arrs[i]
    
    def get_bond_shape(self,i):
        A=self.get_site(i)
        return (A.shape[1], A.shape[2])
    
    def _check_dims(self, i):
        A = self.get_site(i)
        if len(A.shape) != self.ndim:
            raise ValueError("wrong number of axes")
        if (A.shape[0] != self.sps):
            raise ValueError("Local dimension does not match sps: {0}".format(A.shape))
        if self.num_phys_indices==2:
            if (A.shape[3] != self.sps):
                raise ValueError("Local dimension does not match sps: {0}".format(A.shape))
    
    def _check_local_dimensions(self):
        """ Assumes a list of 3-dimensional arrays. Enforces: constant first axis size, 
            and adjacent agreement for 2 and 3. """
        for jj in range(self.L):
            self._check_dims(jj)

        for ii in range(self.L-1):
            A, Anext = self.get_site(ii), self.get_site(ii+1)
            if A.shape[2] != Anext.shape[1]:
                raise ValueError("Bond dimensions at {0} do not agree".format(ii))

    def _set_site(self, i, A):
        self._arrs[i] = A.copy()
                
    def set_sites(self, indx_list, arrs):
        if len(indx_list) != len(arrs):
            raise ValueError("index and array list do not match")
        for i in range(len(indx_list)):
            self._set_site(indx_list[i], arrs[i])
        self._check_local_dimensions()
    
    def _erase_arrs(self):
        self._arrs = []
    
    def init_random(self, D):
        """initialize random values with bond dimension D"""
        self._erase_arrs()
        
        if self.num_phys_indices==2:
            self._arrs.append(np.random.rand(self.sps, 1,D, self.sps))
            for _ in range(self.L-2):
                self._arrs.append(np.random.rand(self.sps, D,D, self.sps) / (D * np.sqrt(self.sps)) )
            self._arrs.append(np.random.rand(self.sps, D,1, self.sps))
            
        else:
            self._arrs.append(np.random.rand(self.sps, 1,D))
            for _ in range(self.L-2):
                self._arrs.append(np.random.rand(self.sps, D,D) / (D * np.sqrt(self.sps)) )
            self._arrs.append(np.random.rand(self.sps, D,1))
            
    
    def __repr__(self):
        if len(self._arrs)!=self.L or self.L >10:
            return "DynamicArray L={0} sps = {1}".format(self.L, self.sps)
        s= "DynamicArray: " 
        for i in range(self.L-1):
            s += "{0}-".format((self.get_bond_shape(i)))
        s+= "{0}".format(self.get_bond_shape(self.L-1))
        s+= "\n sps = {0}".format(self.sps)
        return s
        
        
class MPS(object):
    
    _bc_types = ['open']
    
    def __init__(self, L,sps=2, bc='open'):
        self.sps=sps
        self.L=L
        if bc not in MPS._bc_types:
            raise ValueError("Invalid boundary condition")
        self.bc=bc
        
        self._dynamic_array = DynamicArray(self.sps, self.L)
            
    def init_random(self,D):
        """Randomized state with uniform bond dimension D"""
        self._dynamic_array.init_random(D)
        
    def get_site(self, i):
        """returns array associated with a particular site"""
        return self._dynamic_array.get_site(i)    
    
    def set_site(self, i, A):
        self._dynamic_array.set_sites([i], [A])
    
    def set_sites(self, indx_list, Alist):
        self._dynamic_array.set_sites(indx_list, Alist)            
            
    def norm(self):
        """ Returns <psi | psi> """
        edge_tensor = self.get_site(0)
        M = np.swapaxes( np.tensordot(edge_tensor, np.conj(edge_tensor), axes=([0], [0])), 1,2) # dimensions De, De, D, D
        for i in range(1, self.L):
            A = self.get_site(i)
            A_Adag = np.tensordot(A, np.conj(A), axes=([0], [0]))
            M= np.tensordot(M, A_Adag, axes=([2,3], [0,2]))
        return np.einsum('ijij', M)
    
    def get_left_weight_matrix(self, i):
        """ returns the matrix sum_sigma A^dag A, A being the site matrix. If the state is left-normalized, this should be the identity."""
        A = self.get_site(i)
        return np.tensordot(A, np.conj(A), axes=([0, 1], [0, 1]))
            
    def get_right_weight_matrix(self, i):
        B = self.get_site(i)
        return np.tensordot(B, np.conj(B), axes=([0, 2], [0,2]))
    
    
    def _get_coeff(self, basis_state):
        """ entries of basis state = 0, ..., sps-1"""
        if len(basis_state)!=self.L:
            raise ValueError
        A=None
        for ii in range(self.L):
            sigma = basis_state[ii]

            sitemat = self.get_site(ii)[sigma, :, :]
            if A is None:
                A = sitemat
            else:
                A = np.dot(A, sitemat )
        return np.trace(A)
                
    
    def _brute_force_norm(self):
        """ never use this"""
        nm=0
        bstrs = gen_basis_strings(self.sps, self.L)
        for ii in range(bstrs.shape[1]):
            sigma = bstrs[:, ii]
            nm += np.abs(self._get_coeff(sigma))**2
        return nm
            
       
    def svd_push_right(self, i):
        """Perform SVD on the site matrix i and push the resulting singular values onto the matrix to the right."""
        if i == self.L-1 :
            raise ValueError("Can't push right at right edge site")
            
        A= self.get_site(i)
        sps, d1, d2 = A.shape
        A = np.reshape(A, (sps * d1, d2))
        u,s,v=np.linalg.svd(A, full_matrices=False)
        k=s.shape[0]
        Anew = np.reshape(u, (sps,d1,k))
        A_right_new = np.tensordot(self.get_site(i+1), np.dot(np.diag(s), v), axes = ([1], [1]))
        A_right_new= np.swapaxes(A_right_new, 1, 2)    

        self.set_sites([i, i+1], [Anew, A_right_new]) 
        
        
    def svd_push_left(self, i):
        """ Perform SVD on the site matrix i, push the SV's to the left; site i will end up right-normalized."""
        if i==0:
            raise ValueError("Can't push to left at left edge site")
        B = self.get_site(i)
        sps, D1, D2 = B.shape
        B = np.reshape(np.swapaxes(B, 0,1), (D1, sps * D2))
        u,s,v = np.linalg.svd(B, full_matrices=False)
        k=s.shape[0]
        Anew = np.swapaxes(v.reshape( (k, sps, D2)), 0, 1)
        Aleft_new = np.tensordot(self.get_site(i-1), np.dot(u, np.diag(s)), axes=([2], [0]))
        
        self.set_sites([i-1, i], [Aleft_new, Anew])
        
        
    def roll_right(self):
        """ Roll all SV's into the last site"""
        for i in range(self.L-1):
            self.svd_push_right(i)
            
    def roll_left(self):
        """ roll all SV's into the first site"""
        for i in range(1,self.L):
            self.svd_push_left(self.L - i)
            
    def _normalize_site(self, i):
        """ rescales site matrix to have total trace 1 -- such that, if all other sites are appropriately left/right normalized, the state will have norm 1."""
        A = self.get_site(i)
        nm = np.tensordot(A, np.conj(A), axes = ([0, 1, 2], [0, 1, 2]))
        self.set_site(i,A / np.sqrt(nm))
        
    def _normalize_right_end(self):
        """Rescales the final (righmost) site matrices so that they satisfy a left-normalization condition"""
      
        self._normalize_site(self.L-1)
    
    def _normalize_left_end(self):
        self._normalize_site(0)        
        
    def left_normalize_full(self):
        """Rolls right (i.e. left-normalizes all but the last site matrix), then rescales that one so as to give the whole state norm 1."""
        self.roll_right()
        self._normalize_right_end()
    
    def right_normalize_full(self):
        self.roll_left()
        self._normalize_left_end()

    def gauge(self, i):
        """Left-normalize all matrices to the left of site i, and right-normalize all those to the right.
        This is a prerequisite for easy MPO evaluation on site i.
        """
        for il in range(i):
            self.svd_push_right(il)
        for ir in range(i+1, self.L):
            self.svd_push_left(ir)
    
    def get_bond_shape(self, i):
        return self._dynamic_array.get_bond_shape(i)
    
    def __repr__(self):
        return "MPS. Internal array: \n" + self._dynamic_array.__repr__()



class MPO(object):
    """Stores a matrix product operator in dynamic array. """
    
    



def _act_1qubit_gate(U, psi, i):
    """Apply a single-qubit gate to MPS psi at site i.
       U = (sps) x (sps) array. For now assumed to be unitary.
       psi = (pure ) MPS state.
       
       Updates the state in-place.
       """
       
    A = psi.get_site(i)
    psi.set_site(i, np.tensordot(U, A), axes=([1], [0]))
    
  
    
def _act_2qubit_local_gate(U, psi, i):
    """ Apply two-qubit gate to sites i, i+1 of MPS psi.
        U = 4-index gate, each axis of dimension sps.
        psi = MPS
        
        Updates the state in-place.
        Note that after applying the gate, left-normalization if any is generally not preserved."""
    
    A1, A2 = psi.get_site(i), psi.get_site(i+1)
    sps = A1.shape[0]
    D1, D2 = A1.shape[1], A2.shape[2]
    AA= np.tensordot(A1, A2, axes ([2], [1])) ##sps, D1, sps, D2
    blob = np.tensordot(U, AA, axes =([2, 3], [0,2]))  # sps, sps, D1,D2
    blob = np.swapaxes(blob, 1, 2).reshape((sps*D1, sps*D2)) 
    u,s,v = np.linalg.svd(blob, full_matrices=False)
    k = s.shape[0]
    A1_tilde = u.reshape((sps, D1, k))
    A2_tilde = np.dot(np.diag(s), v).reshape((sps, k, D2))
    psi.set_sites([i, i+1], [A1_tilde, A2_tilde])
        
    
    







         