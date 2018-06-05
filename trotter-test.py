#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 19:38:54 2018

@author: brian
"""

import numpy as np
from mps import MPS
from mps import MPO_from_local_matrix

L=10
sps = 2
D=1

psi = MPS(L, sps=sps)
psi.init_random(D)
print(psi.norm())
psi.right_normalize_full()
print(psi.norm())
psi.gauge(4)


O = np.array([[1, 0], [0, -1]])
mpo = MPO_from_local_matrix(O, 4)
print(mpo._expt_value_local(psi))