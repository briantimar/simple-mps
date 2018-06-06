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


### 