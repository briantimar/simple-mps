#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 19:38:54 2018

@author: brian
"""

import numpy as np
from mps import MPS

L=50
sps = 2
D=5

psi = MPS(L, sps=sps)
psi.init_random(D)
print(psi.norm())
psi.left_normalize_full()
print(psi.norm())