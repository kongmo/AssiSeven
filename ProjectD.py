# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 12:01:34 2016

@author: aa
"""
import numpy as np
def ProjectData(X_norm,U,K):
    Z=np.dot(X_norm,U[:,0:K])
    return Z
    