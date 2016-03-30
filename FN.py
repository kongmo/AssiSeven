# -*- coding: utf-8 -*-
"""
Created on Tue Mar 01 15:36:49 2016

@author: aa
"""
import numpy as np
def featureNormalize(X):
    mu=np.mean(X,axis=0)
    sigma=np.std(X,axis=0)
    X_norm=(X-mu)/sigma
    return (X_norm,mu,sigma)
    