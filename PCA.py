# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 11:02:53 2016

@author: aa
"""
from numpy import linalg as la
import numpy as np

def pca(X):
    m=X.shape[0]
    sigma=1.0/m*(np.dot(X.transpose(),X))
    result=la.svd(sigma)
    return result

