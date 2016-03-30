# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 12:05:00 2016

@author: aa
"""
import numpy as np

def RecoverData(Z,U,K):
    X_rec=np.dot(Z,U[:,0:K].transpose())
    return X_rec