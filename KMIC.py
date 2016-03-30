# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 09:51:24 2016

@author: aa
"""
import numpy as np

def kMeansInitCentroids(X,K):
    centroids=np.zeros((K,X.shape[1]))
    idx=range(X.shape[0])
    np.random.shuffle(idx)
    centroids=X[idx[0:K],:]
    return centroids