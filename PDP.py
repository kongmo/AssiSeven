# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 08:47:52 2016

@author: aa
"""
import matplotlib.pyplot as plt
import numpy as np

def plotDataPoints(X,idx,K):
    palette=np.random.random((K,3))
    pos=np.int32(idx-1)
    colors=palette[pos,:]
    
    plt.scatter(X[:,0],X[:,1],marker='.',color=colors)
