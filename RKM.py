# -*- coding: utf-8 -*-


import numpy as np
import FCC
import PPKM
import CC
import matplotlib.pyplot as plt

def runkMeans(X,initial_centroids,max_iters,plot_progress=None):
    if plot_progress == None :
        plot_progress=False
    
    #if plot_progress:
    shape=X.shape
    m=shape[0]
    K=initial_centroids.shape[0]
    centroids=initial_centroids
    previous_centroids=centroids
    idx=np.zeros((m,1))
    
    
    for i in range(max_iters):
        print 'K-Means iteration： %d %d' % (i,max_iters)
        idx=FCC.findClosestCentroids(X,centroids)

        if plot_progress:
            PPKM.plotProgresskMeans(X,centroids,previous_centroids,idx,K,i)
            previous_centroids=centroids
        centroids= CC.computeCentroids(X,idx,K)
    return (centroids,idx)

    
        
