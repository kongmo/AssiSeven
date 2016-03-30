# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 08:43:25 2016

@author: aa
"""
import PDP
import matplotlib.pyplot as plt
import DL

def plotProgresskMeans(X,centroids,previous_centroids,idx,K,i):
   PDP.plotDataPoints(X,idx,K) 
   plt.plot(centroids[:,0],centroids[:,1],'kx',markersize=5,markeredgewidth=2,linewidth=2)
   
   
   for j in range(centroids.shape[0]):
       DL.drawLine(previous_centroids[j,:],centroids[j,:])
   title='Iteration '+str(i)
   plt.title(title)
