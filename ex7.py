# -*- coding: utf-8 -*-

import scipy.io as sio
import numpy as np
import FCC
import CC
import RKM
import cv2
import KMIC
import matplotlib.pyplot as plt

#Part One: Find Closest Centroids
print 'One: =============Find Closest Centroids...'
data=sio.loadmat('ex7data2')
X=data['X']
K=3

initial_centroids=np.array([[3,3],[6,2],[8,5]])

idx=FCC.findClosestCentroids(X,initial_centroids)
print 'Closest Centroids for the first 3 examples: '
print idx[0:3]
print '(the closest centroids should be 1, 3, 2 respectively)'


#Part Two: Compute Means
print 'Two: =============Compute Means ...'
centroids=CC.computeCentroids(X,idx,K)
print 'Centroids computed after initial finding of closest centroids: '
print centroids
print '( the centroids should be [ [ 2.428301 3.157924 ]  [ 5.813503 2.633656 ]  [ 7.119387 3.616684 ]]'


#Part Three: K-Means Clustering
print 'Three: ================== K-Means Clustering ...'
max_iters=10
res=RKM.runkMeans(X,initial_centroids,max_iters,True)
plt.show()
print 'K-Means Done...'

#Part Four: K-Means Clustering on Pixels
print 'Running K-Means clustering on pixels from an image '
img=cv2.imread('bird_small.png')

img=np.float64(img)/255

X=img.reshape(img.shape[0]*img.shape[1],3)
K=16
max_iters=10

initial_centroids=KMIC.kMeansInitCentroids(X,K)
result=RKM.runkMeans(X,initial_centroids,max_iters)

#Part Five: Image Compression
print 'Applying K-Means to compress an image'
centroids=result[0]

idx=FCC.findClosestCentroids(X,centroids)
X_recovered=centroids[idx-1,:]
X_recovered=X_recovered.reshape(img.shape[0],img.shape[1],3)
cv2.namedWindow('compress')
cv2.imshow('compress',X_recovered)
#title='Compressed with %dth order.' % K
#cv2.waitKey(0)
#cv2.destroyAllWindows()

cv2.namedWindow('origin')
cv2.imshow('origin',img)
cv2.waitKey(0)
cv2.destroyAllWindows()













