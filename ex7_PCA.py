# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 10:54:46 2016

@author: aa
"""

import scipy.io as sio
import matplotlib.pyplot as plt
import FN
import PCA
import numpy as np
import DL
import ProjectD
import RecoverD
import DD
import cv2
import KMIC
import RKM
from mpl_toolkits.mplot3d import Axes3D
import PDP

data=sio.loadmat('ex7data1')
X=data['X']


#Part One: Load Example Dataset
print 'One: ======== Load Example Dataset1 ... '
plt.plot(X[:,0],X[:,1],'bo')
plt.axis(xmin=0.5,xmax=6.5,ymin=2,ymax=8)
plt.title('Example Dataset1')


#Part Two: Principal Component Analysis
print 'Two: ================ Running PCA on example dataset...'
result=FN.featureNormalize(X)

X_norm=result[0]
mu=result[1]
res=PCA.pca(X_norm)
U=res[0]
S=res[1]
S=np.eye(S.shape[0])*S


print 'Top eigenvector: '
print 'U[:,0] = %f %f ' % (U[0,0],U[1,0])
print '(You should expect to see -0.707107, -0.707107)'

tmp1=mu+1.5*np.dot(S[0,0],U[:,0].transpose())
tmp2=mu+1.5*np.dot(S[1,1],U[:,1].transpose())

DL.drawLine(mu,tmp1,color='k',linewidth=2)
DL.drawLine(mu,tmp2,color='b',linewidth=2)
plt.show()

#Part Three: Dimension Reduction
print 'Three: ==========Dimension Reduction on Example Dataset...'
plt.plot(X_norm[:,0],X_norm[:,1],'bo')
plt.axis(xmin=-4,xmax=3,ymin=-4,ymax=3)

K=1
Z=ProjectD.ProjectData(X_norm,U,K)
print 'Projection of the first example: ',Z[0]
print '(This value shoule be about 1.481274)'

X_rec=RecoverD.RecoverData(Z,U,K)
print 'Approximation of the first example: %f %f ' % (X_rec[0,0],X_rec[0,1])
print '(This value should be about -1.047319  -1.047419)'

plt.plot(X_rec[:,0],X_rec[:,0],'ro')
for i in range(X_norm.shape[0]):
    DL.drawLine(X_norm[i,:],X_rec[i,:],color='--k',linewidth=1)
plt.show()

#Part Four: Loading and Visualizing Face Data
print 'Four: ============== Loading face dataset...'
data=sio.loadmat('ex7faces')
X=data['X']
DD.displayData(X[0:100,:])
plt.show()


#Part Five: PCA on Face Data: Eignenfaces
print 'Five: ============ Running PCA on face datasets ...'
res=FN.featureNormalize(X)
X_norm=res[0]
resPCA=PCA.pca(X_norm)
U=resPCA[0]
S=resPCA[1]
DD.displayData(U[:,0:36].transpose())
plt.show()

# Part Six: Dimension Reduction for Faces
print 'Six: ============= Dimension Reduction for Face Datasets...'
K=100
Z=ProjectD.ProjectData(X_norm,U,K)
print 'The projected data  has a size of :',Z.shape

 #Part Seven: Visualizing of Faces after PCA Dimension Reduction
print 'Seven: ============ Visualizing the projected (reduced dimension) faces.'
X_rec=RecoverD.RecoverData(Z,U,K)
DD.displayData(X_norm[0:100,:])
plt.title('Original faces')
plt.show()
#
DD.displayData(X_rec[0:100,:])
plt.title('recovered faces')
plt.show()

#Part Eight: PCA for Visualization
print 'Eight: ========= PCA for Visualization...'
img=cv2.imread('bird_small.png')
img=np.float64(img)/255
X=img.reshape(img.shape[0]*img.shape[1],3)
K=16
max_iters=1
initial_centroids=KMIC.kMeansInitCentroids(X,K)
res=RKM.runkMeans(X,initial_centroids,max_iters)
centroids=res[0]
idx=res[1]

sel=np.int32(np.floor(np.random.random((1000,1))*X.shape[0])).flatten()

pal=np.random.random((K,3))
colors=pal[idx[sel]-1,:]
pos=np.where(colors > 0.5)
colors[pos]=1


fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(X[sel,0],X[sel,1],X[sel,2],marker='.',color=colors)
plt.show()

#Part Nine: PCA for Visualization
print 'Nine: ===== PCA for Visualization...'
res=FN.featureNormalize(X)
X_norm=res[0]
result=PCA.pca(X_norm)
U=result[0]
Z=ProjectD.ProjectData(X_norm,U,2)
PDP.plotDataPoints(Z[sel,:],idx[sel],K)
title='Pixel dataset plotted in 2D, using PCA for dimensionality reduction'
plt.title(title)
plt.show()

