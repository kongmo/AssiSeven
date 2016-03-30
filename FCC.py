import numpy as np

def findClosestCentroids(X,initial_centroids):
    K=initial_centroids.shape[0]
    m=X.shape[0]
    idx=np.zeros(m,dtype=int)

    tmpDis=np.zeros(K)
    for i in range(m):
        for j in range(K):
            #tmp=(X[i,0]-initial_centroids[j,0])**2+(X[i,1]-initial_centroids[j,1])**2
            tmp=((X[i,:]-initial_centroids[j,:])**2).sum()
            tmpDis[j]=np.sqrt(tmp)
        pos = tmpDis.argmin()

        idx[i]=pos+1
    return idx

    
