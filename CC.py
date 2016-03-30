import numpy as np

def computeCentroids(X, idx,K):
    res=np.zeros((K,X.shape[1]))
    for i in range(K):
        pos=np.where(idx == i+1)
        res[i,:]=sum(X[pos[0],:])/len(pos[0])
    return res

        
    
