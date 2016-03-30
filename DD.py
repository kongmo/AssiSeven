import numpy as np
import matplotlib.pyplot as plt

def displayData(X,width=None):

    if not width:
        width=np.round(np.sqrt(X.shape[1]))
        
        
    #plt.colormaps('gray')

    m=X.shape[0]
    n=X.shape[1]
    height=n/width
    width=int(width)
    height=int(height)
    
    display_rows=np.floor(np.sqrt(m))
    display_cols=np.ceil(m/display_rows)
    
    pad=1
    
    display_array=-np.ones((pad+display_rows*(height+pad),pad+display_cols*(width+pad)))
    
    curr_ex=0
    for j in range(int(display_rows)):
        for i in range(int(display_cols)):
            if curr_ex > m:
                break
            max_val=max(abs(X[curr_ex,:]))
            
            tmp_row=np.add(pad+(j-1)*(height+pad),range(height))
            tmp_col=np.add(pad+(i-1)*(width+pad),range(width))
            tmp_val=X[curr_ex,:].reshape(height,width)
            display_array[np.ix_(tmp_row,tmp_col)]=tmp_val/max_val
            curr_ex=curr_ex+1
        if curr_ex > m:
            break
   
    plt.imshow(display_array.transpose(),cmap=plt.cm.gray)
    #plt.title('Example from the dataset')
    plt.axis('off')

