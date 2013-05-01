import numpy as np
def cshift3D(x, m, d):
    s=x.shape
    idx=(np.array(range(s[d]))+(s[d]-m%s[d]))%s[d]
    if d==0:
        return x[idx,:,:]
    elif d==1:
        return x[:,idx,:]
    else:
        return x[:,:,idx];
