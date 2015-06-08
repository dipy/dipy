

# Compute the neighborhood. For now only 6. La

import numpy as np

def neighbor3D(input_volume,nhood):
    
    volume_shape = input_volume.shape[:3]
    Nhood = np.ones((volume_shape[0],volume_shape[1],volume_shape[2],nhood))
    
        
    
    a = np.arange(shape[0])
    b = np.arange(shape[1])
    c = np.arange(shape[2])

    A = a[1:-1]
    B = b[1:-1]
    C = c[1:-1]    
    
    Nhood1 = input_volume[]
    

