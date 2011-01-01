import numpy as np
from os.path import splitext

def read_bvec_file(filename, atol=.001):
    """
    Read gradient table information from a pair of files with extentions
    .bvec and .bval. The bval file should have one row of values
    representing the bvalues of each volume in the dwi data set. The bvec
    file should have three rows, where the rows are the x, y, and z
    components of the normalized gradient direction for each of the
    volumes.
    
    Parameters
    ------------
    filename : 
        The path to the either the bvec or bval file
    atol : float, optional 
        The tolorance used to check all the gradient directions are
        normalized. Defult is .001

    """

    base, ext = splitext(filename)
    if ext == '':
        bvec = base+'.bvec'
        bval = base+'.bval'
    elif ext == '.bvec':
        bvec = filename
        bval = base+'.bval'
    elif ext == '.bval':
        bvec = base+'.bvec'
        bval = filename
    else:
        raise ValueError('filename must have .bvec or .bval extension')
    
    b_values = np.loadtxt(bval)
    grad_table = np.loadtxt(bvec)
    if grad_table.shape[0] != 3:
        raise IOError('bvec file should have three rows')
    if b_values.ndim != 1:
        raise IOError('bval file should have one row')
    if b_values.shape[0] != grad_table.shape[1]:
        raise IOError('the gradient file and b value file should have the same number of columns')

    grad_norms = np.sqrt((grad_table**2).sum(0))
    if not np.allclose(grad_norms[b_values > 0], 1, atol=atol):
        raise IOError('the magnitudes of the gradient directions are not within '+str(atol)+' of 1') 
    grad_table[:,b_values > 0] = grad_table[:,b_values > 0]/grad_norms[b_values > 0]
    
    return (grad_table, b_values)

