''' Process diffusion imaging parameters


'''
import numpy as np
import numpy.linalg as npl


def B2q(B):
    ''' Estimate q vector from input B matrix `B`

    Parameters
    ----------
    B : (3,3) array-like
       B matrix - symmetric
       
    Returns
    -------
    q : (3,) vector
       Estimated q vector from B matrix `B`
    '''
    B = np.asarray(B)
    w, v = npl.eigh(B)
    inds = np.argsort(w)
    max_ind = inds[-1]
    return v[:,max_ind] * np.sqrt(w[max_ind])
