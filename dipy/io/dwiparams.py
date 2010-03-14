''' Process diffusion imaging parameters

* ``q`` is a vector in Q space
* ``b`` is a b value
* ``g`` is the unit vector along the direction of q (the gradient
  direction)

Thus:

   b = norm(q)

   g = q  / norm(q)

(``norm(q)`` is the Euclidean norm of ``q``)

The B matrix ``B`` is a symmetric positive semi-definite matrix.  If
``q_est`` is the closest q vector equivalent to the B matrix, then:

   B ~ (q_est . q_est.T) / norm(q_est)

'''
import numpy as np
import numpy.linalg as npl


def B2q(B, tol=None):
    ''' Estimate q vector from input B matrix `B`

    We assume the input `B` is symmetric positive definite.  If not,
    then you will get the result as for the lower triangular part of `B`.

    Because the solution is a square root, the sign of the returned
    vector is arbitrary.  We set the vector to have a positive x
    component by convention.

    Parameters
    ----------
    B : (3,3) array-like
       B matrix - symmetric. We do not check the symmetry.
    tol : None or float
       absolute tolerance below which to consider eigenvalues of the B
       matrix to be small enough not to worry about them being negative,
       in check for positive semi-definite-ness.  None (default) results
       in a fairly tight numerical threshold proportional the maximum
       eigenvalue

    Returns
    -------
    q : (3,) vector
       Estimated q vector from B matrix `B`
    '''
    B = np.asarray(B)
    B = nearest_positive_semi_definite(B)
    w, v = npl.eigh(B)
<<<<<<< HEAD:dipy/io/dwiparams.py
##    tol = np.abs(w.max() * np.finfo(w.dtype).eps)
##    non_trivial = np.abs(w) > tol
##    if np.any(w[non_trivial] < 0):
##        raise ValueError('B not positive semi-definite')
    inds = np.argsort(w)[::-1]
    max_ind = inds[0]
=======
    if tol is None:
        tol = np.abs(w.max() * np.finfo(w.dtype).eps)
    non_trivial = np.abs(w) > tol
    if np.any(w[non_trivial] < 0):
        raise ValueError('B not positive semi-definite')
    inds = np.argsort(w)
    max_ind = inds[-1]
>>>>>>> 15ea3d0e0eb3c115e69165161a69057773b6fb24:dipy/io/dwiparams.py
    vector = v[:,max_ind]
    # because the factor is a sqrt, the sign of the vector is arbitrary.
    # We arbitrarily set it to have a positive x value.
    if vector[0] < 0:
        vector *= -1
    return vector * w[max_ind]

def nearest_positive_semi_definite(B):
    '''
    Implements least squares positive semi-definite constrained tensor 
    estimation 
    
    Reference: Niethammer M, San Jose Estepar R, Bouix S, Shenton M, Westin CF. 
    On diffusion tensor estimation. Conf Proc IEEE Eng Med Biol Soc. 
    2006;1:2622-5. PubMed PMID: 17946125; PubMed Central PMCID: PMC2791793.
 
    Parameters
    ----------
    B : (3,3) array-like
       B matrix - symmetric. We do not check the symmetry.
       
    Returns
    -------
    npds : (3,3) array
       Estimated nearest positive semi-definite array to matrix `B`.
    '''

    vals,vecs = npl.eigh(B)
    inds = np.argsort(vals)[::-1]
    invs = np.argsort(inds)
    # indexes eigenvalues in descending order
    vals = vals[inds]
    vecs = vecs[inds,:][:,inds]
    cardneg = np.sum(vals < 0)
    lam1a=vals[0]
    lam2a=vals[1]
    lam3a=vals[2]
    lam1b=lam1a+0.25*lam3a
    lam2b=lam2a+0.25*lam3a
    if cardneg == 0:
        return B
    elif cardneg == 1:
        if lam1b >= 0 and lam2b >= 0:
            b111=lam1b
        elif lam2b < 0:
            b111=np.max([0,lam1a+(lam2a+lam3a)/3.])
        else:
            b111=0
        if lam1b >= 0 and lam2b >= 0:
            b221=lam2b
        elif lam1b < 0:
            b221=np.max([0,lam2a+(lam1a+lam3a)/3.])
        else:
            b221=0
        preb = np.dot(vecs,np.dot(np.diag((np.array([b111,b221,0]))),vecs.T))
    elif cardneg == 2:
        b112=np.max([0,lam1a+(lam2a+lam3a)/3.]) 
        preb = np.dot(vecs,np.dot(np.diag(np.array([b112,0,0])),vecs.T))
    elif cardneg == 3:
        return np.zeros((3,3))
    return preb[:,invs][invs,:]
