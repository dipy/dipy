''' Metrics for tracks, where tracks are arrays of points '''

import math

import numpy as np


def length(xyz, along=False):
    ''' Euclidean length of track line

    Parameters
    ----------
    xyz : array-like shape (N,3)
       array representing x,y,z of N points in a track
    along : bool, optional
       If True, return array giving cumulative length along track,
       otherwise (default) return scalar giving total length.

    Returns
    -------
    L : scalar or array shape (N-1,)
       scalar in case of `along` == False, giving total length, array if
       `along` == True, giving cumulative lengths.

    Examples
    --------
    >>> xyz = np.array([[1,1,1],[2,3,4],[0,0,0]])
    >>> expected_lens = np.sqrt([1+2**2+3**2, 2**2+3**2+4**2])
    >>> length(xyz) == expected_lens.sum()
    True
    >>> len_along = length(xyz, along=True)
    >>> np.allclose(len_along, expected_lens.cumsum())
    True
    >>> length([])
    0
    >>> length([[1, 2, 3]])
    0
    >>> length([], along=True)
    array([0])
    '''
    xyz = np.asarray(xyz)
    if xyz.shape[0] < 2:
        if along:
            return np.array([0])
        return 0
    dists = np.sqrt((np.diff(xyz, axis=0)**2).sum(axis=1))
    if along:
        return np.cumsum(dists)
    return np.sum(dists)


def midpoint(xyz):
    ''' Midpoint of track line

    Parameters
    ----------
    xyz : array-like shape (N,3)
       array representing x,y,z of N points in a track

    Returns
    -------
    mp : array shape (3,)
       Middle point of line, such that, if L is the line length then
       `np` is the point such that the length xyz[0] to `mp` and from
       `mp` to xyz[-1] is L/2.  If the middle point is not a point in
       `xyz`, then we take the interpolation between the two nearest
       `xyz` points.  If `xyz` is empty, return a ValueError

    Examples
    --------
    >>> midpoint([])
    Traceback (most recent call last):
       ...
    ValueError: xyz array cannot be empty
    >>> midpoint([[1, 2, 3]])
    array([1, 2, 3])
    >>> xyz = np.array([[1,1,1],[2,3,4]])
    >>> midpoint(xyz)
    array([ 1.5,  2. ,  2.5])
    >>> xyz = np.array([[0,0,0],[1,1,1],[2,2,2]])
    >>> midpoint(xyz)
    array([ 1.,  1.,  1.])
    >>> xyz = np.array([[0,0,0],[1,0,0],[3,0,0]])
    >>> midpoint(xyz)
    array([ 1.5,  0. ,  0. ])
    >>> xyz = np.array([[0,9,7],[1,9,7],[3,9,7]])
    >>> midpoint(xyz)
    array([ 1.5,  9. ,  7. ])
    '''
    xyz = np.asarray(xyz)
    n_pts = xyz.shape[0]
    if n_pts == 0:
        raise ValueError('xyz array cannot be empty')
    if n_pts == 1:
        return xyz.copy().squeeze()
    cumlen = np.zeros(n_pts)
    cumlen[1:] = length(xyz, along=True)
    midlen=cumlen[-1]/2.0
    ind=np.where((cumlen-midlen)>0)[0][0]
    len0=cumlen[ind-1]        
    len1=cumlen[ind]
    Ds=midlen-len0
    Lambda = Ds/(len1-len0)
    return Lambda*xyz[ind]+(1-Lambda)*xyz[ind-1]


def center_of_mass(xyz):
    ''' Center of mass of streamline

    Parameters
    ----------
    xyz : array-like shape (N,3)
       array representing x,y,z of N points in a track

    Returns
    -------
    com : array shape (3,)
       center of mass of streamline

    Examples
    --------
    >>> center_of_mass([])
    Traceback (most recent call last):
       ...
    ValueError: xyz array cannot be empty
    >>> center_of_mass([[1,1,1]])
    array([ 1.,  1.,  1.])
    >>> xyz = np.array([[0,0,0],[1,1,1],[2,2,2]])
    >>> center_of_mass(xyz)
    array([ 1.,  1.,  1.])
    '''
    xyz = np.asarray(xyz)
    if xyz.size == 0:
        raise ValueError('xyz array cannot be empty')
    return np.mean(xyz,axis=0)

def magn(xyz):
    ''' magnitude of vector
        
    '''    
    mag=np.sum(xyz**2,axis=1)*0.5
    imag=np.where(mag==0)
    mag[imag]=2.0e-100 
    
    return mag

def frenet_serret(xyz):
    ''' Frenet-Serret Space Curve Invarients
 
    Calculates the 3 vector and 2 scaler invarients of a space curve defined
    by vectors x,y and z.  If z is omitted then the curve is only a 2D,
    but the equations are still valid.
    
    Similar to
    http://www.mathworks.com/matlabcentral/fileexchange/7256

    _    r'
    T = ----  (Tangent)
        |r'|
 
    _       T'
    N = ----  (Normal)
            |T'|
    _      _   _
    B = T x N (Binormal)

    k = |T'|  (Curvature)
 
    t = dot(-B',N) (Torsion)
    
    Parameters
    ----------
    xyz : array-like shape (N,3)
       array representing x,y,z of N points in a track
    
    Returns
    ---------
    T :
    N :
    B :
    k :
    t :

    '''
    
    dxyz=np.gradient(xyz)[0]    
    ddxyz=np.gradient(dxyz)[0]
        
    #Tangent
    T=np.divide(dxyz.T,magn(dxyz)).T
    
    #Derivative of Tangent
    dT=np.gradient(T)[0]
    
    #Normal
    N = np.divide(dT,magn(dT))
    
    #Binormal
    B = np.cross(T,N)
    
    #Curvature
    #k = magn(np.cross(dxyz,ddxyz))
    
    #Torsion
    t = np.dot(-B,N,2)
    

def frechet_distance(xyz1,xyz2):
    ''' Coming soon
    http://www.cim.mcgill.ca/~stephane/cs507/Project.html
    '''
    pass        
    
    