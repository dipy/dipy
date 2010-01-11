''' Metrics for tracks, where tracks are arrays of points '''

import math
import numpy as np
from scipy.interpolate import splprep, splev

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

def magn(xyz,n=1):
    ''' magnitude of vector
        
    '''    
    mag=np.sum(xyz**2,axis=1)**0.5
    imag=np.where(mag==0)
    mag[imag]=np.finfo(float).eps

    if n>1:
        return np.tile(mag,(n,1)).T
    return mag.reshape(len(mag),1)    
    

def frenet_serret(xyz):
    ''' Frenet-Serret Space Curve Invarients
 
    Calculates the 3 vector and 2 scaler invarients of a space curve defined
    by vectors x,y and z.  If z is omitted then the curve is only a 2D,
    but the equations are still valid.
    
    Similar to
    http://www.mathworks.com/matlabcentral/fileexchange/11169

    _    r'
    T = ----  (Tangent)
        |r'|
 
    _    T'
    N = ----  (Normal)
        |T'|
    _   _   _
    B = T x N (Binormal)

    k = |T'|  (Curvature)
 
    t = dot(-B',N) (Torsion)
    
    Parameters
    ----------
    xyz : array-like shape (N,3)
       array representing x,y,z of N points in a track
    
    
    Returns
    ---------
    T : array shape (N,3)
        array representing the tangent of the curve xyz
    N : array shape (N,3)
        array representing the normal of the curve xyz    
    B : array shape (N,3)
        array representing the binormal of the curve xyz
    k : array shape (N,1)
        array representing the curvature of the curve xyz
    t : array shape (N,1)
        array representing the torsion of the curve xyz

    Examples
    --------
    Create a helix and calculate its tangent,normal, binormal, curvature and torsion
    
    >>> from dipy.core import track_metrics as tm
    >>> import numpy as np
    >>> theta = 2*np.pi*np.linspace(0,2,100)
    >>> x=np.cos(theta)
    >>> y=np.sin(theta)
    >>> z=theta/(2*np.pi)
    >>> xyz=np.vstack((x,y,z)).T
    >>> T,N,B,k,t=tm.frenet_serret(xyz)
    
    '''
    xyz = np.asarray(xyz)
    n_pts = xyz.shape[0]
    if n_pts == 0:
        raise ValueError('xyz array cannot be empty')
    
    dxyz=np.gradient(xyz)[0]        
    
    ddxyz=np.gradient(dxyz)[0]
    
    #Tangent        
    T=np.divide(dxyz,magn(dxyz,3))
    
    #Derivative of Tangent
    dT=np.gradient(T)[0]
    
    #Normal
    N = np.divide(dT,magn(dT,3))
    
    #Binormal
    B = np.cross(T,N)
    
    #Curvature 
    k = magn(np.cross(dxyz,ddxyz),1)/(magn(dxyz,1)**3)    
    
    #Torsion 
    #(In matlab was t=dot(-B,N,2))
    t = np.sum(-B*N,axis=1)
    
    #return T,N,B,k,t,dxyz,ddxyz,dT   
    return T,N,B,k,t


    
def mean_curvature(xyz):    
    ''' Calculates the mean curvature of a curve
    
    Parameters
    ----------
    xyz : array-like shape (N,3)
       array representing x,y,z of N points in a curve
        
    Returns
    ---------
    m : float 
        float representing the mean curvature
    
    Examples
    --------
    Create a straight line and a semi-circle and print their mean curvatures
    
    >>> from dipy.core import track_metrics as tm
    >>> import numpy as np
    >>> x=np.linspace(0,1,100)
    >>> y=0*x
    >>> z=0*x
    >>> xyz=np.vstack((x,y,z)).T
    >>> m=tm.mean_curvature(xyz)
    >>> print('Mean curvature for straight line',m)
    >>> 
    >>> theta=np.pi*np.linspace(0,1,100)
    >>> x=np.cos(theta)
    >>> y=np.sin(theta)
    >>> z=0*x
    >>> xyz=np.vstack((x,y,z)).T
    >>> m=tm.mean_curvature(xyz)
    >>> print('Mean curvature for semi-circle',m)        
    '''
    xyz = np.asarray(xyz)
    n_pts = xyz.shape[0]
    if n_pts == 0:
        raise ValueError('xyz array cannot be empty')
    
    dxyz=np.gradient(xyz)[0]            
    ddxyz=np.gradient(dxyz)[0]
    
    #Curvature
    k = magn(np.cross(dxyz,ddxyz),1)/(magn(dxyz,1)**3)    
        
    return np.mean(k)

def smart_curvature(xyz):
    ''' xyz needs to be downsampled in equi-length segments
    
    '''
    
    pass

def mean_orientation(xyz):
    '''
    Calculates the mean curvature of a curve
    
    Parameters
    ----------
    xyz : array-like shape (N,3)
       array representing x,y,z of N points in a curve
        
    Returns
    ---------
    m : float 
        float representing the mean orientation
    '''
    xyz = np.asarray(xyz)
    n_pts = xyz.shape[0]
    if n_pts == 0:
        raise ValueError('xyz array cannot be empty')
    
    dxyz=np.gradient(xyz)[0]  
        
    return np.mean(dxyz,axis=0)
    
def lee_distances(start0, end0, start1, end1,w=[1.,1.,1.]):
    ''' Based on Lee , Han & Whang SIGMOD07.
        Calculates 3 etric for the distance between two line segments
        and returns a w-weighted combination
    
    Parameters:
    -----------
        start0: float array(3,)
        end0: float array(3,)
        start1: float array(3,)
        end1: float array(3,)
    
    Returns:
    --------
    weighted_distance: float
        w[0]*perpendicular_distance+w[1]*parallel_distance+w[2]*angle_distance
    
    Examples:
    --------
    >>> import dipy.core.track_metrics as tm 
    >>> tm.lee_distances([0,0,0],[1,0,0],[3,4,5],[5,4,3],[1,0,0])
    >>> 5.9380966767403436  
    >>> tm.lee_distances([0,0,0],[1,0,0],[3,4,5],[5,4,3],[0,1,0])
    >>> 3.0  
    >>> tm.lee_distances([0,0,0],[1,0,0],[3,4,5],[5,4,3],[0,0,1])
    >>> 2.0  
    '''
    start0=np.asarray(start0,dtype='float64')    
    end0=np.asarray(end0,dtype='float64')    
    start1=np.asarray(start1,dtype='float64')    
    end1=np.asarray(end1,dtype='float64')    
    
    l_0 = np.inner(end0-start0,end0-start0)
    l_1 = np.inner(end1-start1,end1-start1)

    if l_1 > l_0:
        s_tmp = start0
        e_tmp = end0
        start0 = start1
        end0 = end1
        start1 = s_tmp
        end1 = e_tmp
    
    u1 = np.inner(start1-start0,end0-start0)/np.inner(end0-start0,end0-start0)
    u2 = np.inner(end1-start0,end0-start0)/np.inner(end0-start0,end0-start0)
    ps = start0+u1*(end0-start0)
    pe = start0+u2*(end0-start0)
    lperp1 = np.sqrt(np.inner(ps-start1,ps-start1))
    lperp2 = np.sqrt(np.inner(ps-end1,ps-end1))
    
    perpendicular_distance = (lperp1**2+lperp2**2)/(lperp1+lperp2)
    ## do we need to do something about division by zero????

    lpar1=np.min(np.inner(start0-ps, start0-ps),np.inner(end0-ps, end0-ps))
    lpar2=np.min(np.inner(start0-pe, start0-pe),np.inner(end0-pe, end0-pe))

    parallel_distance=np.sqrt(np.min(lpar1, lpar2))

    cos_theta_squared = np.inner(end0-start0,end1-start1)**2/ \
        (np.inner(end0-start0,end0-start0)*np.inner(end1-start1,end1-start1))

    angle_distance = np.sqrt((1-cos_theta_squared)*np.inner(end1-start1, end1-start1))

    return w[0]*perpendicular_distance+w[1]*parallel_distance+w[2]*angle_distance


def lee_perpendicular_distance(start0, end0, start1, end1):
    ''' Based on Lee , Han & Whang SIGMOD07.
        Calculates perpendicular distance metric for the distance between two line segments
    
    Parameters:
    -----------
        start0: float array(3,)
        end0: float array(3,)
        start1: float array(3,)
        end1: float array(3,)
    
    Returns:
    --------
        perpendicular_distance: float

    Examples:
    --------
    >>> import dipy.core.track_metrics as tm 
    >>> tm.lee_perpendicular_distance([0,0,0],[1,0,0],[3,4,5],[5,4,3])
    >>> 6.658057955239661
    '''
    start0=np.asarray(start0,dtype='float64')    
    end0=np.asarray(end0,dtype='float64')    
    start1=np.asarray(start1,dtype='float64')    
    end1=np.asarray(end1,dtype='float64')    
    
    l_0 = np.inner(end0-start0,end0-start0)
    l_1 = np.inner(end1-start1,end1-start1)

    #''' !
    if l_1 > l_0:
        s_tmp = start0
        e_tmp = end0
        start0 = start1
        end0 = end1
        start1 = s_tmp
        end1 = e_tmp
    #'''
    u1 = np.inner(start1-start0,end0-start0)/np.inner(end0-start0,end0-start0)
    u2 = np.inner(end1-start0 ,end0-start0)/np.inner(end0-start0,end0-start0)

    ps = start0+u1*(end0-start0)
    pe = start0+u2*(end0-start0)

    lperp1 = np.sqrt(np.inner(ps-start1,ps-start1))
    
    lperp2 = np.sqrt(np.inner(pe-end1,pe-end1))

    if lperp1+lperp2 > 0.:
        return (lperp1**2+lperp2**2)/(lperp1+lperp2)
    else:
        return 0.


def lee_parallel_distance(start0, end0, start1, end1):
    ''' Based on Lee , Han & Whang SIGMOD07.
        Calculates parallel distance metric for the distance between two line segments
    
    Parameters:
    -----------
        start0: float array(3,)
        end0: float array(3,)
        start1: float array(3,)
        end1: float array(3,)
    
    Returns:
    --------
        parallel_distance: float

    Examples:
    --------
    >>> import dipy.core.track_metrics as tm 
    >>> tm.lee_parallel_distance([0,0,0],[1,0,0],[3,4,5],[5,4,3])
    >>> 3.0  
    '''
    start0=np.asarray(start0,dtype='float64')    
    end0=np.asarray(end0,dtype='float64')    
    start1=np.asarray(start1,dtype='float64')    
    end1=np.asarray(end1,dtype='float64')    
    
    u1 = np.inner(start1-start0,end0-start0)/np.inner(end0-start0,end0-start0)
    u2 = np.inner(end1-start0,end0-start0)/np.inner(end0-start0,end0-start0)
    ps = start0+u1*(end0-start0)
    pe = start0+u2*(end0-start0)
    lpar1=np.min(np.inner(start0-ps, start0-ps),np.inner(end0-ps, end0-ps))
    lpar2=np.min(np.inner(start0-pe, start0-pe),np.inner(end0-pe, end0-pe))

    return np.sqrt(np.min(lpar1, lpar2))


def lee_angle_distance(start0, end0, start1, end1):
    ''' Based on Lee , Han & Whang SIGMOD07.
        Calculates angle distance metric for the distance between two line segments
    
    Parameters:
    -----------
        start0: float array(3,)
        end0: float array(3,)
        start1: float array(3,)
        end1: float array(3,)
    
    Returns:
    --------
        angle_distance: float

    Examples:
    --------
    >>> import dipy.core.track_metrics as tm 
    >>> tm.lee_angle_distance([0,0,0],[1,0,0],[3,4,5],[5,4,3])
    >>> 2.0 
    '''
    start0=np.asarray(start0,dtype='float64')    
    end0=np.asarray(end0,dtype='float64')    
    start1=np.asarray(start1,dtype='float64')    
    end1=np.asarray(end1,dtype='float64')    
    
    l_0 = np.inner(end0-start0,end0-start0)
    l_1 = np.inner(end1-start1,end1-start1)
        
    #print l_0
    #print l_1

    #''' !!!
    if l_1 > l_0:
        s_tmp = start0
        e_tmp = end0
        start0 = start1
        end0 = end1
        start1 = s_tmp
        end1 = e_tmp
    #'''
    #print l_0
    #print l_1
    
    cos_theta_squared = np.inner(end0-start0,end1-start1)**2/ (l_0*l_1)
    
    #print cos_theta_squared

    return np.sqrt((1-cos_theta_squared)*l_1)


def approximate_trajectory_partitioning(xyz, alpha=1.):
    ''' Implementation of Lee et al Approximate Trajectory
        Partitioning Algorithm
    
    Parameters:
    ------------------
    xyz: array(N,3) 
        initial trajectory
    alpha: float
        smoothing parameter (>1 => smoother, <1 => rougher)
    
    Returns:
    ------------
    characteristic_points: list of M array(3,) points
        which can be turned into an array with np.asarray() 
    '''
    
    characteristic_points=[xyz[0]]
    start_index = 0
    length = 2
    while start_index+length < len(xyz):
        current_index = start_index+length
        cost_par = minimum_description_length_partitoned(xyz[start_index:current_index+1])
        #        print cost_par
        cost_nopar = minimum_description_length_unpartitoned(xyz[start_index:current_index+1])
        #        print cost_nopar
        #print cost_par, cost_nopar, start_index,length 
        if alpha*cost_par>cost_nopar:
        #            print "cost_par>cost_nopar"
            characteristic_points.append(xyz[current_index-1])
            start_index = current_index-1
            length = 2
        else:
        #            print "cost_par<=cost_nopar"
            length+=1
    #        raw_input()
    characteristic_points.append(xyz[-1])
    return np.array(characteristic_points)


def minimum_description_length_partitoned(xyz):   
    # L(H)
    val=np.log2(np.sqrt(np.inner(xyz[-1]-xyz[0],xyz[-1]-xyz[0])))
    
    # L(D|H) 
    val+=np.sum(np.log2([lee_perpendicular_distance(xyz[j],xyz[j+1],xyz[0],xyz[-1]) for j in range(1,len(xyz)-1)]))
    val+=np.sum(np.log2([lee_angle_distance(xyz[j],xyz[j+1],xyz[0],xyz[-1]) for j in range(1,len(xyz)-1)]))
    
    return val


def minimum_description_length_unpartitoned(xyz):
    '''
    Example:
    --------
    >>> xyz = np.array([[0,0,0],[2,2,0],[3,1,0],[4,2,0],[5,0,0]])
    >>> tm.minimum_description_length_unpartitoned(xyz) == np.sum(np.log2([8,2,2,5]))/2
    '''
    return np.sum(np.log2((np.diff(xyz, axis=0)**2).sum(axis=1)))/2
    
def zhang_distances(xyz1,xyz2,metric='all'):
    '''Distance between tracks xyz1 and xyz2 using metrics in Zhang 2008
    
    Based on the metrics in Zhang, Correia, Laidlaw 2008 
    http://ieeexplore.ieee.org/xpl/freeabs_all.jsp?arnumber=4479455
    which in turn are based on those of Corouge et al. 2004
 
    Parameters
    ----------
    xyz1 : array, shape (N1,3)
    xyz2 : array, shape (N2,3)
       arrays representing x,y,z of the N1 and N2 points  of two tracks
    
    Returns
    -------
    avg_mcd : float
       average_mean_closest_distance
    min_mcd : float
       minimum_mean_closest_distance
    max_mcd : float
       maximum_mean_closest_distance
    '''
    mcd12,mcd21 = mean_closest_distances(xyz1,xyz2)
    
    if metric=='all':
        return (mcd12+mcd21)/2.0, min(mcd12,mcd21), max(mcd12,mcd21)
    elif metric=='avg':
        return (mcd12+mcd21)/2.0
    elif metric=='min':            
        return min(mcd12,mcd21)
    elif metric =='max':
        return max(mcd12,mcd21)
    else :
        ValueError('Wrong argument for metric')


def mean_closest_distances(xyz1,xyz2):
    '''Average distances between tracks xyz1 and xyz2
    
    Based on the metrics in Zhang, Correia, Laidlaw 2008 
    http://ieeexplore.ieee.org/xpl/freeabs_all.jsp?arnumber=4479455
    which in turn are based on those of Corouge et al. 2004
 
    Parameters
    ----------
    xyz1 : array, shape (N1,3)
    xyz2 : array, shape (N2,3)
       arrays representing x,y,z of the N1 and N2 points  of two tracks
    
    Returns
    -------
    mcd12 : float
       Mean closest distance from `xyz1` to `xyz2`
    mcd12 : float
       Mean closest distance from `xyz2` to `xyz1`
    '''
    n1 = xyz1.shape[0]
    n2 = xyz2.shape[0]
    d = np.resize(np.tile(xyz1,(n2)),(n1,n2,3)) \
        - np.transpose(np.resize(np.tile(xyz2,(n1)),(n2,n1,3)),(1,0,2))
    dm = np.sqrt(np.sum(d**2,axis=2))
    return np.average(np.min(dm,axis=0)), np.average(np.min(dm,axis=1))
   
def max_end_distances(xyz1,xyz2):
    '''Maximum distance of ends of tracks xyz1 from xyz2
    
     
    Parameters
    ----------
    xyz1 : array, shape (N1,3)
    xyz2 : array, shape (N2,3)
       arrays representing x,y,z of the N1 and N2 points  of two tracks
    
    Returns
    -------
    maxend : float
       Max end distance from `xyz1` to `xyz2`
    
    '''
    maxend=0.0
    
    for end in [xyz1[0],xyz1[-1]]:
        maxend = max(maxend,min([np.inner(t-end,t-end) for t in xyz2]))
    return np.sqrt(maxend)


def generate_combinations(items, n):
    """ Combine sets of size n from items
    
    Parameters:
    ---------------
    items : sequence
    
    n : int
    
    Returns:
    ----------
        
    ic : iterator
    
    Examples:
    -------------
    >>> ic=generate_combinations(range(3),2)
    >>> for i in ic: print i
    [0, 1]
    [0, 2]
    [1, 2]
    """
    
    if n == 0:
        yield []    
    elif n == 2:
        #if n=2 non_recursive
        for i in xrange(len(items)-1):
            for j in xrange(i+1,len(items)):
                yield [i,j]
    else:
        #if n>2 uses recursion 
        for i in xrange(len(items)):
            for cc in generate_combinations(items[i+1:], n-1):
                yield [items[i]] + cc
            
def bundle_similarities_zhang(bundle,metric='avg'):
    ''' Calculate the average, min and max similarity matrices using Zhang 2008 distances
    
    Parameters:
    ---------------
    bundle: sequence 
            of tracks as arrays, shape (N1,3) .. (Nm,3)
    metric: string
            'avg', 'min', 'max'
            
    Returns:
    ----------
    Appropriate metric from this list:
    
    S_avg : array, shape (len(bundle),len(bundle))
                average similarity matrix    
    S_min : array, shape (len(bundle),len(bundle))
                minimum similarity matrix
    S_max : array, shape (len(bundle),len(bundle))  
                maximum similarity matrix
    
        
    '''
    
    track_pairs = generate_combinations(range(len(bundle)), 2)

    S_avg=np.zeros((len(bundle),len(bundle)))
    S_min=np.zeros((len(bundle),len(bundle)))
    S_max=np.zeros((len(bundle),len(bundle)))
    
    for p in track_pairs:
        
        S_avg[p[0],p[1]],S_min[p[0],p[1]],S_max[p[0],p[1]]=zhang_distances(bundle[p[0]],bundle[p[1]])        
        
    return S_avg,S_min,S_max 


def most_similar_track_zhang(bundle,metric='avg'):
    ''' Calculate the average, min and max similarity matrices using Zhang 2008 distances
    
    Parameters:
    ---------------
    bundle: sequence 
            of tracks as arrays, shape (N1,3) .. (Nm,3)
    
    metric : string
            'avg', 'min', 'max'
                    
    Returns:
    ----------
    si : int 
        index of track with least average metric
    s  : array, shape (len(bundle),)
        similarities of si with all the other tracks
    '''
    track_pairs = generate_combinations(range(len(bundle)), 2)
    s = np.zeros((len(bundle)))    
    '''
    for p in track_pairs:
        delta = zhang_distances(bundle[p[0]],bundle[p[1]],metric)
        s[p[0]] += delta
        s[p[1]] += delta
        
    '''
    deltas=[(zhang_distances(bundle[p[0]],bundle[p[1]],metric),p[0],p[1])
            for p in track_pairs]
    for d in deltas:
        s[d[1]]+=d[0]
        s[d[2]]+=d[0]
    si=np.argmin(s)
    smin=[zhang_distances(t,bundle[si],metric) for t in bundle]
    return si, np.array(smin)


def track_bundle_similarity(t,S): 
    ''' Calculate between track `t` and rest of the tracks in the bundle 
    using an upper diagonal similarity matrix S.
    '''    
    return np.hstack((S[:t,t],np.array([0]),S[t,t+1:]))


def longest_track_bundle(bundle,sort=False):
    ''' Return longest track or length sorted track indices in `bundle`

    If `sort` == True, return the indices of the sorted tracks in the
    bundle, otherwise return the longest track. 
    
    Parameters
    ----------
    bundle : sequence 
       of tracks as arrays, shape (N1,3) ... (Nm,3)
    sort : bool, optional
       If False (default) return longest track.  If True, return length
       sorted indices for tracks in bundle
    Returns
    -------
    longest_or_indices : array
       longest track - shape (N,3) -  (if `sort` is False), or indices
       of length sorted tracks (if `sort` is True)
    '''
    alllengths=[tm.length(t) for t in bundle]
    alllengths=np.array(alllengths)        
    if sort:
        ilongest=alllengths.argsort()
        return ilongest
    else:
        ilongest=alllengths.argmax()
        return bundle[ilongest]
 

def most_similar_track(S,sort=False):
    ''' Return the index of the most similar track given a diagonal
    similarity matrix as returned from function
    bundle_similarities_zhang(bundle)
    '''
    if sort:
        return (S+S.T).sum(axis=0).argsort()    
    else:        
        return   (S+S.T).sum(axis=0).argmin()


def any_segment_intersect_sphere(xyz,center,radius):
    ''' If any segment of the track is intersecting with a sphere of
    specific center and radius return True otherwise False
    
    Parameters
    ----------
    xyz : array, shape (N,3)
       representing x,y,z of the N points of the track
    center : array, shape (3,)
       center of the sphere
    radius : float
       radius of the sphere
    
    Returns
    -------
    tf : {True,False}           
       True if track `xyz` intersects sphere
       
    Notes
    -----
    The ray to sphere intersection method used here is similar with 
    http://local.wasp.uwa.edu.au/~pbourke/geometry/sphereline/
    http://local.wasp.uwa.edu.au/~pbourke/geometry/sphereline/source.cpp
    we just applied it for every segment neglecting the intersections where
    the intersecting points are not inside the segment      
    '''
    center=np.array(center)
    #print center
    
    lt=xyz.shape[0]
    
    for i in xrange(lt-1):

        #first point
        x1=xyz[i]
        #second point
        x2=xyz[i+1]
        
        #do the calculations as given in the Notes        
        x=x2-x1
        a=np.inner(x,x)
        x1c=x1-center
        b=2*np.inner(x,x1c)
        c=np.inner(center,center)+np.inner(x1,x1)-2*np.inner(center,x1) - radius**2
        bb4ac =b*b-4*a*c
        #print 'bb4ac',bb4ac
        if abs(a)<np.finfo(float).eps or bb4ac < 0 :#too small segment or no intersection           
            continue
        if bb4ac ==0: #one intersection point p
            mu=-b/2*a
            p=x1+mu*x                        
            #check if point is inside the segment 
            #print 'p',p
            if np.inner(p-x1,p-x1) <= a:
                return True
           
        if bb4ac > 0: #two intersection points p1 and p2
            mu=(-b+np.sqrt(bb4ac))/(2*a)
            p1=x1+mu*x            
            mu=(-b-np.sqrt(bb4ac))/(2*a)
            p2=x1+mu*x       
            #check if points are inside the line segment
            #print 'p1,p2',p1,p2
            if np.inner(p1-x1,p1-x1) <= a or np.inner(p2-x1,p2-x1) <= a:
                return True
    return False
    

def intersect_sphere(xyz,center,radius):
    ''' If any point of the track is inside a sphere of a specified
    center and radius return True otherwise False.  Mathematicaly this
    can be simply described by ||x-c||<=r where ``x`` a point ``c`` the
    center of the sphere and ``r`` the radius of the sphere.
            
    Parameters
    ----------
    xyz : array, shape (N,3)
       representing x,y,z of the N points of the track
    center : array, shape (3,)
       center of the sphere
    radius : float
       radius of the sphere
    
    Returns
    -------
    tf : {True,False}    
    
    Examples
    --------
    >>> line=np.array(([0,0,0],[1,1,1],[2,2,2]))
    >>> sph_cent=np.array([1,1,1])
    >>> sph_radius = 1
    >>> intersect_sphere(line,sph_cent,sph_radius)
    '''
    return (np.sqrt(np.sum((xyz-center)**2,axis=1))<=radius).any()==True


def intersect_sphere_points(xyz,center,radius):
    ''' If a track intersects with a sphere of a specified center and
    radius return the points that are inside the sphere otherwise False.
    Mathematicaly this can be simply described by ||x-c||<=r where ``x``
    a point ``c`` the center of the sphere and ``r`` the radius of the
    sphere.
            
    Parameters
    ----------
    xyz : array, shape (N,3)
       representing x,y,z of the N points of the track
    center : array, shape (3,)
       center of the sphere
    radius : float
       radius of the sphere
    
    Returns
    -------
    xyzn : array, shape(M,3)
       array representing x,y,z of the M points inside the sphere
    
    Examples
    --------
    >>> line=np.array(([0,0,0],[1,1,1],[2,2,2]))
    >>> sph_cent=np.array([1,1,1])
    >>> sph_radius = 1
    >>> intersect_sphere_points(line,sph_cent,sph_radius)
    '''
    return xyz[(np.sqrt(np.sum((xyz-center)**2,axis=1))<=radius)]


def orientation_in_sphere(xyz,center,radius):
    '''Calculate average orientation of a track segment inside a sphere
    
    Parameters
    --------------
    xyz : array, shape (N,3)
       representing x,y,z of the N points of the track
    center : array, shape (3,)
       center of the sphere
    radius : float
       radius of the sphere    
    
    Returns
    -------
    orientation : array, shape (3,)
       vector representing the average orientation of the track inside
       sphere
        
    Examples
    --------
    >>> track=np.array([[1,1,1],[2,2,2],[3,3,3]])    
    >>> center=(2,2,2)
    >>> radius=5
    >>> orientation_in_sphere(track)
    array([1.,1.,1.])
    '''
    xyzn=intersect_sphere_points(xyz,center,radius)   
    if xyzn.shape[0] >1:
        #calculate gradient
        dxyz=np.gradient(xyzn)[0]
        #average orientation
        return np.mean(dxyz,axis=0)
    else:
        return None


def spline(xyz,s=3,k=2,nest=-1):
    ''' Generate B-splines as documented in 
    http://www.scipy.org/Cookbook/Interpolation
    
    The scipy.interpolate packages wraps the netlib FITPACK routines
    (Dierckx) for calculating smoothing splines for various kinds of
    data and geometries. Although the data is evenly spaced in this
    example, it need not be so to use this routine.
    
    Parameters:
    ---------------
    xyz : array, shape (N,3)
       array representing x,y,z of N points in 3d space
    s : float, optional
       A smoothing condition.  The amount of smoothness is determined by
       satisfying the conditions: sum((w * (y - g))**2,axis=0) <= s
       where g(x) is the smoothed interpolation of (x,y).  The user can
       use s to control the tradeoff between closeness and smoothness of
       fit.  Larger satisfying the conditions: sum((w * (y -
       g))**2,axis=0) <= s where g(x) is the smoothed interpolation of
       (x,y).  The user can use s to control the tradeoff between
       closeness and smoothness of fit.  Larger s means more smoothing
       while smaller values of s indicate less smoothing. Recommended
       values of s depend on the weights, w.  If the weights represent
       the inverse of the standard-deviation of y, then a: good s value
       should be found in the range (m-sqrt(2*m),m+sqrt(2*m)) where m is
       the number of datapoints in x, y, and w.
    k : int, optional
       Degree of the spline.  Cubic splines are recommended.  Even
       values of k should be avoided especially with a small s-value.
       for the same set of data.  If task=-1 find the weighted least
       square spline for a given set of knots, t.
    nest : None or int, optional
       An over-estimate of the total number of knots of the spline to
       help in determining the storage space.  None results in value
       m+2*k. -1 results in m+k+1. Always large enough is nest=m+k+1.
       Default is -1.  
    
    
    Returns
    -------
    xyzn : array, shape (M,3)
    
    Examples
    --------
    >>> import numpy as np
    >>> # make ascending spiral in 3-space
    >>> t=np.linspace(0,1.75*2*np.pi,100)

    >>> x = np.sin(t)
    >>> y = np.cos(t)
    >>> z = t

    >>> # add noise
    >>> x+= np.random.normal(scale=0.1, size=x.shape)
    >>> y+= np.random.normal(scale=0.1, size=y.shape)
    >>> z+= np.random.normal(scale=0.1, size=z.shape)
    
    >>> xyz=np.vstack((x,y,z)).T    
    >>> xyzn=spline(xyz,3,2,-1)
    
    See also
    ----------
    From scipy documentation scipy.interpolate.splprep and
    scipy.interpolate.splev
    '''
    # find the knot points
    tckp,u = splprep([xyz[:,0],xyz[:,1],xyz[:,2]],s=s,k=k,nest=nest)
    # evaluate spline, including interpolated points
    xnew,ynew,znew = splev(np.linspace(0,1,400),tckp)
    return np.vstack((xnew,ynew,znew)).T    


def startpoint(xyz):
    return xyz[0]


def endpoint(xyz):
    return xyz[-1]


def arbitrarypoint(xyz,distance):
    ''' Select an arbitrary point along distance on the track (curve)

    Parameters
    ----------
    xyz : array-like shape (N,3)
       array representing x,y,z of N points in a track
    distance : float
        float representing distance travelled from the xyz[0] point of
        the curve along the curve.

    Returns
    -------
    ap : array shape (3,)
       arbitrary point of line, such that, if the arbitrary point is not
       a point in `xyz`, then we take the interpolation between the two
       nearest `xyz` points.  If `xyz` is empty, return a ValueError
    
    Examples
    -----------
    >>> import numpy as np
    >>> theta=np.pi*np.linspace(0,1,100)
    >>> x=np.cos(theta)
    >>> y=np.sin(theta)
    >>> z=0*x
    >>> xyz=np.vstack((x,y,z)).T
    >>> ap=arbitrarypoint(xyz,tm.length(xyz)/3)
    >>> print('The point along the curve that traveled the given distance is ',ap)        
    '''
    xyz = np.asarray(xyz)
    n_pts = xyz.shape[0]
    if n_pts == 0:
        raise ValueError('xyz array cannot be empty')
    if n_pts == 1:
        return xyz.copy().squeeze()
    cumlen = np.zeros(n_pts)
    cumlen[1:] = length(xyz, along=True)    
    if cumlen[-1]<distance:
        raise ValueError('Given distance is bigger than '
                         'the length of the curve')
    ind=np.where((cumlen-distance)>0)[0][0]
    len0=cumlen[ind-1]        
    len1=cumlen[ind]
    Ds=distance-len0
    Lambda = Ds/(len1-len0)
    return Lambda*xyz[ind]+(1-Lambda)*xyz[ind-1]


def _extrap(xyz,cumlen,distance):
    ''' Helper function for extrapolate    
    '''    
    ind=np.where((cumlen-distance)>0)[0][0]
    len0=cumlen[ind-1]        
    len1=cumlen[ind]
    Ds=distance-len0
    Lambda = Ds/(len1-len0)
    return Lambda*xyz[ind]+(1-Lambda)*xyz[ind-1]


def downsample(xyz,n_pols=3):
    ''' downsample for a specific number of points along the curve

    Uses the length of the curve. It works in as similar fashion to
    midpoint and arbitrarypoint.
    
    Parameters
    ----------
    xyz : array-like shape (N,3)
       array representing x,y,z of N points in a track
    n_pol : int
       integer representing number of points (poles) we need along the curve.

    Returns
    -------
    xyz2 : array shape (M,3)
       array representing x,z,z of M points that where extrapolated. M
       should be equal to n_pols
    
    Examples
    --------
    >>> import numpy as np
    >>> # a semi-circle
    >>> theta=np.pi*np.linspace(0,1,100)
    >>> x=np.cos(theta)
    >>> y=np.sin(theta)
    >>> z=0*x
    >>> xyz=np.vstack((x,y,z)).T
    >>> xyz2=downsample(xyz,3)    
    >>> # a cosine
    >>> x=np.pi*np.linspace(0,1,100)
    >>> y=np.cos(theta)
    >>> z=0*y
    >>> xyz=np.vstack((x,y,z)).T
    >>> xyz2=downsample(xyz,3)
    >>> xyz3=downsample(xyz,10)
    '''
    xyz = np.asarray(xyz)
    n_pts = xyz.shape[0]
    if n_pts == 0:
        raise ValueError('xyz array cannot be empty')
    if n_pts == 1:
        return xyz.copy().squeeze()
    cumlen = np.zeros(n_pts)
    cumlen[1:] = length(xyz, along=True)    
    step=cumlen[-1]/(n_pols-1)
    if cumlen[-1]<step:
        raise ValueError('Given numper of points n_pols is incorrect. ')
    if n_pols<=2:
        raise ValueError('Given numper of points n_pols needs to be'
                         ' higher than 2. ')
    xyz2=[_extrap(xyz,cumlen,distance)
          for distance in np.arange(0,cumlen[-1],step)]
    return np.vstack((np.array(xyz2),xyz[-1]))


def principal_components(xyz):
    ''' We use PCA to calculate the 3 principal directions for dataset xyz
    '''
    C=np.cov(xyz.T)    
    va,ve=np.linalg.eig(C)
    return va,ve


def midpoint2point(xyz,p):
    ''' Calculate distance from midpoint of a curve to arbitrary point p
    
    Parameters
    ----------
    xyz : array-like shape (N,3)
       array representing x,y,z of N points in a track
    p : array shape (3,)
       array representing an arbitrary point with x,y,z coordinates in
       space.

    Returns
    -------
    d : float
       a float number representing Euclidean distance         
    '''
    mid=midpoint(xyz)     
    return np.sqrt(np.sum((xyz-mid)**2))


def test_approximate_trajectory_partitioning():
    
    t=np.linspace(0,1.75*2*np.pi,1000)

    x = np.sin(t)
    y = np.cos(t)
    z = t
    
    xyz=np.vstack((x,y,z)).T 
    
    xyza1 = approximate_trajectory_partitioning(xyz,alpha=1.)
    xyza2 = approximate_trajectory_partitioning(xyz,alpha=2.) 
    
    

if __name__ == "__main__":
#    pass

    import cProfile
    cProfile.run('test_approximate_trajectory_partitioning()', 'fooprof')

    

