""" Statistics on spheres
"""

import numpy as np
import dipy.core.geometry as geometry

def eigenstats(points, alpha=0.05):
    r'''Principal direction and confidence ellipse

    Implements equations in section 6.3.1(ii) of Fisher, Lewis and
    Embleton, supplemented by equations in section 3.2.5.

    Parameters
    ----------
    points : arraey_like (N,3)
        array of points on the sphere of radius 1 in $\mathbb{R}^3$
    alpha : real or None
        1 minus the coverage for the confidence ellipsoid, e.g. 0.05 for 95% coverage. 

    Returns
    -------
    centre : vector (3,)
        centre of ellipsoid
    b1 : vector (2,)
        lengths of semi-axes of ellipsoid
    '''
    n = points.shape[0]
    # the number of points

    rad2deg = 180/np.pi
    # scale angles from radians to degrees

    # there is a problem with averaging and axis data.  
    '''
    centroid = np.sum(points, axis=0)/n
    normed_centroid = geometry.normalized_vector(centroid)
    x,y,z = normed_centroid
    #coordinates of normed centroid
    polar_centroid = np.array(geometry.cart2sphere(x,y,z))*rad2deg
    '''
    
    cross = np.dot(points.T,points)/n
    # cross-covariance of points
    
    evals, evecs = np.linalg.eigh(cross)
    # eigen decomposition assuming that cross is symmetric
    
    order = np.argsort(evals)
    # eigenvalues don't necessarily come in an particular order?
    
    tau = evals[order]
    # the ordered eigenvalues
    
    h = evecs[:,order]
    # the eigenvectors in corresponding order

    h[:,2] = h[:,2]*np.sign(h[2,2])
    # map the first principal direction into upper hemisphere
    
    centre = np.array(geometry.cart2sphere(*h[:,2]))[1:]*rad2deg
    # the spherical coordinates of the first principal direction
        
    e = np.zeros((2,2))

    p0 = np.dot(points,h[:,0])
    p1 = np.dot(points,h[:,1])
    p2 = np.dot(points,h[:,2])
    # the principal coordinates of the points
    
    e[0,0] = np.sum((p0**2)*(p2**2))/(n*(tau[0]-tau[2])**2)
    e[1,1] = np.sum((p1**2)*(p2**2))/(n*(tau[1]-tau[2])**2)
    e[0,1] = np.sum((p0*p1*(p2**2))/(n*(tau[0]-tau[2])*(tau[1]-tau[2])))
    e[1,0] = e[0,1]
    # e is a 2x2 helper matrix
    
    b1 = np.array([np.NaN,np.NaN])

    d = -2*np.log(alpha)/n
    s,w = np.linalg.eig(e)
    g = np.sqrt(d*s)
    b1= np.arcsin(g)*rad2deg
    # b1 are the estimated 100*(1-alpha)% confidence ellipsoid semi-axes 
    # in degrees

    return centre, b1

    '''
    # b2 is equivalent to b1 above 

    # try to invert e and calculate vector b the standard errors of
    # centre - these are forced to a mixture of NaN and/or 0 in singular cases
    b2 = np.array([np.NaN,np.NaN])
    if np.abs(np.linalg.det(e)) < 10**-20:
        b2 = np.array([0,np.NaN])
    else:
        try:
            f = np.linalg.inv(e)
        except np.linalg.LigAlgError:
            b2 = np.array([np.NaN, np.NaN])
        else:
            t, y = np.linalg.eig(f)
            d = -2*np.log(alpha)/n
            g = np.sqrt(d/t)
            b2= np.arcsin(g)*rad2deg
    '''

