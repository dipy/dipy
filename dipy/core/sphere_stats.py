import sys

import numpy as np
import dipy.core.geometry as geometry

def eigenstats(points, alpha=0.05):
    '''Principal direction and confidence ellipse

    Implements equations in section 6.3.1(ii) of Fisher, Lewis and
    Embleton, supplemented by equations in section 3.2.5.
    '''
    n = points.shape[0]
    # the number of points

    rad2deg = 180/np.pi
    
    cross = np.dot(points.T,points)/n
    # cross-covariance of points
    
    evals, evecs = np.linalg.eigh(cross)
    # eigen decomposition assumingt  cross is symmetric
    
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
    e[1,1] = np.sum((p0**2)*(p2**2))/(n*(tau[1]-tau[2])**2)
    e[0,1] = np.sum((p0*p1*(p2**2))/(n*(tau[0]-tau[2])*(tau[1]-tau[2])))
    e[1,0] = e[0,1]
    # e is a 2x2 helper matrix
    
    b = np.array([np.NaN,np.NaN])

    # try to invert e and calculate vector b the standard errors of
    # centre - these are forced to 0 in singular cases
    if np.abs(np.linalg.det(e)) < 10**-20:
        b = np.array([0,0])
    else:
        try:
            f = np.linalg.inv(e)
        except np.linalg.LigAlgError:
            b = np.array([0, 0])
        else:
            t, y = np.linalg.eig(f)
            d = -2*np.log(alpha)/n
            g = np.sqrt(d/t)
            b= np.arcsin(g)*rad2deg
    
    return centre, b
