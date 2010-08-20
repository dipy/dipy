import numpy as np
import dipy.core.geometry as geometry

def eigenstats(points, alpha=0.05):
    '''Principal direction and confidence ellipse

    Implements equations in section 6.3.1(ii) of Fisher, Lewis and
    Embleton, supplemented by equations in section 3.2.5.
    '''
    n = points.shape[0]

    rad2deg = 180/np.pi
    
    cross = np.dot(points.T,points)/n

    evals, evecs = np.linalg.eig(cross)

    order = np.argsort(evals)

    tau = evals[order]
    h = evecs[:,order]

    h[:,2] = h[:,2]*np.sign(h[2,2])

    centre = np.array(geometry.cart2sphere(*h[:,2]))[1:]*rad2deg

    e = np.zeros((2,2))

    p0 = np.dot(points,h[:,0])
    p1 = np.dot(points,h[:,1])
    p2 = np.dot(points,h[:,2])

    e[0,0] = np.sum((p0**2)*(p2**2))/(n*(tau[0]-tau[2])**2)
    e[1,1] = np.sum((p0**2)*(p2**2))/(n*(tau[1]-tau[2])**2)
    e[0,1] = np.sum((p0*p1*(p2**2))/(n*(tau[0]-tau[2])*(tau[1]-tau[2])))
    e[1,0] = e[0,1]

    try:
        f = np.linalg.inv(e)

    except np.linalg.LinAlgError, sing:
        return centre, np.array([0,0])
    
    b = np.array([0,0])
    t, y = np.linalg.eig(f)
    d = -2*np.log(alpha)/n
    g = np.sqrt(d/t)
    b = np.arcsin(g)*rad2deg

    return centre, b
