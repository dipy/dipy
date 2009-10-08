''' Streamline object '''

import numpy as np
from numpy.linalg import norm

from .track_metrics as tm


class Volume(object):
    def __init__(self, func):
        self._func = func

    def at_points(self, points):
        return np.array([self._func(pt) for pt in points.T])


class PointVolume(Volume):
    def __init__(self, points, values, out_value=np.nan):
        self._points = points
        self._values = values
        
    def at_points(self, points):
        in_pts =  np.array(
            np.all(self._points == pt, axis=2) for pt in points.T)
        return in_pts

class Bundle(object):
    '''	Class representing a bundle of streamlines 
    '''
    def __init__(self):
        pass	


class StreamLine(object):
    ''' Class representing streamline
    '''
    def __init__(self, xyz, scalars=None, properties=None, copy=False):
        xyz = np.array(xyz, copy=copy)
        if xyz.ndim != 2:
            raise ValueError('Need 2 dimensional xyz array')
        if xyz.shape[1] != 3:
            raise ValueError('Need 2nd dimension to be length 3')
        n_pts = xyz.shape[0]
        if scalars is not None:
            if scalars.ndim > 2:
                raise ValueError('Scalars must have ndims <=2')
            if scalars.ndim == 1:
                scalars = scalars[None, :]
            if scalars.shape[0] != n_pts:
                raise ValueError('Scalars should have first dimension'
                                 'matching xyz')
        self.xyz = xyz
        self.x, self.y, self.z = xyz.T
        self.scalars = scalars
        self.properties = properties
        self.n_pts = n_pts

    def __len__(self):
        return self.n_pts

    def __iter__(self):
        
        if self.scalars is not None and self.properties is not None:
            for i in xrange(self.n_pts):
                    yield self.xyz[i], self.scalars[i], self.properties[i]
        elif self.scalars is not None and self.properties is None:
            for i in xrange(self.n_pts):
                    yield self.xyz[i], self.scalars[i], None
        elif self.scalars is None and self.properties is not None:
            for i in xrange(self.n_pts):
                    yield self.xyz[i], None, self.properties()
        else:
            for i in xrange(self.n_pts):
                    yield self.xyz[i], None, None
            
    
    def length(self,along=False):
        '''
        Returns streamline length in mm.
        '''
        return tm.length(self.xyz, along)

    def center(self):
        '''
        Returns streamline center of mass.
        '''
        return np.mean(self.xyz,axis=0)

    def midpoint(self):
        '''
        Returns middle point of streamline.
        '''
        if self.n_pts>=3:
            cumlen=self.length(along=True)        
            midlen=cumlen[-1]/2.0                
            ind=np.where(cumlen-midlen>0)[0][0]                
            orient=self.xyz[ind+1]-self.xyz[ind]
            len=cumlen[ind-1]        
            Ds=midlen-len
            
            return self.xyz[ind]+Ds*orient/norm(orient)
        else:
            return None
    
   
