''' Streamline object '''

import numpy as np


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

    def __len__(self):
        return self.xyz.shape[0]
    
    
