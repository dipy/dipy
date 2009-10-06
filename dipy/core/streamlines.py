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

class Bundle(object):
    '''	Class representing bundles of streamlines 
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
        for i in self.n_pts:
                yield self.xyz[i], self.scalars[i], self.properties[i]
    
    def length(self,along=False):
        '''
        Returns streamline length in mm.
        '''
        XYZ=self.xyz[1:]
        XYZ=(XYZ-self.xyz[:-1])**2

        if along:
            return np.cumsum(np.sqrt(XYZ[:,0]+XYZ[:,1]+XYZ[:,2]))
        else:
            return np.sum(np.sqrt(XYZ[:,0]+XYZ[:,1]+XYZ[:,2]))

    def center(self):
        '''
        Returns streamline center of mass.
        '''
        return np.mean(self.xyz,axis=0)

    def midpoint(self):
        '''
        Returns middle point of streamline.
        '''
        cumlen=self.length(along=True)
        
        midlen=cumlen[-1]        
        ind=where(cumlen-midlen>=0)[0]
                
        orient=self.xyz[ind]-self.xyz[ind-1]
        
        len=cumlen[ind-1]
        Ds=midlen-len
        
        return self.xyz[ind-1]+Ds*orient
    
    


    
	
