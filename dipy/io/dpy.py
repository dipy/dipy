''' A class for handling large tractography datasets.

    It is built using the pytables tools which in turn implement 
    key features of the HDF5 (hierachical data format) API [1]_.

    References
    ----------
    .. [1] http://www.hdfgroup.org/HDF5/doc/H5.intro.html
'''

import numpy as np

# Conditional import machinery for pytables
from ..utils.optpkg import optional_package

# Allow import, but disable doctests, if we don't have pytables
tables, have_tables, setup_module = optional_package('tables')

# Make sure not to carry across setup module from * import
__all__ = ['Dpy']


class Dpy(object):

    def __init__(self,fname,mode='r',compression=0):
        ''' Advanced storage system for tractography based on HDF5

        Parameters
        ------------
        fname : str, full filename
        mode : 'r' read
         'w' write
         'r+' read and write only if file already exists
         'a'  read and write even if file doesn't exist (not used yet)
        compression : 0 no compression to 9 maximum compression

        Examples
        ----------
        >>> import os
        >>> from tempfile import mkstemp #temp file
        >>> from dipy.io.dpy import Dpy
        >>> fd,fname = mkstemp()
        >>> fname = fname + '.dpy' #add correct extension
        >>> dpw = Dpy(fname,'w')
        >>> A=np.ones((5,3))
        >>> B=2*A.copy()
        >>> C=3*A.copy()
        >>> dpw.write_track(A)
        >>> dpw.write_track(B)
        >>> dpw.write_track(C)    
        >>> dpw.close()    
        >>> dpr = Dpy(fname,'r')    
        >>> A=dpr.read_track()
        >>> B=dpr.read_track()    
        >>> T=dpr.read_tracksi([0,1,2,0,0,2])
        >>> dpr.close()
        >>> os.remove(fname) #delete file from disk
        
        '''                
        
        self.mode=mode        
        self.f = tables.openFile(fname, mode = self.mode)
        self.N = 5*10**9
        self.compression = compression
        
        if self.mode=='w':
            self.streamlines=self.f.createGroup(self.f.root,'streamlines')
            #create a version number
            self.version=self.f.createArray(self.f.root,'version',['0.0.1'],'Dpy Version Number')
            
            self.tracks = self.f.createEArray(self.f.root.streamlines, 'tracks',tables.Float32Atom(), (0, 3),
                    "scalar Float32 earray", tables.Filters(self.compression),expectedrows=self.N)            
            self.offsets = self.f.createEArray(self.f.root.streamlines, 'offsets',tables.Int64Atom(), (0,),
                    "scalar Int64 earray", tables.Filters(self.compression), expectedrows=self.N+1)                                    
            self.curr_pos=0
            self.offsets.append(np.array([self.curr_pos]).astype(np.int64)) 
            
        if self.mode=='r':
            self.tracks=self.f.root.streamlines.tracks
            self.offsets=self.f.root.streamlines.offsets
            self.track_no=len(self.offsets)-1            
            self.offs_pos=0
            
    def version(self):
        ver=self.f.root.version[:]
        return ver[0]
                         
    def write_track(self,track):      
        ''' write on track each time
        '''  
        self.tracks.append(track.astype(np.float32))
        self.curr_pos+=track.shape[0]
        self.offsets.append(np.array([self.curr_pos]).astype(np.int64))
        
    def write_tracks(self,T):
        ''' write many tracks together
        '''
        for track in T:
            self.tracks.append(track.astype(np.float32))
            self.curr_pos+=track.shape[0]
            self.offsets.append(np.array([self.curr_pos]).astype(np.int64))
                    
    def read_track(self):    
        ''' read one track each time
        ''' 
        off0,off1=self.offsets[self.offs_pos:self.offs_pos+2]        
        self.offs_pos+=1       
        return self.tracks[off0:off1]
    
    def read_tracksi(self,indices):
        ''' read tracks with specific indices
        '''                       
        T=[]
        for i in indices:
            #print(self.offsets[i:i+2])
            off0,off1=self.offsets[i:i+2]            
            T.append(self.tracks[off0:off1])        
        return T
    
    def read_tracks(self):
        ''' read the entire tractography
        '''
        I=self.offsets[:]        
        TR=self.tracks[:]
        T=[]
        for i in range(len(I)-1):
            off0,off1=I[i:i+2]
            T.append(TR[off0:off1])    
        return T
    
    def close(self):                        
        self.f.close()



if __name__ == '__main__':
    pass
