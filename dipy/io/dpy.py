''' Some ideas for the dipy format
'''

import numpy as np
import tables


class Dpy_Old():

    def __init__(self,fname,mode):        
        self.fobj=open(fname,mode)
        self.mode=mode
        
        if self.mode=='w':
            self.pos=[0]
            
        if self.mode=='r':
            self.fobj.seek(-99,2)
            s=int(self.fobj.read())    
            self.fobj.seek(s)
            self.pos=np.load(self.fobj)[:-1]
            self.ipos=0
            self.lpos=len(self.pos)-1
    
    def write(self,arr):     
        np.save(self.fobj,arr)
        self.pos.append(self.fobj.tell())
    
    def read(self):
        if self.ipos<=self.lpos:
            self.fobj.seek(self.pos[self.ipos])
            self.ipos+=1                        
            return np.load(self.fobj)
        else:
            return None
    
    def read_indexed(self,indices):
        T=[]
        for i in indices:
            self.fobj.seek(self.pos[i])
            T.append(np.load(self.fobj))            
        return T
    
    def close(self):                
        if self.mode=='w':
            pos=np.array(self.pos)
            np.save(self.fobj,pos)    
            #reserve 100 bytes for position of pos in the file which 
            #holds where the numpy arrays are saved in the file
            self.fobj.write(str(pos[-1]).zfill(100))
                       
        self.fobj.close()
    


def test_on_the_fly():    

    a=2*np.ones((4,3)).astype(np.float32)
    b=3*np.ones((4,3)).astype(np.float32)
    c=4*np.ones((4,3)).astype(np.float32)
    d=5*np.ones((4,3)).astype(np.float32)         
    T=[a,b,c,d]    
    
    from time import time
    
    t1=time()
    for t in range(10**2):
        T.append(np.random.rand(100,3).astype(np.float32))    
    
    t2=time()
    print(t2-t1)
    
    print('>>> Writing file on the fly <<<')
    fname='test.dpy'
    dpw=Dpy(fname,'w')
    #for t in T:
    for t in range(10**2):
        t=np.random.rand(100,3).astype(np.float32)
        dpw.write(t)    
    dpw.close() 
    
    t3=time()
    print(t3-t2)
           
    print('>>> Reading file one array each time <<<')
    dpr=Dpy(fname,'r')        
    while True:         
        a=dpr.read()
        if a!=None: 
            #print(a)
            pass
        else:
            break
    dpr.close() 
    
    t4=time()
    print(t4-t3)
       
    print('>>> Reading directly with indexes from the disk <<<')
    dpi=Dpy(fname,'r')
    indices=[0,1,2,0,1,0]
    print dpi.read_indexed(indices)
    dpi.close()    
       
class Dpy():

    def __init__(self,fname,mode,compression=0):
        '''
        Parameters
        ----------
        fname: str, full filename
        mode: 'r' read or 'w' write
        compression: 0 no compression to 9 maximum compression
        
        Examples
        ---------
        
        >>> dpw = Dpy('test.dpy','w')
        >>> A=np.ones((5,3))
        >>> B=2*A.copy()
        >>> C=3*A.copy()    
        >>> dpw.write_track(A)
        >>> dpw.write_track(B)
        >>> dpw.write_track(C)    
        >>> dpw.close()    
        >>> dpr = Dpy('test.dpy','r')    
        >>> print(dpr.read_track())
        >>> print(dpr.read_track())    
        >>> T=dpr.read_indexed([0,1,2,0,0,2])    
        >>> print T    
        >>> dpr.close()
        
        '''                
        
        self.mode=mode        
        self.f = tables.openFile(fname, mode = self.mode)
        self.N = 5*10**8
        self.compression = compression
        
        if self.mode=='w':
            self.streamlines=self.f.createGroup(self.f.root,'streamlines')
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
     
    def write_track(self,track):        
        self.tracks.append(track.astype(np.float32))
        self.curr_pos+=track.shape[0]
        self.offsets.append(np.array([self.curr_pos]).astype(np.int64))                        
        
                    
    def read_track(self):     
        off0,off1=self.offsets[self.offs_pos:self.offs_pos+2]        
        self.offs_pos+=1       
        return self.tracks[off0:off1]
    
    def read_indexed(self,indices):                       
        T=[]
        for i in indices:
            #print(self.offsets[i:i+2])
            off0,off1=self.offsets[i:i+2]            
            T.append(self.tracks[off0:off1])        
        return T
    
    def close(self):                        
        self.f.close()



if __name__ == '__main__':
    

    '''
    complevel=1
    N=5*10**8
    
    f = tables.openFile("test.h5", mode = "w")
    
    e = f.createEArray(f.root, 'tracks',
                    tables.Float32Atom(), (0, 3),
                    "scalar Float32 carray", tables.Filters(complevel),
                    expectedrows=N)

    A=np.ones((10**6,3))
    B=2*A.copy()
    
    print A.shape, A.size
    print B.shape, A.size
    
    e.append(A)
    e.append(B)
    
    f.close()
    '''
    
    dpw = Dpy('test.dpy','w')
    
    A=np.ones((5,3))
    B=2*A.copy()
    C=3*A.copy()    
    dpw.write_track(A)
    dpw.write_track(B)
    dpw.write_track(C)    
    dpw.close()
    
    dpr = Dpy('test.dpy','r')    
    print(dpr.read_track())
    print(dpr.read_track())    
    T=dpr.read_indexed([0,1,2,0,0,2])    
    print T    
    dpr.close()
    
    from time import time
    
#    t1=time()
#    dpw2 = Dpy('testbig.dpy','w')
#    for i in xrange(10**6):
#        dpw2.write_track(np.random.rand(np.random.randint(50,100),3).astype(np.float32))        
#    dpw2.close()
#    t2=time()
#    print(t2-t1)
    
    t3=time()
    dpr2 = Dpy('testbig.dpy','r')
    indices=np.random.randint(10,900000,20000)
    T=dpr2.read_indexed(indices)
    t4=time()
    print(t4-t3)
    dpr2.close()
    
        
    
    
    
    



        

        
        
    
    
    
    
    
    
    
    
    
