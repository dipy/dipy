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

    def __init__(self,fname,mode):        
        
        self.mode=mode        
        self.f = tables.openFile(fname, mode = self.mode)
        self.N = 5*10**8
        self.compression = 1
        
        if self.mode=='w':
            self.tracks = self.f.createEArray(self.f.root, 'tracks',tables.Float32Atom(), (0, 3),
                    "scalar Float32 earray", tables.Filters(self.compression),expectedrows=self.N)            
            self.offsets = self.f.createEArray(self.f.root, 'offsets',tables.Int64Atom(), (0, 2),
                    "scalar Int64 earray", tables.Filters(self.compression), expectedrows=self.N)            
            self.curr_pos=0
            
        if self.mode=='r':
            self.tracks=self.f.root.tracks
            self.offsets=self.f.root.offsets
            self.offs_pos=0
     
    def write_track(self,track):        
        self.tracks.append(track.astype(np.float32))
        self.offsets.append(np.array([[self.curr_pos,track.shape[0] ]]).astype(np.int64))                        
        self.curr_pos+=track.shape[0]
                    
    def read_track(self):        
        print 'op',self.offs_pos
        print self.offsets[:]
        off=self.offsets[self.offs_pos] 
        print off
        self.offs_pos+=1       
        return self.tracks[off[0]:off[0]+off[1]]
    
    def read_indexed(self,indices):                       
        T=[]
        for i in indices:
            off=self.offsets[i]
            T.append(self.tracks[off[0]:off[0]+off[1]])        
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
    



        

        
        
    
    
    
    
    
    
    
    
    
