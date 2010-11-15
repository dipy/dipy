''' Some ideas for the dipy format
'''

import numpy as np

class Dpy():

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
    



if __name__ == '__main__':
    

    a=2*np.ones((4,3)).astype(np.float32)
    b=3*np.ones((4,3)).astype(np.float32)
    c=4*np.ones((4,3)).astype(np.float32)
    d=5*np.ones((4,3)).astype(np.float32)
                
    T=[a,b,c,d]
    print('>>> Writing file on the fly <<<')
    fname='test.dpy'
    dpw=Dpy(fname,'w')
    for t in T:
        dpw.write(t)    
    dpw.close()        
    print('>>> Reading file one array each time <<<')
    dpr=Dpy(fname,'r')        
    while True:         
        a=dpr.read()
        if a!=None: 
            print(a)
        else:
            break
    dpr.close()    
    print('>>> Reading directly with indexes from the disk <<<')
    dpi=Dpy(fname,'r')
    indices=[0,1,2,0,1,0]
    print dpi.read_indexed(indices)
    
        
   


        

        
        
    
    
    
    
    
    
    
    
    
