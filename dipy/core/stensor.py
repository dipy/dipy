class sltensor():
    ''' Calculate a single tensor with linear list squares fitting model

    Calculate tensors from a 4d numpy array and return an FA image
    and much more using linear list squares fitting.
    bvals and bvecs must be provided as well.  FA calculated from Mori
    et.al, Neuron 2006 . See also David Tuch PhD thesis p. 64 and Mahnaz Maddah thesis p. 44 for the tensor derivation.
    
    What this algorithm does? Solves a system of equations for every voxel j
    
    g0^2*d00 + g1^2*d11+g2^2*d22+ 2*g1*g0*d01+ 2*g0*g2*d02+2*g1*g2*d12 = - ln(S_ij/S0_j)/b_i
    
    where b_i the current b-value and g_i=[g0,g1,g2] the current gradient direction. dxx are the values of 
    the symmetric matrix D. dxx are also the unknown variables.
    
    D=[[d00 ,d01,d02],[d01,d11,d12],[d02,d12,d22]]

    Examples:
    ---------

    
         
    '''


    def __init__(b,g):

        A=[]
        
        for i in range(1,len(b)):

            if not b[i]==0.:

                g0,g1,g2=g[i]
                A.append(np.array([g0*g0,g1*g1,g2*g2,2*g0*g1,2*g0*g2,2*g1*g2],shape=(1,3)))

        self.A=np.concatenate(A)
        self.b=b
        self.g=g


    def fit(data):              

        d=data[0]; s0=d[0]; s=d[1:]
        
        d=np.log(s/s0)

        
                        
        pass

    def evaluate():

        pass

    @property
    def fa():
        pass

    @property
    def adc():
        pass


    





def simpletensor(arr,bvals,bvecs,S0ind,Sind,thr=50.0): 
    ''' Calculate tensors from a 4d numpy array and return an FA image and much more.
    bvals and bvecs must be provided as well.

    FA calculated from Mori et.al, Neuron 2006
    
    See also David Tuch PhD thesis p. 64 and Mahnaz Maddah thesis p. 44 for the tensor derivation.
    
    What this algorithm does? Solves a system of equations for every voxel j
    
    g0^2*d00 + g1^2*d11+g2^2*d22+ 2*g1*g0*d01+ 2*g0*g2*d02+2*g1*g2*d12 = - ln(S_ij/S0_j)/b_i
    
    where b_i the current b-value and g_i=[g0,g1,g2] the current gradient direction. dxx are the values of 
    the symmetric matrix D. dxx are also the unknown variables.
    
    D=  [[d00 ,d01,d02],
             [d01,d11,d12],
             [d02,d12,d22]]

    Parameters:
    -----------

    Returns:
    --------
    
    '''
    if arr.ndim!=4:
        print('Please provide a 4d numpy array as arr here')
        return      
 
    B=bvals[Sind].astype('float32')
    G=bvecs[Sind].astype('float32')
    
    print 'B.shape',B.shape
    print 'G.shape',G.shape

    arsh=arr.shape
    volshape=(arsh[0],arsh[1],arsh[2])
    
    voxno=arsh[0]*arsh[1]*arsh[2]
    
    directionsno=len(Sind)
 
    arr=arr.astype('float32')

    #A=sp.zeros((directionsno,6))
        
    #FA=sp.zeros(volshape,dtype='float32')
    #msk=sp.zeros(volshape,dtype='float32')    
    
    S=arr[:,:,:,Sind]    
    S0=arr[:,:,:,S0ind]
        
    S0[S0<thr]=0.0
    
    print 'S.shape',S.shape
    print 'S0.shape',S0.shape
           
    S=S.reshape(voxno,directionsno)    
    S0=S0.reshape(voxno)
    
    S=S.transpose()    
    #S[S<1.0]=1.0
    #S0[S0<1.0]=1.0
    
    print '#voxno',voxno
    print '#directionsno',directionsno
    print '#S.shape',S.shape
    print '#S0.shape',S0.shape
        
    #S[S<thr]=0
    
    S=sp.log(S/S0)
    
    print 'S.shape',S.shape
    print 'S0.shape',S0.shape    

    S=S.transpose()
    print 'S.shape',S.shape
    print 'S0.shape',S0.shape

    #Remove NaNs (0/0) and Inf (very small numbers in log)
    S[sp.isnan(S)]=0
    S[sp.isinf(S)]=0
    

    S=-S/B
    
    itrG=iter(G)
    #itrA=iter(A)
       
    A=[] # this variable will hold the matrix of the Ax=S system  which we will solve for every voxel
     
    while True:
        
        try:
            g=itrG.next()        
            #g1,g2,g3=g[0],g[1],g[2]        
            #A.append(sp.array([g1*g1,g2*g2,g3*g3,2*g1*g2,2*g1*g3,2*g2*g3]))
            A.append(sp.array([g[0]*g[0],g[1]*g[1],g[2]*g[2],2*g[0]*g[1],2*g[0]*g[2],2*g[1]*g[2]]))    
            
        except StopIteration:
            A=sp.array(A)
            break
        
    print 'A.shape',A.shape
    print 'S.shape',S.shape
    print 'S0.shape',S0.shape
    
    S=S.transpose()
    
    #Remove NaNs (0/0) and Inf (very small numbers in log)
    #S[sp.isnan(S)]=1
    #S[sp.isinf(S)]=1
    
    d,resids,rank,sing=la.lstsq(A,S)
    
    print 'd.shape',d.shape
    
    d=d.transpose()
    
    print 'd.shape',d.shape
        
    itrd=iter(d)
    
    tensors=[]
    
    while True:
        
        try:
        
            d00,d11,d22,d01,d02,d12=itrd.next()
            #print x0,x1,x2,x3,x4,x5
            
            D=sp.array([[d00, d01, d02],[d01,d11,d12],[d02,d12,d22]])
                            
            evals,evecs=la.eigh(D)
                
            l1=evals[0]; l2=evals[1]; l3=evals[2]
                 
            FA=sp.sqrt( ( (l1-l2)**2 + (l2-l3)**2 + (l3-l1)**2 )/( 2*(l1**2+l2**2+l3**2) )  )                       
            
            #tensors.append(sp.array([l1,l2,l3,evecs[0,0],evecs[1,0],evecs[2,0],evecs[0,1],evecs[1,1],evecs[2,1],evecs[0,2],evecs[1,2],evecs[2,2],FA]))
            tensors.append([l1,l2,l3,evecs[0,0],evecs[1,0],evecs[2,0],evecs[0,1],evecs[1,1],evecs[2,1],evecs[0,2],evecs[1,2],evecs[2,2],FA])
            
        except StopIteration:
            
            tensors=sp.array(tensors)
               
            break
    
    tensors[sp.isnan(tensors)]=0
    tensors[sp.isinf(tensors)]=0
    
    tensors=tensors.reshape((arsh[0],arsh[1],arsh[2],13))
    
    print 'tensors.shape:', tensors.shape
        
    return tensors      
