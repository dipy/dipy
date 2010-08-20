#Calculate joint histogram and related metrics
from math import sin,cos
import numpy as np
from scipy.ndimage import affine_transform
from scipy.ndimage.interpolation import rotate,shift,zoom
from scipy.optimize import fmin as fmin_simplex, fmin_powell, fmin_cg

def joint_histogram(A,B,binA,binB):
    ''' Calculate joint histogram and individual histograms for A and B
    ndarrays

    Parameters
    ----------
    A, B: ndarrays
    binA, binB: 1d arrays with the bins    

    Returns
    -------
    JH: joint histogram
    HA: histogram for A
    HB: histogram for B

    Example
    -------
    >>> A=np.array([[1,.5,.2,0,0],[.5,1,.5,0,0],[.2,.5,1,0,0],[0,0,0,0,0],[0,0,0,0,0]])
    >>> B=np.array([[0,0,0,0,0],[0,1,.5,.2,0],[0,.5,1,.5,0],[0,.2,.5,1,0],[0,0,0,0,0]])
    >>> bin_A=np.array([-np.Inf,.1,.35,.75,np.Inf])
    >>> bin_B=np.array([-np.Inf,.1,.35,.75,np.Inf])
    >>> JH,HA,HB=joint_histogram(A,B,bin_A,bin_B)
    
    '''    

    A=A.ravel()
    B=B.ravel()
    A2=A.copy()
    B2=B.copy()
    
    #assign bins
    for i in range(1,len(binA)):
        Ai=np.where(np.bitwise_and(A>binA[i-1],A<=binA[i]))
        A2[Ai]=i-1
    for i in range(1,len(binB)):
        Bi=np.where(np.bitwise_and(B>binB[i-1],B<=binB[i]))
        B2[Bi]=i-1
    JH=np.zeros((len(binA)-1,len(binB)-1))
    #calculate joint histogram
    for i in range(len(A)):
        JH[A2[i],B2[i]]+=1
    #calculate histogram for A
    HA=np.zeros(len(binA)-1)
    for i in range(len(A)):
        HA[A2[i]]+=1    
    #calculate histogram for B
    HB=np.zeros(len(binB)-1)
    for i in range(len(B)):
        HB[B2[i]]+=1       
    return JH,HA,HB


def mutual_information(A,B,binA,binB):
    ''' Calculate mutual information for A and B
    '''
    JH,HA,HB=joint_histogram(A,B,binA,binB)
    N=float(len(A.ravel()))    
    MI=np.zeros(JH.shape)
    #print N    
    for i in range(JH.shape[0]):
        for j in range(JH.shape[1]):
            Pij= JH[i,j]/N
            Pi = HA[i]/N
            Pj=  HB[j]/N
            #print i,j, Pij, Pi, Pj, JH[i,j], HA[i], HB[j]                 
            MI[i,j]=Pij*np.log2(Pij/(Pi*Pj))
    MI[np.isnan(MI)]=0
    return MI.sum()

def correlation_ratio(data):
    ''' Calculate correlation ratio for data

    '''
    pass

    '''

    ntimepoints = data.shape[0]
    correlations = np.ones((ntimepoints, ntimepoints))
    for i in range(ntimepoints):
        unique_values = np.unique(data[i])
        for k in xrange(unique_values.size):
            iso_indices = (data[i] == unique_values[k])
            iso_N = np.sum(iso_indices)
            iso_var = data[:, iso_indices].var(axis=1)
            correlations[i, :] += iso_var * iso_N

    for j in range(ntimepoints):
        correlations[:, j] /= data[j].var()

    correlations = 1 - (1.0 / data[i].size) * correlations
    return correlations
    '''


def transform(A,T,order=0):
    ''' Transform A by an affine transformation T 
    '''
    if A.ndim==2:#rigid 2d        
        th=T[0]        
        P=T[1:]
        AR=rotate(A,np.rad2deg(th),reshape=False,order=0)
        R=np.eye(2)        
        AT=affine_transform(AR,R,offset=P,order=order)
    else:
        print('only 2d arrays for now')
    return AT

def objective_mi(T,A,B,binA,binB,order=3):
    ''' Objective function for mutual information
    '''    
    AT=transform(A,T,order=order)
    #AT=np.round(AT)    
    NegMI= -mutual_information(AT,B,binA,binB)
    print T, NegMI    
    return NegMI


def register(A,B,binA,binB,guess,xtol=0.1,ftol=0.01):
    ''' Register source A to target B using powell's method
    '''
    res=fmin_powell(objective_mi,x0=guess,args=(A,B,binA,binB),xtol=xtol,ftol=ftol)
    return transform(A,res),res

if __name__ == '__main__':

    '''
    A=np.array([[1,.5,.2,0,0],[.5,1,.5,0,0],[.2,.5,1,0,0],[0,0,0,0,0],[0,0,0,0,0]])
    B=np.array([[0,0,0,0,0],[0,1,.5,.2,0],[0,.5,1,.5,0],[0,.2,.5,1,0],[0,0,0,0,0]])
    
    binA=np.array([-np.Inf,.1,.35,.75,np.Inf])
    binB=np.array([-np.Inf,.1,.35,.75,np.Inf])

    '''

    import Image
    pic='/home/eg01/Desktop/brain.jpg'
    #pic='/home/eg01/Desktop/alpha.png'
    imgA=Image.open(pic)    
    imgA=imgA.resize((100,100))
    A=np.array(imgA).astype('float32')
    A=(A[:,:,0]+A[:,:,1]+A[:,:,2])/3.
    
    imgB=imgA.copy()    
    B=np.array(imgB.rotate(90)).astype('float32')
    B=(B[:,:,0]+B[:,:,1]+B[:,:,2])/3.

    #A=A[:800,:800]
    #B=B[:800,:800]
    
    binA=np.r_[-np.inf,np.linspace(A.min(),A.max(),30),np.inf]
    binB=np.r_[-np.inf,np.linspace(B.min(),B.max(),30),np.inf]    
    
    if A.ndim==2:
        guess=np.array([0,0,0])

    print A.shape
    print B.shape    

    '''
    
    AR,trans=register(A,B,binA,binB,guess)   
    
    print 'guess'
    print guess  
    
    print 'A'
    print A
    print 'B'    
    print B
    print 'AR'
    print AR
    print 'trans'
    print trans


    figure(1);imshow(A)
    figure(2);imshow(B)
    figure(3);imshow(AR)

    '''
    

    

    




    


