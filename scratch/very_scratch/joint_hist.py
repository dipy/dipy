#Calculate joint histogram and related metrics
from math import sin,cos,pi
import numpy as np
from scipy.ndimage import affine_transform, geometric_transform
from scipy.ndimage.interpolation import rotate,shift,zoom
from scipy.optimize import fmin as fmin_simplex, fmin_powell, fmin_cg
from scipy.optimize import leastsq
from dipy.core import geometry as gm

import pylab

def affine_transform2d(I,M):
    ''' Inspired by the work of Alexis Roche and the independent work of D. Kroon

    Parameters
    ----------
    I: array, shape(N,M), 2d image
    M: inverse transformation matrix 3x3, array, shape (3,3)
    mode:
      0: linear interpolation and outside pixels set to nearest pixel

    Returns
    -------
    Iout: array, shape(N,M), transformed image
      
    '''
    #the transpose is for contiguous C arrays (default)
    #I=I.T
    
    #create all x,y indices
    xy=np.array([(i,j) for (i,j) in np.ndindex(I.shape)])
        
    #image center is now our origin (0,0)
    mean=np.array(I.shape)/2.
    mean=mean.reshape(1,2)
    xyd=xy-mean   

    #transformed coordinates    
    lxy = mean.T + np.dot(M[:2,:2],xyd.T) + M[:2,2].reshape(2,1)
    lxy=lxy.T
    
    #neighborh pixels for linear interp
    bas0=np.floor(lxy)
    bas1=bas0+1

    #linear interp. constants
    com=lxy-bas0
    perc0=(1-com[:,0])*(1-com[:,1])
    perc1=(1-com[:,0])*com[:,1]
    perc2=com[:,0]*(1-com[:,1])
    perc3=com[:,0]*com[:,1]

    #create final image
    # Iout=np.zeros(I.shape)

    #zeroing indices outside boundaries

    check_xbas0=np.where(np.bitwise_or(bas0[:,0]<0,bas0[:,0]>=I.shape[0]))
    check_ybas0=np.where(np.bitwise_or(bas0[:,1]<0,bas0[:,1]>=I.shape[1]))    
    
    bas0[check_xbas0,0]=0
    bas0[check_ybas0,1]=0

    check_xbas1=np.where(np.bitwise_or(bas1[:,0]<0,bas1[:,0]>=I.shape[0]))
    check_ybas1=np.where(np.bitwise_or(bas1[:,1]<0,bas1[:,1]>=I.shape[1]))
    
    bas1[check_xbas1,0]=0
    bas1[check_ybas1,1]=0

    #hold shape
    Ish=I.shape[0]
    #ravel image
    Ione=I.ravel()
    
    #new intensities
    xyz0=Ione[(bas0[:,0]+bas0[:,1]*Ish).astype('int')]
    xyz1=Ione[(bas0[:,0]+bas1[:,1]*Ish).astype('int')]
    xyz2=Ione[(bas1[:,0]+bas0[:,1]*Ish).astype('int')]
    xyz3=Ione[(bas1[:,0]+bas1[:,1]*Ish).astype('int')]

    #kill mirroring
    #xyz0[np.bitwise_or(check_xbas0,check_ybas0)]=0    
    #xyz1[np.bitwise_or(check_xbas0,check_ybas1)]=0
    #xyz2[np.bitwise_or(check_xbas1,check_ybas0)]=0
    #xyz3[np.bitwise_or(check_xbas1,check_ybas1)]=0
        
    #apply recalculated intensities
    Iout=xyz0*perc0+xyz1*perc1+xyz2*perc2+xyz3*perc3

    return Iout.reshape(I.shape)


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

    
def apply_mapping(A,T,order=0,map_type='affine2d'):
    ''' Apply mapping
    '''
    
    if map_type=='affine2d':

        #create the different components
        #translation[2], scale[2], rotation[1], shear[2]

        if len(T)==7:            
            tc1,tc2,sc1,sc2,rc,sch1,sch2=T
            
        if len(T)==5:
            tc1,tc2,sc1,sc2,rc=T
            sch1,sch2=(0,0)

        if len(T)==4:
            tc1,tc2,rc,sc=T
            sc1,sc2,sch1,sch2=(sc,sc,1,1)  
            
        if len(T)==3:
            tc1,tc2,rc=T            
            sc1,sc2,sch1,sch2=(1,1,0,0)            
        
        #translation
        TC=np.matrix([[1,0,tc1],
                      [0,1,tc2],
                      [0,0,  1]])

        #scaling
        SC=np.matrix([[sc1,  0,   0],
                      [0,  sc2,   0],
                      [0,    0,   1]])

        #rotation
        RC=np.matrix([[cos(rc), sin(rc), 0],
                      [-sin(rc), cos(rc), 0],
                      [0      ,       0, 1]])
        
        #shear        
        SHC=np.matrix([[1,   sch1,0],
                       [sch2,   1,0],
                       [0,      0,1]])            
        
        
        #apply
        #M=TC*SC*RC*SHC

        if len(T)==3:
            M=TC*RC
        if len(T)==4:
            M=TC*SC*RC
        if len(T)==5:
            M=TC*SC*RC
        if len(T)==7:
            M=TC*SC*RC*SHC

        M=np.array(M)

        AT=affine_transform2d(A,M)
        
    return AT


def objective_mi(T,A,B,binA,binB,order=0,map_type='affine2d'):
    ''' Objective function for mutual information
    '''
    AT=apply_mapping(A,T,order=0,map_type=map_type)
    #AT=np.round(AT)


    AT=AT.T
    
    NegMI= -mutual_information(AT,B,binA,binB)
    print '====',T,'====> - MI : ',NegMI
    
    #pylab.imshow(AT)    
    #raw_input('Press Enter...')

    

    #pylab.imshow(np.hstack((A,B,AT)))
    #raw_input('Press Enter...')
    
    return NegMI


def objective_sd(T,A,B,order=0,map_type='affine2d'):

    AT=apply_mapping(A,T,order=0,map_type=map_type)

    AT=AT.T
    
    if AT.sum()==0:
        SD=10**15
    else:
        SD= np.sum((AT-B)**2)/np.prod(AT.shape)
    print '====',T,'====>  SD : ',SD

    #pylab.imshow(np.hstack((A,B,AT)))    
    #raw_input('Press Enter...')

    
    return SD

    


def register(A,B,guess,metric='sd',binA=None,binB=None,xtol=0.1,ftol=0.01,order=0,map_type='affine2d'):
    ''' Register source A to target B using modified powell's method

    Powell's method tries to minimize the objective function
    '''
    if metric=='mi':
        finalT=fmin_powell(objective_mi,x0=guess,args=(A,B,binA,binB,order,map_type),xtol=xtol,ftol=ftol)
        #finalT=leastsq(func=objective_mi,x0=np.array(guess),args=(A,B,binA,binB,order,map_type))

    if metric=='sd':        
        finalT=fmin_powell(objective_sd,x0=guess,args=(A,B,order,map_type),xtol=xtol,ftol=ftol)
        #finalT=leastsq(func=objective_sd,x0=np.array(guess),args=(A,B,order,map_type))
    
    return finalT

def evaluate(A,B,guess,metric='sd',binA=None,binB=None,xtol=0.1,ftol=0.01,order=0,map_type='affine2d'):

    #tc1,tc2,sc1,sc2,rc=T

    tc1=np.linspace(-50,50,20)
    tc2=np.linspace(-50,50,20)
    # sc1=np.linspace(-1.2,1.2,10)
    # sc2=np.linspace(-1.2,1.2,10)
    # rc=np.linspace(0,np.pi,8)

    f_min=np.inf

    T_final=[]

    '''
    for c1 in tc1:
        for c2 in tc2:
            for s1 in sc1:
                for s2 in sc2:
                    for r in rc:
                        T=[c1,c2,s1,s2,r]
                        f=objective_sd(T,A,B,order=0,map_type='affine2d')
                        if f<f_min:
                            f_min=f
                            T_final=T
    '''

    for c1 in tc1:
        for c2 in tc2:
            T=[c1,c2,guess[2],guess[3],guess[4]]
            f=objective_sd(T,A,B,order=0,map_type='affine2d')
            if f<f_min:
                f_min=f
                T_final=T
                             
    return T_final


def test(map_type='affine2d',xtol=0.0001,ftol=0.0001):

    import Image
    #pic='/home/eg01/Desktop/brain.jpg'
    #pic='/home/eg01/Desktop/alpha.png'

    pic='/tmp/landmarks1.png'
    #pic='/tmp/lenag2.png'
    imgA=Image.open(pic)    
    #imgA=imgA.resize((100,100))
    imgA=imgA.rotate(25)
    A=np.array(imgA).astype('float32')    
    A=(A[:,:,0]+A[:,:,1]+A[:,:,2])/3.
    
    #A=A.sum(axis=-1)/3.
    
    # imgB=imgA.copy()
    pic2='/tmp/landmarks2.png'
    #pic2='/tmp/lenag2.png'
    
    imgB=Image.open(pic2)
    
    #imgB=imgB.resize(
    #B=np.array(imgB.rotate(90)).astype('float32')
    B=np.array(imgB).astype('float32')    
    B=(B[:,:,0]+B[:,:,1]+B[:,:,2])/3.

    #zero padding

    Z=np.zeros((A.shape[0]+50,A.shape[1]+50))
    Z[25:25+A.shape[0],25:25+A.shape[1]]=A    
    A=Z

    Z2=np.zeros((B.shape[0]+50,B.shape[1]+50))
    Z2[25:25+B.shape[0],25:25+B.shape[1]]=B    
    B=Z2
    
    # binA=np.r_[-np.inf,np.linspace(A.min(),A.max(),30),np.inf]
    # binB=np.r_[-np.inf,np.linspace(B.min(),B.max(),30),np.inf]
    
    # if A.ndim==2:
    #     #guess=np.array([0.,0.,0])
    #     guess=np.array([0.,0.,1.,1.,0])
    #     #translation[2], scale[2], rotation[1], shear[2]
    #     #guess=np.array([0,0,1,1,0,0,0])

    print A.shape
    print B.shape    
    
    #res=register(A,B,guess=guess,metric='sd',xtol=xtol,ftol=ftol,order=0,map_type=map_type)
    #res=register(A,B,guess=guess,metric='mi',binA=binA,binB=binB,xtol=xtol,ftol=ftol,order=0,map_type=map_type)
    
    #res=evaluate(A,B,guess=guess,metric='sd')

    res=[-44.736842105263158, 44.736842105263165, 1.0,1.]#, 1.0, 0.0]

    #res=guess
    res=register(A,B,guess=res,metric='sd',xtol=xtol,ftol=ftol,order=0,map_type=map_type)
        
    print res
    #return A,B,

    AR=apply_mapping(A,res,order=0,map_type=map_type)       

    pylab.imshow(np.hstack((A,B,AR.T)))    
    raw_input('Press Enter...')

    return A,B,AR.T


    
if __name__ == '__main__':

    
    A=np.array([[1,.5,.2,0,0],[.5,1,.5,0,0],[.2,.5,1,0,0],[0,0,0,0,0],[0,0,0,0,0]])
    B=np.array([[0,0,0,0,0],[0,1,.5,.2,0],[0,.5,1,.5,0],[0,.2,.5,1,0],[0,0,0,0,0]])
    
    binA=np.array([-np.Inf,.1,.35,.75,np.Inf])
    binB=np.array([-np.Inf,.1,.35,.75,np.Inf])

    





    


