import math
import numpy as np
from wavelet import dwt3D
from wavelet import idwt3D
def ascm(ima,fimau,fimao,h):
# Pierrick Coupe - pierrick.coupe@gmail.com                                  
# Jose V. Manjon - jmanjon@fis.upv.es                                        
# Addapted to Python by Omar Ocegueda - jomaroceguedag@gmail.com
# Brain Imaging Center, Montreal Neurological Institute.                     
# Mc Gill University                                                         
#                                                                            
# Copyright (C) 2008 Pierrick Coupe and Jose V. Manjon                       

#****************************************************************************
#              3D Adaptive Multiresolution Non-Local Means Filter           *
#            P. Coupe a, J. V. Manjon, M. Robles , D. L. Collin             * 
#****************************************************************************
    s=fimau.shape;
    p=[0,0,0]
    p[0]=2**math.ceil(math.log(s[0],2))
    p[1]=2**math.ceil(math.log(s[1],2))
    p[2]=2**math.ceil(math.log(s[2],2))
    pad1=np.zeros((p[0],p[1],p[2]));
    pad2=np.zeros((p[0],p[1],p[2]));
    pad3=np.zeros((p[0],p[1],p[2]));
    pad1[:s[0], :s[1], :s[2]]=fimau[:,:,:]
    pad2[:s[0], :s[1], :s[2]]=fimao[:,:,:]
    pad3[:s[0], :s[1], :s[2]]=ima[:,:,:]
    af = np.array([  [0, -0.01122679215254],
            [0, 0.01122679215254],
            [-0.08838834764832,   0.08838834764832],
            [0.08838834764832,   0.08838834764832],
            [0.69587998903400,  -0.69587998903400],
            [0.69587998903400,   0.69587998903400],
            [0.08838834764832,  -0.08838834764832],
            [-0.08838834764832,  -0.08838834764832],
            [0.01122679215254,                  0],
            [0.01122679215254,                  0]])
    sf=np.array(af[::-1,:])
    w1=dwt3D.dwt3D(pad1,1,af)
    w2=dwt3D.dwt3D(pad2,1,af)
    w3=dwt3D.dwt3D(pad3,1,af)
    #BayeSkrink for Coeff mixing
    for i in xrange(7):
        tmp = np.array(w3[0][i])
        tmp = tmp[:(s[0]//2), :(s[1]//2), :(s[2]//2)]
        sigY = np.std(tmp, ddof=1)
        sigX = (sigY*sigY) - h*h
        if sigX<0:
            T=abs(w3[0][i]).max()
        else:
            T=(h*h)/(sigX**0.5)
        w3[0][i]=abs(w3[0][i])
        dist=np.array(w3[0][i])-T
        dist=np.exp(-0.01*dist)
        dist=1./(1+dist)
        w3[0][i]=dist*w1[0][i] + (1-dist)*w2[0][i]
    w3[1]=w1[1]
    fima=idwt3D.idwt3D(w3,1,sf)
    fima=fima[:s[0], :s[1], :s[2]]
    return fima
