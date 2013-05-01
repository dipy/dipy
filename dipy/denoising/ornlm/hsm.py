import math
import numpy as np
from wavelet import dwt3D
from wavelet import idwt3D
def hsm(fimau, fimao):
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

#                          Details on Wavelet mixing                         
#***************************************************************************
 #  The hard wavelet subbands mixing is described in:                      *
 #                                                                         *
 #  P. Coupe, S. Prima, P. Hellier, C. Kervrann, C. Barillot.              *
 #  3D Wavelet Sub-Bands Mixing for Image Denoising                        *
 #  International Journal of Biomedical Imaging, 2008                      * 
# ***************************************************************************/
    s=fimau.shape;
    p=[0,0,0]
    p[0]=2**math.ceil(math.log(s[0],2))
    p[1]=2**math.ceil(math.log(s[1],2))
    p[2]=2**math.ceil(math.log(s[2],2))
    pad1=np.zeros((p[0],p[1],p[2]));
    pad2=np.zeros((p[0],p[1],p[2]));
    pad1[0:s[0], 0:s[1], 0:s[2]]=fimau[:,:,:];
    pad2[0:s[0], 0:s[1], 0:s[2]]=fimao[:,:,:];
    af = np.array([  [0, -0.01122679215254],
            [0, 0.01122679215254],
            [-0.08838834764832,   0.08838834764832],
            [0.08838834764832,   0.08838834764832],
            [0.69587998903400,  -0.69587998903400],
            [0.69587998903400,   0.69587998903400],
            [0.08838834764832,  -0.08838834764832],
            [-0.08838834764832,  -0.08838834764832],
            [0.01122679215254,                  0],
            [0.01122679215254,                  0]]);
    sf=np.array(af[::-1,:])
    w1=dwt3D.dwt3D(pad1,1,af);
    w2=dwt3D.dwt3D(pad2,1,af);
    w1[0][2] = (w2[0][2]+w1[0][2])/2;
    w1[0][4] = (w2[0][4]+w1[0][4])/2;
    w1[0][5] = (w2[0][5]+w1[0][5])/2;
    w1[0][6] = (w2[0][6]+w1[0][6])/2;
    fima = idwt3D.idwt3D(w1,1,sf);
    fima = fima[:s[0],:s[1],:s[2]];
    return fima
