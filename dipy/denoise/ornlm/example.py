import numpy as np
import ornlm_module as ornml
from hsm import hsm
from ascm import ascm
from dipy.data import fetch_stanford_hardi
from dipy.data import read_stanford_hardi
fetch_stanford_hardi()

img, gtab = read_stanford_hardi()
data = img.get_data()
S0 = data[..., 0].astype(np.float64)
mv=np.max(S0)
#Note: in P. Coupe et al. the ricial noise was simulated as 
#sqrt((f+x)^2 + (y)^2) where f is the pixel value and x and y are 
#independent realizations of a random variable with Normal distribution, 
#with mean=0 and standard deviation=h. The user must tune the 'h' parameter 
#taking that into consideration
h=0.01*mv
f1=np.array(ornml.ornlm_pyx(S0, 3, 1, h))
f2=np.array(ornml.ornlm_pyx(S0, 3, 2, h))
fhsm=hsm(f1,f2)
filterd=ascm(S0,f1,f2,h)#this is reported to have the highest SNR in synthetic experiments
