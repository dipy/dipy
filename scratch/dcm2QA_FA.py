import time
import numpy as np
from dipy.io import dicomreaders as dcm
import dipy.core.generalized_q_sampling as gq
import dipy.core.dti as st


dname = '/home/eg01/Data_Backup/Data/Eleftherios/CBU090134_METHODS/20090227_154122/Series_003_CBU_DTI_64D_iso_1000'

t1=time.clock()

data,affine,bvals,gradients=dcm.read_mosaic_dir(dname)

t2=time.clock()

print t2-t1, 'secs'

print data.shape

x,y,z,g = data.shape

#calculate QA
#gqs = gq.GeneralizedQSampling(data[:,:,25:30,:],bvals,gradients)

gqs = gq.GeneralizedQSampling(data[:,:,0:32,:],bvals,gradients)


t3=time.clock()

print t3-t2, 'secs'

'''

ten=st.tensor(data,gradients.T,bvals)

t4=time.clock()

print t4-t3, 'secs'

'''
