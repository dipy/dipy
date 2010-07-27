import numpy as np
import dipy as dp
import dipy.io.pickles as pkl


fname='/home/ian/Data/SimData/results_SNR030_1fibre'
#fname='/home/eg01/Data_Backup/Data/Marta/DSI/SimData/results_SNR030_isotropic'


''' file  has one row for every voxel, every voxel is repeating 1000
times with the same noise level , then we have 100 different
directions. 1000 * 100 is the number of all rows.

'''
marta_table_fname='/home/ian/Data/SimData/Dir_and_bvals_DSI_marta.txt'
sim_data=np.loadtxt(fname)
#bvalsf='/home/eg01/Data_Backup/Data/Marta/DSI/SimData/bvals101D_float.txt'

b_vals_dirs=np.loadtxt(marta_table_fname)

bvals=b_vals_dirs[:,0]*1000
gradients=b_vals_dirs[:,1:]

sim_data=sim_data

gq = dp.GeneralizedQSampling(sim_data,bvals,gradients)
tn = dp.Tensor(sim_data,bvals,gradients)
#'''

print tn.evals.shape
print tn.evecs.shape

evals=tn.evals[0]
evecs=tn.evecs[0]

print evecs.shape 


'''

import numpy as np
evecs[:,0]
np.dot(evecs[:,0].T,evecs[:,0])
np.dot(evecs[:,1].T,evecs[:,1])
np.dot(evecs[:,2].T,evecs[:,2])
np.dot(evecs[:,:].T,evecs[:,:])
evecs[:,0]
tn.evecs
first_directions = tn.evecs[:,:,0]
first_directions.shape
np.sum(first_direction,axis=0)
np.sum(first_directions,axis=0)
np.sum(first_directions[:1000,:],axis=0)
np.sum(first_directions[:1000,:],axis=0)[2]
np.sum(first_directions[:1000,:],axis=0)[2]**2
plot(first_directions[:1000,0])
plot(first_directions[:1000,1])
plot(first_directions[:1000,2])
plot(first_directions[:1000,0]**2)
_ip.magic("clear ")
plot(first_directions[:1000,0]**2)
plot(first_directions[:1000,1]**2)
plot(first_directions[:1000,2]**2)
plot(first_directions[:1000,2]**2)
first1000 = first_directions[:1000,:]
cross = np.dot(first1000.T,first1000)
cross
np.trace(cross)
import scipy as sp
np.linalg.eig(cross)
_ip.magic("history -n")

'''
    
    
