import numpy as np
import dipy as dp


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

print tn.evals.shape()
print tn.evecs.shape()

evals=tn.evals[0]
evecs=tn.evecs[0]

 



    
    
