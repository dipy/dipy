from scipy.io import loadmat
import numpy as np
#?loadmat

phantom=loadmat('/home/eg309/Desktop/phantom_test_data.mat',struct_as_record=True)
all=phantom['all']
b_table=phantom['b_table']
odf_vertices=phantom['odf_vertices']
odf_faces=phantom['odf_faces']
s = all[14,14,1]
l_values=np.sqrt(b_table[0]*0.01506)
tmp=np.tile(l_values,(3,1))
b_vector=b_table[1:4,:]*tmp
q2odf=np.sinc(np.dot(b_vector.T, odf_vertices) * 1.2/np.pi)
odf=np.dot(s,q2odf)







