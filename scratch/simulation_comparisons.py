import os
import numpy as np
import dipy as dp
import dipy.io.pickles as pkl
import scipy as sp
from matplotlib.mlab import find
import dipy.core.sphere_plots as splots
import dipy.core.sphere_stats as sphats
import dipy.core.geometry as geometry
import get_vertices as gv

fname='/home/ian/Data/SimData/results_SNR030_1fibre'
#fname='/home/eg01/Data_Backup/Data/Marta/DSI/SimData/results_SNR030_isotropic'

sim_data=np.loadtxt(fname)

''' file  has one row for every voxel, every voxel is repeating 1000
times with the same noise level , then we have 100 different
directions. 1000 * 100 is the number of all rows.

'''

marta_table_fname='/home/ian/Data/SimData/Dir_and_bvals_DSI_marta.txt'
#bvalsf='/home/eg01/Data_Backup/Data/Marta/DSI/SimData/bvals101D_float.txt'

b_vals_dirs=np.loadtxt(marta_table_fname)

bvals=b_vals_dirs[:,0]*1000
gradients=b_vals_dirs[:,1:]

#splots.plot_sphere(gradients, 'Marta DSI gradients')

v = gv.get_vertex_set('dsi102')

#splots.plot_sphere(v, 'dsi102 axes')

gqfile = '/home/ian/Data/SimData/gq_SNR030_1fibre.pkl'
gq = pkl.load_pickle(gqfile)

'''
gq.IN               gq.__doc__          gq.glob_norm_param
gq.QA               gq.__init__         gq.odf              
gq.__class__        gq.__module__       gq.q2odf_params
'''

tnfile = '/home/ian/Data/SimData/tn_SNR030_1fibre.pkl'
tn = pkl.load_pickle(tnfile)

'''
tn.ADC               tn.__init__          tn._getevals
tn.B                 tn.__module__        tn._getevecs
tn.D                 tn.__new__           tn._getndim
tn.FA                tn.__reduce__        tn._getshape
tn.IN                tn.__reduce_ex__     tn._setevals
tn.MD                tn.__repr__          tn._setevecs
tn.__class__         tn.__setattr__       tn.adc
tn.__delattr__       tn.__sizeof__        tn.evals
tn.__dict__          tn.__str__           tn.evecs
tn.__doc__           tn.__subclasshook__  tn.fa
tn.__format__        tn.__weakref__       tn.md
tn.__getattribute__  tn._evals            tn.ndim
tn.__getitem__       tn._evecs            tn.shape
tn.__hash__          tn._getD             
'''

''' file  has one row for every voxel, every voxel is repeating 1000
times with the same noise level , then we have 100 different
directions. 100 * 1000 is the number of all rows.
'''

def analyze_maxima(max_dirs):

    results = []


    for direction in range(100):

        batch = max_dirs[direction,:,:]

        c,b = sphats.eigenstats(batch)

        results.append(np.concatenate((c,b)))

    return results

#dt_first_directions = tn.evecs[:,:,0].reshape((100,1000,3))
# these are the principal direcections for the full set of simulations


eds=np.load(os.path.join(os.path.dirname(dp.__file__),'core','matrices','evenly_distributed_sphere_362.npz'))

odf_vertices=eds['vertices']

dt_first_directions_in=odf_vertices[tn.IN]

print dt_first_directions_in.shape

results = analyze_maxima(dt_first_directions_in.reshape((100,1000,3)))

#for gqi see example dicoms_2_tracks gq.IN[:,0]

np.set_printoptions(precision=6, suppress=True)

out = open('newtable.txt','w')

print >> out, np.vstack(results)

out.close()


    #up = dt_batch[:,2]>= 0

    #splots.plot_sphere(dt_batch[up], 'batch '+str(direction))

    #splots.plot_lambert(dt_batch[up],'batch '+str(direction), centre)
    
    #spread = gq.q2odf_params e,v = np.linalg.eigh(np.dot(spread,spread.transpose())) effective_dimension = len(find(np.cumsum(e) > 0.05*np.sum(e))) #95%

    #rotated = np.dot(dt_batch,evecs)

    #rot_evals, rot_evecs =  np.linalg.eig(np.dot(rotated.T,rotated)/rotated.shape[0])

    #eval_order = np.argsort(rot_evals)

    #rotated = rotated[:,eval_order]

    #up = rotated[:,2]>= 0

    #splot.plot_sphere(rotated[up],'first1000')

    #splot.plot_lambert(rotated[up],'batch '+str(direction))

