''' Benchmarks for sphere odf maxima finding '''

import os
from os.path import join as pjoin, dirname
import numpy as np
import dipy.core.reconstruction_performance as rp
import dipy.core.meshes as msh

from numpy.testing import measure

data_path = pjoin(dirname(__file__), '..', 'matrices')
sphere_data = np.load(pjoin(data_path,
                            'evenly_distributed_sphere_362.npz'))
vertices = sphere_data['vertices'].astype(np.float)
n_vertices = vertices.shape[0]
faces16 = sphere_data['faces'].astype(np.uint16)
faces32 = sphere_data['faces'].astype(np.uint32)
less_faces16 = msh.vertinds_faces(np.arange(n_vertices//2),
                                faces16)
less_faces32 = msh.vertinds_faces(np.arange(n_vertices//2),
                                  faces32)
np.random.seed(42)
odf = np.random.uniform(size=(n_vertices,))
sym_vertinds = msh.sym_hemisphere(vertices).astype(np.uint32)
adj = msh.seq_to_objarr(msh.vertinds_to_neighbors(sym_vertinds, faces32))


def bench_maximae():
    print 'ODF maximae'
    print '=' * 10
    opt_time = measure('rp.peak_finding(odf, faces16)', 1000)
    print 'optimized time: %f' % opt_time
    print


if __name__ == '__main__' :
    bench_maximae()
    
