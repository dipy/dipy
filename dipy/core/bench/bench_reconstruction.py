''' Benchmarks for sphere odf maxima finding '''

import os
from os.path import join as pjoin, dirname
import numpy as np
import dipy.core.reconstruction_performance as rp

from numpy.testing import measure

data_path = pjoin(dirname(__file__), '..', 'matrices')
sphere_data = np.load(pjoin(data_path,
                            'evenly_distributed_sphere_362.npz'))
faces = sphere_data['faces']
n_vertices = sphere_data['vertices'].shape[0]
np.random.seed(42)
odf = np.random.uniform(size=(n_vertices,))


def bench_maximae():
    print 'ODF maximae'
    print '=' * 10
    opt_time = measure('rp.peak_finding(odf, faces)')
    print 'optimized time: %f' % opt_time
    print


if __name__ == '__main__' :
    bench_maximae()
    
