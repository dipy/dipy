''' Benchmarks for Zhang metrics '''

import os
from os.path import join as pjoin
import numpy as np
import dipy.io.trackvis as tv
import dipy.core.track_performance as pf
import dipy.core.track_metrics as tm

from numpy.testing import measure

_data_path = pjoin(os.path.dirname(__file__), 'data')
_track_tuples, _ = tv.read(pjoin(_data_path, 'tracks300.trk.gz'))
tracks300 = [_t[0] for _t in _track_tuples]


def bench_zhang():
    print 'Zhang min'
    print '=' * 10
    #ref_time = measure('tm.most_similar_track_zhang(tracks300)')
    #print 'reference time: %f' % ref_time
    opt_time = measure('pf.most_similar_track_zhang(tracks300)')
    print 'optimized time: %f' % opt_time
    print

def bench_cut_plane():
    print 'Cut plane'
    print '=' * 10
    opt_time = measure('pf.cut_plane(tracks300, tracks300[0])')
    print 'optimized time: %f' % opt_time
    print

def bench_mdl_traj():
    t=np.concatenate(tracks300)
    #t=tracks300[0]
    
    print 'MDL traj'
    print '=' * 10
    opt_time = measure('pf.approximate_mdl_trajectory(t)')
    #opt_time = measure('tm.approximate_trajectory_partitioning(t)')
    #opt_time= measure('tm.minimum_description_length_unpartitoned(t)')
    print 'optimized time: %f' % opt_time
    print
   
    

if __name__ == '__main__' :
    #bench_zhang()
    #bench_cut_plane()
    bench_mdl_traj()
    '''
    import pstats, cProfile
    
    cProfile.runctx("bench_mdl_traj()", globals(), locals(), "Profile.prof")

    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()
    '''
    
