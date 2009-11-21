''' Benchmarks for Zhang metrics '''

import os
from os.path import join as pjoin

import dipy.io.trackvis as tv
import dipy.core.performance as pf
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


if __name__ == '__main__' :
    bench_zhang()
