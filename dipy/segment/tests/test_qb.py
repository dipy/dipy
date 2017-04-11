import numpy as np
import nibabel as nib
from nose.tools import (assert_true, assert_false, assert_equal,
                        assert_almost_equal, assert_raises)
from numpy.testing import assert_array_equal, assert_array_almost_equal
from dipy.data import get_data
from dipy.segment.quickbundles import QuickBundles


def test_qbundles():
    streams, hdr = nib.trackvis.read(get_data('fornix'))
    T = [s[0] for s in streams]
    Trk = np.array(T, dtype=np.object)
    qb = QuickBundles(T, 10., 12)
    Tqb = qb.virtuals()
    # Tqbe,Tqbei=qb.exemplars(T)
    Tqbe, Tqbei = qb.exemplars()
    assert_equal(4, qb.total_clusters)
