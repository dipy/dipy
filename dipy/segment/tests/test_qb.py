import nibabel as nib
from nose.tools import assert_equal
from dipy.data import get_data
from dipy.segment.quickbundles import QuickBundles


def test_qbundles():
    streams, hdr = nib.trackvis.read(get_data('fornix'))
    T = [s[0] for s in streams]
    qb = QuickBundles(T, 10., 12)
    qb.virtuals()
    qb.exemplars()
    assert_equal(4, qb.total_clusters)
