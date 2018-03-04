import numpy as np
import nibabel as nib
from numpy.testing import assert_array_equal, assert_array_almost_equal
from dipy.data import get_data
from dipy.segment.bundles import RecoBundles



def test_recobundles():
    trkfile = nib.streamlines.load(get_data('fornix'))
    fornix = trkfile.streamlines
