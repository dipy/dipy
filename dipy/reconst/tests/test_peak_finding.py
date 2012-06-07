import numpy as np
from nose.tools import assert_true, assert_false, assert_equal, assert_almost_equal, assert_raises
from reconst.recspeed import peak_finding, peak_finding_onedge
from dipy.data import get_sphere


def test_peak_finding():

    vertices,faces=get_sphere('symmetric724')
    odf=np.zeros(len(vertices))
    odf[1]=10.

    peaks,inds=peak_finding(odf.astype('f8'),faces.astype('uint16'))

    print peaks, inds

