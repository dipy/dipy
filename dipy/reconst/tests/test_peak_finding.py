import numpy as np
from nose.tools import assert_true, assert_false, assert_equal, assert_almost_equal, assert_raises
from dipy.reconst.recspeed import peak_finding, peaks
from dipy.data import get_sphere


def test_peak_finding():

    vertices,faces=get_sphere('symmetric724')
    odf=np.zeros(len(vertices))
    odf = np.abs(vertices.sum(-1))

    odf[1] = 10.
    odf[505] = 505.
    odf[143] = 143.

    peaks,inds=peak_finding(odf.astype('f8'),faces.astype('uint16'))

    print peaks, inds


if __name__ == '__main__':
    test_peak_finding()

